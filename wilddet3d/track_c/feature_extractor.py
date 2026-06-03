"""Frozen-stack feature extractor for Track C.

Runs the frozen WildDet3D (SAM3 backbone + encoder + decoder + LingBot-Depth)
on one image with one geometric box prompt per tracked object, and returns the
exact tensors Track C trains on top of:

- per-object selected-query hidden states across ALL decoder layers
  ``hidden_states [L, N_obj, 256]`` (deep supervision),
- the selected query's predicted 2D box ``pred_box_2d [N_obj, 4]`` (normalized
  xyxy, model input space) — the reference for re-encoding the temporal box,
- shared per-image ``depth_latents [N_tok, 256]`` and ``ray_embeddings
  [N_tok, 81]`` (the latter is derivable from intrinsics; cached for convenience),
- the model-space intrinsics ``K [3, 3]`` and the IoU/selection bookkeeping.

Mechanism (validated): a geo box prompt yields 200 decoder queries; we pick the
query with max IoU to the prompt box (IoU ~0.98 in practice). This is the
object<->query correspondence — no Hungarian, no decoder seeding.

The frozen stack runs in **eval mode** (so the LingBot-Depth backend takes its
loss-free ``forward_test`` path and never imports ``moge``); a hook on
``sam3._run_decoder`` captures all decoder layers' hidden states (the matching
that needs GT only runs in training mode, which we avoid).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from wilddet3d.data_types import WildDet3DInput
from wilddet3d.inference import _orig_to_input_hw_box, build_model
from wilddet3d.preprocessing import preprocess


def _pairwise_iou_xyxy(a: Tensor, b: Tensor) -> Tensor:
    """a: [M,4], b: [4] (same pixel space). Returns [M]."""
    lt = torch.maximum(a[:, :2], b[:2])
    rb = torch.minimum(a[:, 2:], b[2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[2] - b[0]).clamp(min=0) * (b[3] - b[1]).clamp(min=0)
    return inter / (area_a + area_b - inter).clamp(min=1e-8)


class FrozenFeatureExtractor:
    """Wraps a frozen WildDet3D and extracts Track C features per frame."""

    def __init__(
        self,
        checkpoint: str,
        sam3_checkpoint: Optional[str] = None,
        device: str = "cuda",
        use_depth_input: bool = True,
    ) -> None:
        kwargs = dict(
            checkpoint=checkpoint,
            skip_pretrained=True,
            score_threshold=0.0,
            nms=False,
            device=device,
            use_depth_input_test=use_depth_input,
        )
        if sam3_checkpoint is not None:
            kwargs["sam3_checkpoint"] = sam3_checkpoint
        predictor = build_model(**kwargs)
        self.wd = predictor.wilddet3d
        self.wd.eval()
        for p in self.wd.parameters():
            p.requires_grad_(False)
        self.device = next(self.wd.parameters()).device
        self.use_depth_input = use_depth_input

    @torch.no_grad()
    def extract(
        self,
        image: np.ndarray,
        intrinsics: np.ndarray,
        prompt_boxes_xyxy: Sequence[Sequence[float]],
        depth: Optional[np.ndarray] = None,
    ) -> dict:
        """Extract Track C features for one frame.

        Args:
            image: RGB (H, W, 3) float/uint8 array (original resolution).
            intrinsics: (3, 3) camera intrinsics in original image space.
            prompt_boxes_xyxy: list of N object 2D boxes (pixel xyxy, original
                image space) — e.g. Step-2 mask bounding boxes.
            depth: optional (H, W) metric depth (metres) at image resolution.

        Returns:
            dict with (all CPU tensors):
              hidden_states  [L, N, 256]  selected-query feats per decoder layer
              pred_box_2d    [N, 4]       selected query 2D box, normalized xyxy
              sel_idx        [N]          selected query index (0..199)
              sel_iou        [N]          IoU of selected box to the prompt box
              depth_latents  [N_tok, 256] shared per-image depth latents
              ray_embeddings [N_tok, 81]  shared per-image ray embeddings (or None)
              intrinsics     [3, 3]       model-space intrinsics
              input_hw       (H, W)       model input size
        """
        image = np.asarray(image).astype(np.float32)
        data = preprocess(image, intrinsics.astype(np.float32),
                          depth=depth if (depth is not None and self.use_depth_input) else None)
        ih, iw = data["input_hw"]
        dev = self.device

        # Original-px prompt boxes -> input_hw px -> normalized cxcywh.
        boxes_model_px = [
            _orig_to_input_hw_box(list(b), data["original_hw"], data["padding"], (ih, iw))
            for b in prompt_boxes_xyxy
        ]
        cxcywh = []
        for x1, y1, x2, y2 in boxes_model_px:
            cxcywh.append([(x1 + x2) / 2 / iw, (y1 + y2) / 2 / ih,
                           (x2 - x1) / iw, (y2 - y1) / ih])
        n = len(cxcywh)
        geo = torch.tensor(cxcywh, dtype=torch.float32, device=dev).unsqueeze(1)  # [N,1,4]

        batch_kwargs = dict(
            images=data["images"].to(dev),
            intrinsics=data["intrinsics"][None].to(dev),
            img_ids=torch.zeros(n, dtype=torch.long, device=dev),
            text_ids=torch.zeros(n, dtype=torch.long, device=dev),
            unique_texts=["geometric"],
            geo_boxes=geo,
            geo_boxes_mask=torch.zeros(n, 1, dtype=torch.bool, device=dev),
            geo_box_labels=torch.ones(n, 1, dtype=torch.long, device=dev),
            padding=[data["padding"]],
        )
        if depth is not None and self.use_depth_input:
            batch_kwargs["depth_gt"] = data["depth_gt"].to(dev)
        batch = WildDet3DInput(**batch_kwargs)

        cap: dict = {}

        # Hook the decoder to grab ALL layers' hidden states [L, bs, 200, d].
        orig_run_decoder = self.wd.sam3._run_decoder
        def run_decoder_wrap(*a, **k):
            out, hs = orig_run_decoder(*a, **k)
            cap["hs"] = hs  # [L, bs, num_queries, d] (batch-first per layer)
            return out, hs
        self.wd.sam3._run_decoder = run_decoder_wrap

        # Hook the 3D head pre-forward to grab ray + depth latents (per-prompt).
        def head_pre_hook(module, args, kwargs):
            cap["ray_embeddings"] = args[1] if len(args) > 1 else kwargs.get("ray_embeddings")
            cap["depth_latents"] = args[2] if len(args) > 2 else kwargs.get("depth_latents")
        h_head = self.wd.bbox3d_head.register_forward_pre_hook(head_pre_hook, with_kwargs=True)

        # Wrap _forward_test to grab per-query final 2D boxes.
        orig_ft = self.wd._forward_test
        def ft_wrap(*a, **k):
            cap["pred_boxes_2d"] = k.get("pred_boxes_2d", a[1] if len(a) > 1 else None)
            return orig_ft(*a, **k)
        self.wd._forward_test = ft_wrap

        try:
            self.wd(batch)
        finally:
            self.wd.sam3._run_decoder = orig_run_decoder
            self.wd._forward_test = orig_ft
            h_head.remove()

        hs = cap["hs"]                       # [L, bs=N, 200, d]
        pred2d = cap["pred_boxes_2d"]        # [N, 200, 4] normalized xyxy
        L = hs.shape[0]

        # Select the query per object by IoU to its prompt box (in model px).
        sel_idx = torch.empty(n, dtype=torch.long)
        sel_iou = torch.empty(n, dtype=torch.float32)
        pred2d_px = pred2d.clone()
        pred2d_px[..., 0::2] *= iw
        pred2d_px[..., 1::2] *= ih
        for j in range(n):
            pb = torch.tensor(boxes_model_px[j], dtype=torch.float32, device=pred2d_px.device)
            ious = _pairwise_iou_xyxy(pred2d_px[j], pb)
            k = int(ious.argmax())
            sel_idx[j] = k
            sel_iou[j] = ious[k]

        # Gather selected query across layers: [L, N, d]
        hidden_sel = torch.stack(
            [hs[:, j, sel_idx[j], :] for j in range(n)], dim=1
        )  # [L, N, d]
        box2d_sel = torch.stack(
            [pred2d[j, sel_idx[j]] for j in range(n)], dim=0
        )  # [N, 4] normalized xyxy

        depth_latents = cap["depth_latents"]   # [N, N_tok, 256]; per-prompt copies
        ray = cap.get("ray_embeddings")        # [N, N_tok, 81] or None

        return {
            "hidden_states": hidden_sel.detach().float().cpu(),     # [L, N, 256]
            "pred_box_2d": box2d_sel.detach().float().cpu(),        # [N, 4]
            "sel_idx": sel_idx.cpu(),
            "sel_iou": sel_iou.cpu(),
            "depth_latents": depth_latents[0].detach().float().cpu(),  # [N_tok, 256] (shared)
            "ray_embeddings": (ray[0].detach().float().cpu() if ray is not None else None),
            "intrinsics": data["intrinsics"].float().cpu(),         # [3,3] model space
            "input_hw": (ih, iw),
            "num_layers": L,
        }
