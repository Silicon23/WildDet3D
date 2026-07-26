"""Exact frozen-feature extraction for v2 positive-point WildDet3D tracks.

Unlike the original Track C cache (GT-box prompts), v2 must reproduce the
point-prompted Stage-4 forward exactly:

* one repeated image per positive-point prompt,
* shared category-conditioned ``prompt_text="geometric: <label>"``,
* metric VGGT depth and saved-frame intrinsics,
* Stage-4 score/NMS followed by point-containment proposal routing.

The selected decoder query is recovered from the raw model tensors by
replaying that routing.  The final public prediction is returned as a QA
oracle so cache production can compare it field-for-field with the frozen v2
record before accepting any feature.
"""

from __future__ import annotations

import os
import types
from typing import Optional, Sequence

import numpy as np
import torch
from torch import Tensor
from torchvision.ops import batched_nms

from wilddet3d.data_types import WildDet3DInput
from wilddet3d.inference import (
    _orig_to_input_hw_point,
    build_model,
)
from wilddet3d.preprocessing import preprocess


def patch_predictor_point_batching(predictor) -> None:
    """Apply the exact repeated-image point batching used by v2 Stage 4."""

    def _create_point_batch_patched(
        self,
        images,
        intrinsics,
        points_list,
        input_hw,
        device,
        text="object",
        padding=None,
    ):
        height, width = input_hw
        n_prompts = len(points_list)
        max_points = max(len(points) for points in points_list)
        geo_points = torch.zeros(n_prompts, max_points, 2, device=device)
        geo_point_labels = torch.zeros(
            n_prompts, max_points, dtype=torch.long, device=device
        )
        geo_points_mask = torch.ones(
            n_prompts, max_points, dtype=torch.bool, device=device
        )
        for i, points in enumerate(points_list):
            for j, (x, y, label) in enumerate(points):
                geo_points[i, j] = torch.tensor(
                    [x / width, y / height], device=device
                )
                geo_point_labels[i, j] = int(label)
                geo_points_mask[i, j] = False
        return WildDet3DInput(
            images=images,
            intrinsics=intrinsics,
            img_ids=torch.arange(n_prompts, dtype=torch.long, device=device),
            text_ids=torch.zeros(n_prompts, dtype=torch.long, device=device),
            unique_texts=[text],
            geo_points=geo_points,
            geo_points_mask=geo_points_mask,
            geo_point_labels=geo_point_labels,
            padding=padding,
        )

    predictor._create_point_batch = types.MethodType(
        _create_point_batch_patched, predictor
    )


def _combined_scores(wd, captured: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
    scores_2d = captured["pred_logits"].sigmoid().squeeze(-1)
    pred_conf_3d = captured.get("pred_conf_3d")
    scores_3d = (
        pred_conf_3d.sigmoid().squeeze(-1)
        if pred_conf_3d is not None
        else torch.zeros_like(scores_2d)
    )
    conf_weight = float(wd.eval_3d_conf_weight)
    override = os.environ.get("WILDDET3D_CONF_WEIGHT")
    if override is not None:
        conf_weight = float(override)
    scores = scores_2d + conf_weight * scores_3d if conf_weight > 0 else scores_2d

    presence_logits = captured.get("presence_logits")
    if presence_logits is not None and wd.use_presence_score:
        presence = presence_logits.sigmoid()
        if presence.ndim == 1:
            presence = presence.unsqueeze(-1)
        scores = scores * presence
        scores_2d = scores_2d * presence
    return scores, scores_2d, scores_3d


def _select_raw_queries(
    wd,
    captured: dict[str, Tensor],
    points_model: Sequence[Sequence[tuple[float, float, int]]],
    input_hw: tuple[int, int],
) -> tuple[Tensor, Tensor]:
    """Replay model NMS + predictor point routing; return raw query indices."""
    height, width = input_hw
    boxes_norm = captured["pred_boxes_2d"]
    boxes_px = boxes_norm.clone()
    boxes_px[..., 0::2] *= width
    boxes_px[..., 1::2] *= height
    scores, _, _ = _combined_scores(wd, captured)

    roi = wd.roi2det3d
    score_threshold = float(getattr(roi, "score_threshold", -1.0))
    use_nms = bool(getattr(roi, "nms", False))
    iou_threshold = float(getattr(roi, "iou_threshold", 0.5))
    if os.environ.get("SAM3_SCORE_THRESH") is not None:
        score_threshold = float(os.environ["SAM3_SCORE_THRESH"])
    if os.environ.get("SAM3_NMS") is not None:
        use_nms = os.environ["SAM3_NMS"] == "1"
    if os.environ.get("SAM3_IOU_THRESH") is not None:
        iou_threshold = float(os.environ["SAM3_IOU_THRESH"])

    selected = []
    selected_n_inside = []
    for prompt_index, points in enumerate(points_model):
        raw_indices = torch.arange(
            boxes_px.shape[1], device=boxes_px.device, dtype=torch.long
        )
        candidate_boxes = boxes_px[prompt_index]
        candidate_scores = scores[prompt_index]
        if score_threshold > 0:
            keep_threshold = (
                captured["pred_logits"][prompt_index].sigmoid().squeeze(-1)
                > score_threshold
            )
            raw_indices = raw_indices[keep_threshold]
            candidate_boxes = candidate_boxes[keep_threshold]
            candidate_scores = candidate_scores[keep_threshold]
        if use_nms and raw_indices.numel():
            class_ids = torch.zeros_like(raw_indices)
            keep_nms = batched_nms(
                candidate_boxes, candidate_scores, class_ids, iou_threshold
            )
            raw_indices = raw_indices[keep_nms]
            candidate_boxes = candidate_boxes[keep_nms]
            candidate_scores = candidate_scores[keep_nms]
        if not raw_indices.numel():
            raise RuntimeError(f"no proposal survived for prompt {prompt_index}")

        positives = [(x, y) for x, y, label in points if int(label) == 1]
        if not positives:
            best = int(candidate_scores.argmax())
            n_inside = 0
        else:
            pos = torch.tensor(
                positives, dtype=torch.float32, device=candidate_boxes.device
            )
            x1, y1, x2, y2 = candidate_boxes.unbind(-1)
            inside = (
                (pos[:, 0][None, :] >= x1[:, None])
                & (pos[:, 0][None, :] <= x2[:, None])
                & (pos[:, 1][None, :] >= y1[:, None])
                & (pos[:, 1][None, :] <= y2[:, None])
            )
            counts = inside.sum(dim=1).float()
            n_inside = int(counts.max().item())
            if n_inside <= 0:
                best = int(candidate_scores.argmax())
            else:
                rank = counts + candidate_scores.clamp(0, 1) * 0.999
                best = int(rank.argmax())
        selected.append(raw_indices[best])
        selected_n_inside.append(n_inside)
    return torch.stack(selected), torch.tensor(selected_n_inside, dtype=torch.long)


class V2PointFeatureExtractor:
    """Frozen WildDet3D wrapper that exposes exact selected-query features."""

    def __init__(
        self,
        checkpoint: str,
        sam3_checkpoint: Optional[str] = None,
        device: str = "cuda",
        nms_iou_threshold: float = 0.6,
    ) -> None:
        os.environ.setdefault("XFORMERS_DISABLED", "1")
        try:
            torch.backends.cuda.enable_cudnn_sdp(False)
        except Exception:
            pass
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        kwargs = dict(
            checkpoint=checkpoint,
            skip_pretrained=True,
            score_threshold=0.0,
            score_3d_threshold=0.0,
            nms=True,
            iou_threshold=nms_iou_threshold,
            device=device,
            use_depth_input_test=True,
            use_predicted_intrinsics=False,
        )
        if sam3_checkpoint is not None:
            kwargs["sam3_checkpoint"] = sam3_checkpoint
        self.predictor = build_model(**kwargs)
        patch_predictor_point_batching(self.predictor)
        self.wd = self.predictor.wilddet3d
        self.wd.eval()
        for parameter in self.wd.parameters():
            parameter.requires_grad_(False)
        self.device = next(self.wd.parameters()).device

    @torch.inference_mode()
    def extract(
        self,
        image: np.ndarray,
        intrinsics: np.ndarray,
        points_xy_label: Sequence[Sequence[tuple[float, float, int]]],
        label: str,
        depth_m: np.ndarray,
        amp_dtype: str = "bf16",
    ) -> dict:
        """Run an exact v2 point-prompt group and return CPU cache tensors."""
        if not points_xy_label or any(not points for points in points_xy_label):
            raise ValueError("every prompt needs at least one point")
        image = np.asarray(image).astype(np.float32)
        intrinsics = np.asarray(intrinsics).astype(np.float32)
        depth_m = np.asarray(depth_m).astype(np.float32)
        data = preprocess(image, intrinsics, depth=depth_m)
        height, width = data["input_hw"]
        n_prompts = len(points_xy_label)
        images = data["images"].to(self.device).repeat(n_prompts, 1, 1, 1)
        K = data["intrinsics"].to(self.device)[None].repeat(n_prompts, 1, 1)
        depth_gt = data["depth_gt"].to(self.device).repeat(n_prompts, 1, 1, 1)
        input_hw = [data["input_hw"]] * n_prompts
        original_hw = [data["original_hw"]] * n_prompts
        padding = [data["padding"]] * n_prompts
        points = [
            [(float(x), float(y), int(point_label)) for x, y, point_label in prompt]
            for prompt in points_xy_label
        ]
        points_model = [
            [
                _orig_to_input_hw_point(
                    point, original_hw[i], padding[i], (height, width)
                )
                for point in prompt
            ]
            for i, prompt in enumerate(points)
        ]

        captured: dict[str, Tensor] = {}
        original_decoder = self.wd.sam3._run_decoder
        original_forward_test = self.wd._forward_test

        def decoder_wrapper(*args, **kwargs):
            output, hidden_states = original_decoder(*args, **kwargs)
            captured["hidden_states"] = hidden_states
            return output, hidden_states

        def forward_test_wrapper(*args, **kwargs):
            names = (
                "pred_logits",
                "pred_boxes_2d",
                "pred_boxes_3d",
                "pred_conf_3d",
                "presence_logits",
                "batch",
                "geom_out",
            )
            for index, name in enumerate(names):
                value = kwargs.get(name)
                if value is None and index < len(args):
                    value = args[index]
                if value is not None:
                    captured[name] = value
            return original_forward_test(*args, **kwargs)

        def head_pre_hook(module, args, kwargs):
            captured["ray_embeddings"] = (
                args[1] if len(args) > 1 else kwargs["ray_embeddings"]
            )
            captured["depth_latents"] = (
                args[2] if len(args) > 2 else kwargs["depth_latents"]
            )

        self.wd.sam3._run_decoder = decoder_wrapper
        self.wd._forward_test = forward_test_wrapper
        hook = self.wd.bbox3d_head.register_forward_pre_hook(
            head_pre_hook, with_kwargs=True
        )
        dtype = {
            "none": None,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[amp_dtype]
        try:
            if dtype is None or self.device.type != "cuda":
                result = self.predictor(
                    images=images,
                    intrinsics=K,
                    input_hw=input_hw,
                    original_hw=original_hw,
                    padding=padding,
                    input_points=points,
                    prompt_text=f"geometric: {label}",
                    depth_gt=depth_gt,
                )
            else:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    result = self.predictor(
                        images=images,
                        intrinsics=K,
                        input_hw=input_hw,
                        original_hw=original_hw,
                        padding=padding,
                        input_points=points,
                        prompt_text=f"geometric: {label}",
                        depth_gt=depth_gt,
                    )
        finally:
            self.wd.sam3._run_decoder = original_decoder
            self.wd._forward_test = original_forward_test
            hook.remove()

        selected, n_inside = _select_raw_queries(
            self.wd, captured, points_model, (height, width)
        )
        hidden = captured["hidden_states"]
        if hidden.ndim != 4 or hidden.shape[1] != n_prompts:
            raise RuntimeError(
                f"unexpected decoder hidden shape {tuple(hidden.shape)} "
                f"for {n_prompts} prompts"
            )
        hidden_selected = torch.stack(
            [hidden[:, i, selected[i], :] for i in range(n_prompts)], dim=1
        )
        pred_box_norm = torch.stack(
            [captured["pred_boxes_2d"][i, selected[i]] for i in range(n_prompts)]
        )
        depth_latents = captured["depth_latents"]
        ray_embeddings = captured["ray_embeddings"]
        scores, scores_2d, scores_3d = _combined_scores(self.wd, captured)

        final_box2d = torch.stack([boxes[0] for boxes in result[0]])
        final_box3d = torch.stack([boxes[0] for boxes in result[1]])
        final_score = torch.stack([values[0] for values in result[2]])
        final_score_2d = torch.stack([values[0] for values in result[3]])
        final_score_3d = torch.stack([values[0] for values in result[4]])
        routed_score = torch.stack(
            [scores[i, selected[i]] for i in range(n_prompts)]
        )
        routed_score_2d = torch.stack(
            [scores_2d[i, selected[i]] for i in range(n_prompts)]
        )
        routed_score_3d = torch.stack(
            [scores_3d[i, selected[i]] for i in range(n_prompts)]
        )
        if not torch.allclose(routed_score, final_score, atol=2e-3, rtol=2e-3):
            raise RuntimeError(
                "raw-query routing disagrees with predictor score: "
                f"raw={routed_score.tolist()} final={final_score.tolist()}"
            )

        return {
            "hidden_states": hidden_selected.detach().to(torch.bfloat16).cpu(),
            "pred_box_2d": pred_box_norm.detach().float().cpu(),
            "sel_idx": selected.detach().cpu(),
            "sel_n_positive_inside": n_inside,
            "depth_latents": depth_latents[0].detach().to(torch.bfloat16).cpu(),
            "ray_embeddings": ray_embeddings[0].detach().to(torch.bfloat16).cpu(),
            "intrinsics": data["intrinsics"].detach().float().cpu(),
            "input_hw": (height, width),
            "final_box_2d_xyxy": final_box2d.detach().float().cpu(),
            "final_box_3d": final_box3d.detach().float().cpu(),
            "final_score": final_score.detach().float().cpu(),
            "final_score_2d": final_score_2d.detach().float().cpu(),
            "final_score_3d": final_score_3d.detach().float().cpu(),
            "routed_score": routed_score.detach().float().cpu(),
            "routed_score_2d": routed_score_2d.detach().float().cpu(),
            "routed_score_3d": routed_score_3d.detach().float().cpu(),
        }
