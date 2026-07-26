"""Variant-conditioned Track C refiner for v2 positive-point trajectories."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from vis4d.op.geometry.rotation import matrix_to_quaternion

from wilddet3d.ops.rotation import rotation_6d_to_matrix
from wilddet3d.track_c.refiner import TrackCRefiner


class V2TrackCRefiner(TrackCRefiner):
    """Track C with explicit point-prompt lineage conditioning.

    ``point_v3`` and ``point_vlm_v1`` share the image-grounded refiner but are
    never silently pooled: a learned variant embedding is added to every
    selected decoder-query hidden state.  The embedding is zero-initialized, so
    a v1 checkpoint remains a valid warm start while the lineage branch learns
    only from v2 data.
    """

    def __init__(self, num_prompt_variants: int = 2, **kwargs) -> None:
        super().__init__(**kwargs)
        if num_prompt_variants < 2:
            raise ValueError("v2 requires at least two explicit prompt variants")
        self.num_prompt_variants = num_prompt_variants
        self.prompt_variant_embed = nn.Embedding(num_prompt_variants, 256)
        nn.init.zeros_(self.prompt_variant_embed.weight)

    def forward_vectorized(
        self,
        batch: dict,
        frame_mask: Tensor | None = None,
        variant_intervention: str = "correct",
    ) -> dict:
        """Forward with optional lineage intervention for causal diagnostics.

        ``variant_intervention`` is one of:
        - ``correct``: use the recorded variant id;
        - ``swapped``: exchange ids 0 and 1;
        - ``zero``: remove the learned conditioning vector.
        """
        variant_ids = batch["prompt_variant_id_per_frame"].long()
        if variant_ids.numel() != batch["hidden"].shape[1]:
            raise ValueError("per-frame variant ids do not align with hidden states")
        if int(variant_ids.min()) < 0 or int(variant_ids.max()) >= self.num_prompt_variants:
            raise ValueError(f"variant ids out of range: {variant_ids.unique().tolist()}")
        if variant_intervention == "swapped":
            if self.num_prompt_variants != 2:
                raise ValueError("swapped diagnostic is defined for two variants")
            variant_ids = 1 - variant_ids
        elif variant_intervention not in ("correct", "zero"):
            raise ValueError(f"unknown variant intervention {variant_intervention!r}")

        hidden = batch["hidden"].clone()
        depth = batch["depth"]
        if frame_mask is not None and bool(frame_mask.any()):
            hidden[:, frame_mask] = self.mask_embed.to(hidden.dtype)
            depth = depth.clone()
            depth[frame_mask] = 0
        if variant_intervention != "zero":
            variant_tokens = self.prompt_variant_embed(variant_ids).to(hidden.dtype)
            hidden = hidden + variant_tokens.unsqueeze(0).unsqueeze(2)
        conditioned = {**batch, "hidden": hidden, "depth": depth}
        # Masking was applied above so the explicit variant token survives on
        # masked frames.  Do not ask the base class to replace hidden again.
        return super().forward_vectorized(conditioned, frame_mask=None)

    def decode_layer(
        self,
        reg_layer: Tensor,
        pred_box_2d: Tensor,
        intrinsics: Tensor,
        input_hw,
    ) -> Tensor:
        """Decode with one distinct intrinsics matrix per v2 frame.

        ``Det3DCoder.decode`` delegates to a Vis4D unprojection helper that
        accepts only one shared K. V2 batches trajectories, cameras, and
        datasets together, so silently taking K[0] corrupts every other frame.
        """
        if intrinsics.ndim == 2 or intrinsics.shape[0] == 1:
            return super().decode_layer(
                reg_layer, pred_box_2d, intrinsics, input_hw
            )
        if intrinsics.shape[0] != reg_layer.shape[0]:
            raise ValueError(
                f"per-frame K count {intrinsics.shape[0]} != "
                f"box count {reg_layer.shape[0]}"
            )
        if self.coder.orientation != "rotation_6d":
            raise ValueError("v2 Track C requires rotation_6d coder orientation")
        if self.coder.canonical_rotation or self.coder.ambiguous_rotation:
            raise ValueError(
                "v2 Track C physical-axis loss requires unnormalized rotations"
            )
        height, width = input_hw
        boxes_px = pred_box_2d.clone()
        boxes_px[:, 0::2] *= width
        boxes_px[:, 1::2] *= height
        delta_center = reg_layer[:, :2] * self.coder.center_scale
        center_2d = torch.stack(
            [
                (boxes_px[:, 0] + boxes_px[:, 2]) * 0.5,
                (boxes_px[:, 1] + boxes_px[:, 3]) * 0.5,
            ],
            dim=-1,
        )
        projected = center_2d + delta_center
        homogeneous = torch.cat(
            [projected, torch.ones_like(projected[:, :1])], dim=-1
        )
        rays = torch.linalg.solve(
            intrinsics.float(), homogeneous.float().unsqueeze(-1)
        ).squeeze(-1)
        depth = torch.exp(reg_layer[:, 2] / self.coder.depth_scale)
        center = rays.to(depth.dtype) * depth.unsqueeze(-1)
        dims = torch.exp(reg_layer[:, 3:6] / self.coder.dim_scale)
        rotation = rotation_6d_to_matrix(reg_layer[:, 6:12])
        quaternion = matrix_to_quaternion(rotation)
        return torch.cat([center, dims, quaternion], dim=-1)
