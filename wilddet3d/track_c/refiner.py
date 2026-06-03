"""TrackCRefiner: trajectory encoder + temporal 3D head, trained on cached/live
frozen features.

This is the *trainable* part of Track C. It owns:

- ``traj_encoder`` (TrajectoryEncoder): consumes the prior camera-frame box
  trajectory -> per-frame temporal token + temporal-aware camera-frame box,
- ``head`` (temporal Det3DHead): loaded from the pretrained WildDet3D 3D head,
  plus the new temporal prompt + output residual,
- ``coder`` (Det3DCoder): encodes/decodes the 12-d box <-> camera-frame box.

The frozen SAM3 / LingBot stack is NOT held here; its per-frame outputs (selected
query hidden states, depth latents, ray embeddings, predicted 2D box, intrinsics)
come from ``FrozenFeatureExtractor`` (live) or the feature cache. So training
backprops only through ``traj_encoder`` + ``head``.

Per-frame batching: each refine frame is one batch element with a single query
(``hidden_states [L, T, 1, 256]``); the trajectory encoder ties them together
temporally. The temporal box is re-encoded to the 12-d coder space against each
frame's predicted 2D box (delta_center is 2D-box-relative).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
from vis4d.op.geometry.projection import project_points

from wilddet3d.head import Det3DCoder, Det3DHead, TrajectoryEncoder


class TrackCRefiner(nn.Module):
    def __init__(
        self,
        num_decoder_layer: int = 6,
        traj_layers: int = 4,
        traj_token_dim: int = 256,
        reg_residual_from_prior: bool = True,
        box_coder: Optional[Det3DCoder] = None,
        depth_latent_dim: int = 256,
    ) -> None:
        super().__init__()
        self.coder = box_coder or Det3DCoder()
        self.reg_residual_from_prior = reg_residual_from_prior
        # num_pred_layer = num_decoder_layer + 1 (as_two_stage); we feed the L
        # decoder layers we actually have (6) -> head uses pred layers 0..L-1.
        self.head = Det3DHead(
            num_decoder_layer=num_decoder_layer,
            box_coder=self.coder,
            depth_latent_dim=depth_latent_dim,
            use_camera_prompt=True,
            use_depth_prompt=True,
            use_temporal_prompt=True,
            traj_token_dim=traj_token_dim,
        )
        self.traj_encoder = TrajectoryEncoder(
            embed_dims=traj_token_dim, num_layers=traj_layers
        )

    # ---- pretrained-head loading -------------------------------------------
    def load_pretrained_head(self, checkpoint_path: str, map_location="cpu") -> dict:
        """Load WildDet3D 3D-head weights (reg/conf/prompt_*/project_*) into
        ``self.head``. New temporal modules stay at their (identity) init.
        Call ``finalize_init`` afterwards.
        """
        ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        head_sd = {}
        for k, v in sd.items():
            kk = k[len("model."):] if k.startswith("model.") else k
            if kk.startswith("bbox3d_head."):
                head_sd[kk[len("bbox3d_head."):]] = v
        missing, unexpected = self.head.load_state_dict(head_sd, strict=False)
        # 'missing' should be exactly the new temporal params + temporal_gate.
        return {"loaded": len(head_sd), "missing": list(missing), "unexpected": list(unexpected)}

    def finalize_init(self) -> None:
        """Apply Track C init choices after loading pretrained head weights."""
        if self.reg_residual_from_prior:
            self.head.zero_init_reg_residual()

    # ---- temporal box (abs cam) -> 12-d coder encoding ---------------------
    def encode_temporal_box_12d(
        self, box_repr_abs: Tensor, pred_box_2d_norm: Tensor,
        intrinsics: Tensor, input_hw,
    ) -> Tensor:
        """box_repr_abs [B,12]=[center(3),log_dims(3),rot6d(6)] ->
        coder 12-d [delta_center/cs, log_depth*ds, log_dims*dimscale, rot6d].

        pred_box_2d_norm [B,4] normalized xyxy (model space); intrinsics [B,3,3]
        model space; input_hw=(H,W).
        """
        H, W = input_hw
        center = box_repr_abs[:, 0:3]
        log_dims = box_repr_abs[:, 3:6]
        rot6d = box_repr_abs[:, 6:12]

        proj = project_points(center, intrinsics)              # [B,2] model px
        ctr_x = (pred_box_2d_norm[:, 0] + pred_box_2d_norm[:, 2]) * 0.5 * W
        ctr_y = (pred_box_2d_norm[:, 1] + pred_box_2d_norm[:, 3]) * 0.5 * H
        center_2d = torch.stack([ctr_x, ctr_y], -1)
        delta_center = (proj - center_2d) / self.coder.center_scale

        depth = center[:, 2].clamp(min=1e-3)
        log_depth = (torch.log(depth) * self.coder.depth_scale).unsqueeze(-1)
        log_dims_enc = log_dims * self.coder.dim_scale
        return torch.cat([delta_center, log_depth, log_dims_enc, rot6d], dim=-1)

    # ---- forward ------------------------------------------------------------
    def forward(
        self,
        hidden_states: Tensor,    # [L, T, 1, 256] selected query per frame
        ray_embeddings: Tensor,   # [T, N_tok, 81]
        depth_latents: Tensor,    # [T, N_tok, 256]
        pred_box_2d: Tensor,      # [T, 4] normalized xyxy (model space)
        intrinsics: Tensor,       # [T, 3, 3] model space
        box_repr: Tensor,         # [T, 12] absolute cam-frame prior boxes
        timestamps: Tensor,       # [T] seconds
        measured_mask: Tensor,    # [T] bool
        input_hw,
    ) -> dict:
        T = box_repr.shape[0]
        tokens, box_out = self.traj_encoder(
            box_repr.unsqueeze(0), timestamps.unsqueeze(0), measured_mask.unsqueeze(0)
        )                                            # [1,T,256], [1,T,12]
        tokens = tokens[0]                           # [T,256]
        box_out = box_out[0]                         # [T,12] temporal-aware abs box

        temporal_box_12d = self.encode_temporal_box_12d(
            box_out, pred_box_2d, intrinsics, input_hw
        )                                            # [T,12] coder space

        # head batch = T frames, S=1 query each
        temporal_tokens = tokens.unsqueeze(1)        # [T,1,256]
        temporal_box_in = temporal_box_12d.unsqueeze(1)  # [T,1,12]
        stacked_reg, stacked_conf = self.head(
            hidden_states=hidden_states,
            ray_embeddings=ray_embeddings,
            depth_latents=depth_latents,
            temporal_tokens=temporal_tokens,
            temporal_box_12d=temporal_box_in,
        )                                            # [L,T,1,12], [L,T,1,1]
        return {
            "reg": stacked_reg,                      # [L,T,1,12] coder-encoded
            "conf": stacked_conf,                    # [L,T,1,1]
            "temporal_box_12d": temporal_box_12d,    # [T,12]
            "traj_box_out": box_out,                 # [T,12] abs cam prior (refined)
        }

    def decode_layer(self, reg_layer: Tensor, pred_box_2d: Tensor,
                     intrinsics: Tensor, input_hw) -> Tensor:
        """reg_layer [T,12] -> decoded 3D box [T,10]=[center(3),dims(3),quat(4)]
        in camera frame, using each frame's predicted 2D box (pixels) + K."""
        H, W = input_hw
        boxes_px = pred_box_2d.clone()
        boxes_px[:, 0::2] *= W
        boxes_px[:, 1::2] *= H
        return self.coder.decode(boxes_px, reg_layer, intrinsics)
