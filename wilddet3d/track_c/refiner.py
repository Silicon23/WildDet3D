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
        use_temporal_modules: bool = True,
        use_layer_bias: bool = False,
        use_temporal_kv_norm: bool = False,
        temporal_multi_token: bool = False,
    ) -> None:
        super().__init__()
        self.coder = box_coder or Det3DCoder()
        self.reg_residual_from_prior = reg_residual_from_prior
        self.use_temporal_modules = use_temporal_modules
        self.use_layer_bias = use_layer_bias
        self.temporal_multi_token = use_temporal_modules and temporal_multi_token
        # num_pred_layer = num_decoder_layer + 1 (as_two_stage); we feed the L
        # decoder layers we actually have (6) -> head uses pred layers 0..L-1.
        self.head = Det3DHead(
            num_decoder_layer=num_decoder_layer,
            box_coder=self.coder,
            depth_latent_dim=depth_latent_dim,
            use_camera_prompt=True,
            use_depth_prompt=True,
            use_temporal_prompt=use_temporal_modules,
            traj_token_dim=traj_token_dim,
            use_layer_bias=use_layer_bias,
            use_temporal_kv_norm=use_temporal_kv_norm,
            temporal_multi_token=temporal_multi_token,
        )
        if use_temporal_modules:
            self.traj_encoder = TrajectoryEncoder(
                embed_dims=traj_token_dim, num_layers=traj_layers
            )
        else:
            self.traj_encoder = None

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

    def warm_start_temporal_from_depth(self) -> None:
        """Warm-start the temporal prompt branch from the pretrained depth
        branch (scale-fix init). Call AFTER load_pretrained_head."""
        if self.use_temporal_modules:
            self.head.warm_start_temporal_from_depth()

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

        if intrinsics.dim() == 2:
            proj = project_points(center, intrinsics)          # [B,2] (shared K)
        else:
            proj = project_points(center.unsqueeze(1), intrinsics).squeeze(1)  # per-frame K
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
        temporal_pe = None
        temporal_attn_mask = None  # single trajectory -> all frames attend each other
        if self.use_temporal_modules:
            tok, box_out_b = self.traj_encoder(
                box_repr.unsqueeze(0), timestamps.unsqueeze(0), measured_mask.unsqueeze(0)
            )                                        # [1,T,256], [1,T,12]
            box_out = box_out_b[0]                   # [T,12]
            base_box = box_out
            if self.temporal_multi_token:
                temporal_tokens = tok                # [1,T,256] (frames as sequence)
                temporal_pe = self.traj_encoder.time_pe(timestamps.unsqueeze(0))  # [1,T,256]
            else:
                temporal_tokens = tok[0].unsqueeze(1)  # [T,1,256]
        else:
            box_out = None
            temporal_tokens = None
            # without the encoder, the residual base is just the (interpolated)
            # input prior at each frame
            base_box = box_repr

        with torch.autocast(device_type="cuda", enabled=False):
            temporal_box_12d = self.encode_temporal_box_12d(
                base_box.float(), pred_box_2d.float(), intrinsics.float(), input_hw
            )

        temporal_box_in = (temporal_box_12d.unsqueeze(1)
                           if self.reg_residual_from_prior else None)
        stacked_reg, stacked_conf = self.head(
            hidden_states=hidden_states,
            ray_embeddings=ray_embeddings,
            depth_latents=depth_latents,
            temporal_tokens=temporal_tokens,
            temporal_box_12d=temporal_box_in,
            temporal_pe=temporal_pe,
            temporal_attn_mask=temporal_attn_mask,
        )                                            # [L,T,1,12], [L,T,1,1]
        return {
            "reg": stacked_reg,                      # [L,T,1,12] coder-encoded
            "conf": stacked_conf,                    # [L,T,1,1]
            "temporal_box_12d": temporal_box_12d,    # [T,12]
            "traj_box_out": box_out,                 # [T,12] abs cam prior (refined)
        }

    def forward_batch(self, packs: list) -> dict:
        """Batched forward over many trajectories for training throughput.

        The cheap per-trajectory parts (trajectory encoder + temporal-box
        geometry, each needing that trajectory's single intrinsics) run in a
        loop; the expensive 3D head runs ONCE over all trajectories' frames
        concatenated along the query/batch dim. Returns concatenated reg plus
        per-trajectory sizes so the loss/decode can split frames back.
        """
        toks, tb12s, hiddens, rays, depths, box_outs, sizes = [], [], [], [], [], [], []
        for p in packs:
            tokens, box_out = self.traj_encoder(
                p["box_repr"].unsqueeze(0), p["ts"].unsqueeze(0),
                p["measured"].unsqueeze(0))
            tokens = tokens[0]
            box_out = box_out[0]
            with torch.autocast(device_type="cuda", enabled=False):
                tb = self.encode_temporal_box_12d(
                    box_out.float(), p["box2d"].float(), p["K"].float(), p["input_hw"])
            toks.append(tokens)
            tb12s.append(tb)
            box_outs.append(box_out)
            hiddens.append(p["hidden"])
            rays.append(p["ray"])
            depths.append(p["depth"])
            sizes.append(tokens.shape[0])
        hidden = torch.cat(hiddens, dim=1)                 # [L, sum_T, 1, 256]
        ray = torch.cat(rays, dim=0)                       # [sum_T, ntok, 81]
        depth = torch.cat(depths, dim=0)                   # [sum_T, ntok, 256]
        temporal_tokens = torch.cat(toks, 0).unsqueeze(1)  # [sum_T, 1, 256]
        temporal_box_in = torch.cat(tb12s, 0).unsqueeze(1)  # [sum_T, 1, 12]
        stacked_reg, stacked_conf = self.head(
            hidden_states=hidden, ray_embeddings=ray, depth_latents=depth,
            temporal_tokens=temporal_tokens, temporal_box_12d=temporal_box_in,
        )                                                  # [L, sum_T, 1, 12]
        return {"reg": stacked_reg, "conf": stacked_conf,
                "sizes": sizes, "box_out": box_outs}

    def forward_vectorized(self, batch: dict) -> dict:
        """Fully-vectorized batched forward (no per-trajectory Python loop).

        ``batch`` (from ``collate_trajs``):
          box_repr/ts/measured/pad_mask  [K, Tmax, *]  (padded, for the encoder)
          hidden [L, sum_T, 1, 256], ray [sum_T, ntok, 81], depth [sum_T, ntok, 256],
          box2d [sum_T, 4], K [sum_T, 3, 3] (per-frame), input_hw
        Valid (non-pad) frames in row-major (k, t) order match the concatenated
        head inputs, so ``tokens[~pad_mask]`` aligns with them.
        """
        valid = ~batch["pad_mask"]                       # [K,Tmax]
        temporal_pe = None
        temporal_attn_mask = None
        if self.use_temporal_modules:
            tokens, box_out = self.traj_encoder(
                batch["box_repr"], batch["ts"], batch["measured"],
                key_padding_mask=batch["pad_mask"])      # [K,Tmax,256], [K,Tmax,12]
            tokens_v = tokens[valid]                     # [sum_T, 256]
            box_out_v = box_out[valid]                   # [sum_T, 12]
            with torch.autocast(device_type="cuda", enabled=False):
                temporal_box_12d = self.encode_temporal_box_12d(
                    box_out_v.float(), batch["box2d"].float(), batch["K"].float(),
                    batch["input_hw"])                   # [sum_T, 12] per-frame K
            temporal_box_in = (temporal_box_12d.unsqueeze(1)
                               if self.reg_residual_from_prior else None)
            if self.temporal_multi_token:
                # Flavor 2: per-frame query attends the whole (within-object)
                # timeline. tokens [1, sum_T, 256]; timestamp PE per frame
                # (computed per-trajectory so each has its own time origin);
                # block-diagonal attn_mask so frames only see their own object.
                temporal_tokens_in = tokens_v.unsqueeze(0)          # [1, sum_T, 256]
                pe_pad = self.traj_encoder.time_pe(batch["ts"])     # [K, Tmax, 256]
                temporal_pe = pe_pad[valid].unsqueeze(0)            # [1, sum_T, 256]
                sizes = valid.sum(dim=1)                            # [K]
                traj_id = torch.repeat_interleave(
                    torch.arange(valid.shape[0], device=valid.device), sizes
                )                                                  # [sum_T]
                temporal_attn_mask = traj_id[:, None] != traj_id[None, :]  # [sum_T,sum_T] True=block
            else:
                temporal_tokens_in = tokens_v.unsqueeze(1)         # [sum_T, 1, 256]
        else:
            # No trajectory encoder. If residual-from-prior is on, anchor on the
            # raw (interpolated) box_repr at each frame — no temporal smoothing,
            # just the same Step-4 prior. Otherwise pass nothing.
            tokens_v = None; box_out_v = None
            temporal_tokens_in = None; temporal_box_12d = None; temporal_box_in = None
            if self.reg_residual_from_prior:
                br_v = batch["box_repr"][valid]          # [sum_T, 12] abs cam
                with torch.autocast(device_type="cuda", enabled=False):
                    temporal_box_12d = self.encode_temporal_box_12d(
                        br_v.float(), batch["box2d"].float(),
                        batch["K"].float(), batch["input_hw"])
                temporal_box_in = temporal_box_12d.unsqueeze(1)

        stacked_reg, stacked_conf = self.head(
            hidden_states=batch["hidden"], ray_embeddings=batch["ray"],
            depth_latents=batch["depth"],
            temporal_tokens=temporal_tokens_in,
            temporal_box_12d=temporal_box_in,
            temporal_pe=temporal_pe,
            temporal_attn_mask=temporal_attn_mask,
        )                                                # [L, sum_T, 1, 12]
        return {"reg": stacked_reg, "conf": stacked_conf,
                "box_out": box_out_v, "temporal_box_12d": temporal_box_12d}

    def decode_layer(self, reg_layer: Tensor, pred_box_2d: Tensor,
                     intrinsics: Tensor, input_hw) -> Tensor:
        """reg_layer [T,12] -> decoded 3D box [T,10]=[center(3),dims(3),quat(4)]
        in camera frame, using each frame's predicted 2D box (pixels) + K."""
        H, W = input_hw
        boxes_px = pred_box_2d.clone()
        boxes_px[:, 0::2] *= W
        boxes_px[:, 1::2] *= H
        return self.coder.decode(boxes_px, reg_layer, intrinsics)
