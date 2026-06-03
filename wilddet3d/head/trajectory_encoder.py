"""Trajectory encoder for the Track C temporal 3D-box refiner.

A small *bidirectional* transformer over an object's prior per-frame
camera-frame 3D box trajectory (Steps 1-4, or Step-5 Kalman as a stronger
prior). It produces, per (object, output frame):

- a 256-d **temporal token** consumed by the 3D head's temporal prompt, and
- a **temporal-aware camera-frame 3D box** (the densified/denoised prior) used
  as the output-space residual base in the 3D head.

Design (see ``temporal_smoother_design.md`` sections 3, 4.3):

- **Camera frame only.** The box trajectory is in camera coordinates. We do NOT
  feed extrinsics: on CA-1M's static scenes that would let the model learn the
  ``world_box = inv(RT_t) @ box_t`` shortcut and re-introduce the world-frame
  degeneracy the camera-frame design exists to avoid.
- **Timestamp-based positional encoding.** Real frame time (seconds), not
  sequence index, so irregular subsampling and dropout gaps are handled.
- **Bidirectional.** Smoothing is offline; future keyframes are as informative
  as past ones. A causal mask can be supplied later for an online use-case.
- **Identity at init.** The decoded box is ``input_box + zero_init_residual``,
  so at initialization the encoder returns its (interpolated) input prior
  unchanged and learns refinements from there.

Box representation (``box_repr``, 12-d, *absolute* camera frame):
``[center_xyz (3, metres), log_dims (3), rot_6d (6)]``. Note this is the
absolute 3D box, NOT the 2D-box-relative 12-d ``Det3DCoder`` encoding (which has
``delta_center`` instead of ``center`` and needs a 2D box). Conversion to the
coder's encoding happens per frame in the model call site, against that frame's
predicted 2D box (``delta_center`` is 2D-box-relative).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


# Absolute camera-frame box representation dim: center(3) + log_dims(3) + rot_6d(6)
BOX_REPR_DIM = 12


class TimestampEncoding(nn.Module):
    """Continuous sinusoidal positional encoding over real timestamps.

    Maps a ``[B, T]`` tensor of timestamps (seconds, arbitrary origin) to
    ``[B, T, dim]``. Uses a geometric spread of periods so both fast jitter and
    slow drift are representable; works for irregular spacing because it is a
    function of the continuous time value, not the index.
    """

    def __init__(
        self,
        dim: int = 256,
        min_period: float = 0.05,
        max_period: float = 60.0,
    ) -> None:
        super().__init__()
        assert dim % 2 == 0, "TimestampEncoding dim must be even"
        self.dim = dim
        n = dim // 2
        # Geometric progression of angular frequencies between the two periods.
        periods = torch.exp(
            torch.linspace(math.log(min_period), math.log(max_period), n)
        )
        freqs = (2.0 * math.pi) / periods  # [n]
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, timestamps: Tensor) -> Tensor:
        """timestamps: [B, T] (seconds). Returns [B, T, dim]."""
        # Subtract per-sequence min so the encoding is shift-invariant and
        # numerically stable regardless of absolute timestamp magnitude.
        t = timestamps - timestamps.amin(dim=1, keepdim=True)
        ang = t.unsqueeze(-1) * self.freqs.to(t.dtype)  # [B, T, n]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # [B, T, dim]


class TrajectoryEncoder(nn.Module):
    """Bidirectional transformer over a per-object camera-frame box trajectory.

    Args:
        embed_dims: Token / model dimension (matches the 3D head, 256).
        num_layers: Number of transformer encoder layers.
        num_heads: Attention heads.
        ffn_dim: Feed-forward hidden dim.
        dropout: Dropout in the transformer layers.
        box_repr_dim: Input box representation dim (default 12, see module doc).
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.0,
        box_repr_dim: int = BOX_REPR_DIM,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.box_repr_dim = box_repr_dim

        # Box -> token embedding (small MLP).
        self.box_embed = nn.Sequential(
            nn.Linear(box_repr_dim, embed_dims),
            nn.GELU(),
            nn.Linear(embed_dims, embed_dims),
        )
        # Learned embedding for measured (1) vs interpolated/filled (0) frames.
        self.measured_embed = nn.Embedding(2, embed_dims)
        self.time_pe = TimestampEncoding(embed_dims)
        self.input_norm = nn.LayerNorm(embed_dims)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dims,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        # Residual head: decoded box = input box + zero-init residual(token).
        self.box_decode = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.GELU(),
            nn.Linear(embed_dims, box_repr_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        # Zero-init the final residual projection so the decoded box equals the
        # (interpolated) input prior at initialization -> identity start.
        nn.init.zeros_(self.box_decode[-1].weight)
        nn.init.zeros_(self.box_decode[-1].bias)
        # Small measured/unmeasured embedding so it doesn't dominate at init.
        nn.init.normal_(self.measured_embed.weight, std=0.02)

    def forward(
        self,
        box_repr: Tensor,
        timestamps: Tensor,
        measured_mask: Tensor,
        key_padding_mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Encode a batch of per-object box trajectories.

        Args:
            box_repr: [B, T, box_repr_dim] absolute camera-frame boxes,
                pre-filled (e.g. interpolated) at unmeasured frames.
            timestamps: [B, T] real frame time in seconds.
            measured_mask: [B, T] bool/int, True(1) where a real measurement
                exists, False(0) at interpolated/filled frames.
            key_padding_mask: [B, T] bool, True at padded positions to ignore
                in attention (variable-length trajectories). Optional.
            attn_mask: optional [T, T] additive/boolean attention mask (e.g. a
                causal mask for a future online mode). Default None
                (bidirectional).

        Returns:
            tokens: [B, T, embed_dims] temporal tokens (one per output frame).
            box_out: [B, T, box_repr_dim] temporal-aware camera-frame boxes
                (= box_repr + zero-init residual; equals box_repr at init).
        """
        measured_idx = measured_mask.long()
        x = (
            self.box_embed(box_repr)
            + self.measured_embed(measured_idx)
            + self.time_pe(timestamps)
        )
        x = self.input_norm(x)
        tokens = self.encoder(
            x, mask=attn_mask, src_key_padding_mask=key_padding_mask
        )
        box_out = box_repr + self.box_decode(tokens)
        return tokens, box_out
