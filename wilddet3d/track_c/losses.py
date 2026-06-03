"""Track C losses.

Philosophy (design §6): a *denoising* objective — reconstruct the clean CA-1M GT
trajectory; smoothness lives in the target, not in a regularizer. Per-frame 3D
box regression, deep-supervised over all 3D-head prediction layers.

- ``delta_center``, ``log_depth``, ``log_dims``: L1 (matches WildDet3D), with the
  GT re-encoded against each frame's *predicted* 2D box (so ``delta_center``'s
  reference matches the prediction).
- rotation: **symmetry-aware geodesic** (angle between rotation matrices), min
  over a 180°-flip symmetry group so annotation flips on near-symmetric objects
  aren't learned as jitter.
- per-(object, frame) validity mask (the ``weights_3d=0`` idea, per-frame).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor
from vis4d.op.geometry.rotation import quaternion_to_matrix

from wilddet3d.head import Det3DCoder
from wilddet3d.ops.rotation import rotation_6d_to_matrix


def _flip_symmetry_group(device, dtype) -> Tensor:
    """4 box-preserving 180° rotations: I and 180° about x / y / z. [4,3,3]."""
    I = torch.eye(3, device=device, dtype=dtype)
    Rx = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=device, dtype=dtype))
    Ry = torch.diag(torch.tensor([-1.0, 1.0, -1.0], device=device, dtype=dtype))
    Rz = torch.diag(torch.tensor([-1.0, -1.0, 1.0], device=device, dtype=dtype))
    return torch.stack([I, Rx, Ry, Rz], 0)


def geodesic_rotation_loss(
    R_pred: Tensor, R_gt: Tensor, symmetry_aware: bool = True
) -> Tensor:
    """Per-sample geodesic angle (radians) between R_pred and R_gt. [N,3,3]->[N].

    Symmetry-aware: min over the 180°-flip group applied to R_gt.
    """
    if symmetry_aware:
        S = _flip_symmetry_group(R_pred.device, R_pred.dtype)        # [4,3,3]
        R_gt_cand = torch.einsum("nij,sjk->nsik", R_gt, S)           # [N,4,3,3]
        Rp = R_pred.unsqueeze(1)                                     # [N,1,3,3]
        rel = torch.matmul(Rp.transpose(-1, -2), R_gt_cand)         # [N,4,3,3]
        trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]     # [N,4]
        cos = ((trace - 1.0) * 0.5).clamp(-1 + 1e-6, 1 - 1e-6)
        ang = torch.acos(cos)                                        # [N,4]
        return ang.min(dim=1).values                                # [N]
    rel = torch.matmul(R_pred.transpose(-1, -2), R_gt)
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos = ((trace - 1.0) * 0.5).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.acos(cos)


def build_targets(
    coder: Det3DCoder,
    gt_center: Tensor,      # [T,3]
    gt_dims: Tensor,        # [T,3] full extents
    gt_quat: Tensor,        # [T,4] wxyz
    pred_box_2d: Tensor,    # [T,4] normalized xyxy
    intrinsics: Tensor,     # [T,3,3] model space
    input_hw,
) -> tuple[Tensor, Tensor]:
    """Encode GT 3D boxes to the coder's 12-d space vs each frame's predicted
    2D box. Returns (target_12d [T,12], weights_12d [T,12])."""
    H, W = input_hw
    boxes_px = pred_box_2d.clone()
    boxes_px[:, 0::2] *= W
    boxes_px[:, 1::2] *= H
    boxes3d = torch.cat([gt_center, gt_dims, gt_quat], dim=-1)       # [T,10]
    target, weights = coder.encode(boxes_px, boxes3d, intrinsics)
    return target, weights


def track_c_loss(
    pred_reg: Tensor,        # [L,T,1,12] coder-encoded predictions
    coder: Det3DCoder,
    gt_center: Tensor, gt_dims: Tensor, gt_quat: Tensor, gt_R: Tensor,
    pred_box_2d: Tensor,     # [T,4] normalized xyxy
    intrinsics: Tensor,      # [T,3,3]
    input_hw,
    valid: Optional[Tensor] = None,   # [T] bool, per-frame GT-quality mask
    w_center: float = 1.0,
    w_depth: float = 1.0,
    w_dims: float = 1.0,
    w_rot: float = 1.0,
    symmetry_aware: bool = True,
) -> dict:
    L, T = pred_reg.shape[0], pred_reg.shape[1]
    pred = pred_reg[:, :, 0, :]                                      # [L,T,12]
    target, weights = build_targets(
        coder, gt_center, gt_dims, gt_quat, pred_box_2d, intrinsics, input_hw
    )                                                               # [T,12]
    if valid is None:
        valid = torch.ones(T, dtype=torch.bool, device=pred.device)
    vf = valid.float()                                              # [T]
    # combine coder per-element weights (invalid depth/dims -> 0) with frame mask
    w = weights * vf.unsqueeze(-1)                                  # [T,12]
    n = vf.sum().clamp(min=1.0)

    tgt = target.unsqueeze(0).expand(L, -1, -1)                    # [L,T,12]
    wexp = w.unsqueeze(0).expand(L, -1, -1)

    def l1(idx0, idx1):
        diff = (pred[..., idx0:idx1] - tgt[..., idx0:idx1]).abs()
        ww = wexp[..., idx0:idx1]
        return (diff * ww).sum() / (ww.sum().clamp(min=1.0))

    loss_center = l1(0, 2)
    loss_depth = l1(2, 3)
    loss_dims = l1(3, 6)

    # rotation: geodesic per layer, averaged
    rot_losses = []
    for li in range(L):
        R_pred = rotation_6d_to_matrix(pred[li, :, 6:12])           # [T,3,3]
        ang = geodesic_rotation_loss(R_pred, gt_R, symmetry_aware)  # [T]
        rot_losses.append((ang * vf).sum() / n)
    loss_rot = torch.stack(rot_losses).mean()

    total = (w_center * loss_center + w_depth * loss_depth
             + w_dims * loss_dims + w_rot * loss_rot)
    return {
        "loss": total,
        "loss_center": loss_center.detach(),
        "loss_depth": loss_depth.detach(),
        "loss_dims": loss_dims.detach(),
        "loss_rot_deg": (loss_rot.detach() * 180.0 / math.pi),
    }
