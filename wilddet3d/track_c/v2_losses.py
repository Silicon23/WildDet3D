"""Physical-box losses for v2 Track C's arbitrary GT local-axis convention.

CA-1M and ADT do not promise that GT local axis 0/1/2 has WildDet3D's semantic
L/H/W meaning.  A physical cuboid has 24 equivalent proper axis relabelings:
``R' = R S`` for any signed permutation rotation ``S``, with dimensions
permuted to follow the columns.  Supervision must choose dims and rotation
*jointly* from one such representative; independent slotwise dims plus a D2
rotation minimum can fight an otherwise correct physical box.
"""

from __future__ import annotations

import itertools
import math

import torch
from torch import Tensor
from vis4d.op.geometry.rotation import quaternion_to_matrix

from wilddet3d.ops.rotation import rotation_6d_to_matrix


def proper_axis_relabelings(device, dtype) -> tuple[Tensor, Tensor]:
    """Return the 24 proper signed permutations and their dimension maps.

    Returns:
        rotations: [24,3,3], det +1.
        dim_permutations: [24,3], where candidate_dims = dims[:, permutation].
    """
    matrices = []
    permutations = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = torch.zeros(3, 3, dtype=torch.float64)
            # New column j is sign[j] * old column permutation[j].
            for new_axis, old_axis in enumerate(permutation):
                matrix[old_axis, new_axis] = signs[new_axis]
            if torch.linalg.det(matrix) > 0.5:
                matrices.append(matrix)
                permutations.append(permutation)
    rotations = torch.stack(matrices).to(device=device, dtype=dtype)
    dim_permutations = torch.tensor(
        permutations, device=device, dtype=torch.long
    )
    if rotations.shape != (24, 3, 3):
        raise AssertionError(f"expected 24 axis relabelings, got {rotations.shape}")
    return rotations, dim_permutations


def equivalent_gt_boxes(gt_dims: Tensor, gt_R: Tensor) -> tuple[Tensor, Tensor]:
    """Return all equivalent GT representations.

    Args:
        gt_dims: [N,3] full local-axis extents.
        gt_R: [N,3,3] local-to-camera rotations.

    Returns:
        candidate_dims: [N,24,3]
        candidate_R: [N,24,3,3]
    """
    relabelings, permutations = proper_axis_relabelings(
        gt_R.device, gt_R.dtype
    )
    candidate_R = torch.einsum("nij,sjk->nsik", gt_R, relabelings)
    candidate_dims = gt_dims[:, permutations]
    return candidate_dims, candidate_R


def _geodesic_candidates(pred_R: Tensor, candidate_R: Tensor) -> Tensor:
    relative = torch.matmul(
        pred_R.unsqueeze(1).transpose(-1, -2), candidate_R
    )
    trace = (
        relative[..., 0, 0] + relative[..., 1, 1] + relative[..., 2, 2]
    )
    cosine = ((trace - 1.0) * 0.5).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.acos(cosine)


def v2_track_c_loss(
    pred: Tensor,
    target: Tensor,
    weights: Tensor,
    gt_dims: Tensor,
    gt_quat: Tensor,
    valid: Tensor,
    dim_scale: float,
    w_center: float = 1.0,
    w_depth: float = 1.0,
    w_dims: float = 1.0,
    w_rot: float = 1.0,
) -> dict[str, Tensor]:
    """Deep-supervised coder loss with joint 24-way dims/rotation matching.

    ``pred`` is [L,N,12].  Only target center/depth slots are consumed; GT
    dims/rotation are expanded into physically equivalent joint candidates.
    """
    layers = pred.shape[0]
    valid_float = valid.float()
    n_valid = valid_float.sum().clamp(min=1.0)
    target_expanded = target.unsqueeze(0).expand(layers, -1, -1)
    weight_expanded = (weights * valid_float.unsqueeze(-1)).unsqueeze(0).expand(
        layers, -1, -1
    )

    def l1_slots(start: int, end: int) -> Tensor:
        diff = (pred[..., start:end] - target_expanded[..., start:end]).abs()
        selected_weights = weight_expanded[..., start:end]
        return (diff * selected_weights).sum() / selected_weights.sum().clamp(
            min=1.0
        )

    loss_center = l1_slots(0, 2)
    loss_depth = l1_slots(2, 3)

    gt_R = quaternion_to_matrix(gt_quat)
    candidate_dims, candidate_R = equivalent_gt_boxes(gt_dims, gt_R)
    candidate_log_dims = (
        torch.log(candidate_dims.clamp(min=1e-6)) * float(dim_scale)
    )
    joint_losses = []
    dims_logs = []
    rotation_logs = []
    for layer in range(layers):
        pred_dims = pred[layer, :, 3:6]
        dims_l1 = (
            pred_dims.unsqueeze(1) - candidate_log_dims
        ).abs().mean(dim=-1)
        pred_R = rotation_6d_to_matrix(pred[layer, :, 6:12])
        rotation_angle = _geodesic_candidates(pred_R, candidate_R)
        joint = w_dims * dims_l1 + w_rot * rotation_angle
        joint_min, representative = joint.min(dim=1)
        chosen_dims = dims_l1.gather(1, representative[:, None]).squeeze(1)
        chosen_rotation = rotation_angle.gather(
            1, representative[:, None]
        ).squeeze(1)
        joint_losses.append((joint_min * valid_float).sum() / n_valid)
        dims_logs.append((chosen_dims * valid_float).sum() / n_valid)
        rotation_logs.append((chosen_rotation * valid_float).sum() / n_valid)

    loss_joint = torch.stack(joint_losses).mean()
    loss_dims = torch.stack(dims_logs).mean()
    loss_rotation = torch.stack(rotation_logs).mean()
    total = w_center * loss_center + w_depth * loss_depth + loss_joint
    return {
        "loss": total,
        "loss_center": loss_center.detach(),
        "loss_depth": loss_depth.detach(),
        "loss_dims_encoded": loss_dims.detach(),
        "loss_rot_deg": loss_rotation.detach() * (180.0 / math.pi),
    }


def physical_dims_rotation_errors(
    pred_dims: Tensor,
    pred_R: Tensor,
    gt_dims: Tensor,
    gt_R: Tensor,
) -> tuple[Tensor, Tensor]:
    """Per-frame dims-MAE and angle after one joint physical relabeling."""
    candidate_dims, candidate_R = equivalent_gt_boxes(gt_dims, gt_R)
    # Log-ratio makes the representative choice scale-neutral across tiny and
    # large objects.  Reported dims error remains metric MAE after selection.
    dims_choice_cost = (
        torch.log(pred_dims.clamp(min=1e-6)).unsqueeze(1)
        - torch.log(candidate_dims.clamp(min=1e-6))
    ).abs().mean(dim=-1)
    rotation_angle = _geodesic_candidates(pred_R, candidate_R)
    representative = (dims_choice_cost + rotation_angle).argmin(dim=1)
    chosen_dims = candidate_dims.gather(
        1, representative[:, None, None].expand(-1, 1, 3)
    ).squeeze(1)
    chosen_rotation = rotation_angle.gather(
        1, representative[:, None]
    ).squeeze(1)
    return (pred_dims - chosen_dims).abs().mean(dim=-1), chosen_rotation
