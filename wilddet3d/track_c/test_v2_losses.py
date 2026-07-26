"""CPU regressions for physical v2 Track C box supervision."""

from __future__ import annotations

import torch

from wilddet3d.track_c.v2_losses import (
    equivalent_gt_boxes,
    physical_dims_rotation_errors,
    proper_axis_relabelings,
)


def test_group_has_24_proper_rotations() -> None:
    group, permutations = proper_axis_relabelings("cpu", torch.float64)
    assert group.shape == (24, 3, 3)
    assert permutations.shape == (24, 3)
    identity = torch.eye(3, dtype=torch.float64)
    assert torch.allclose(group.transpose(-1, -2) @ group, identity.expand(24, -1, -1))
    assert torch.allclose(torch.linalg.det(group), torch.ones(24, dtype=torch.float64))
    assert torch.unique(group.reshape(24, -1), dim=0).shape[0] == 24


def test_axis_permutation_is_physically_zero_error() -> None:
    gt_dims = torch.tensor([[4.2, 1.5, 1.8]])
    gt_R = torch.eye(3).unsqueeze(0)
    candidates_dims, candidates_R = equivalent_gt_boxes(gt_dims, gt_R)
    # Pick a non-D2 candidate that permutes axes.
    candidate = next(
        i
        for i in range(24)
        if not torch.equal(candidates_dims[0, i], gt_dims[0])
    )
    dims_error, rotation_error = physical_dims_rotation_errors(
        candidates_dims[:, candidate],
        candidates_R[:, candidate],
        gt_dims,
        gt_R,
    )
    assert float(dims_error.max()) < 1e-7
    # acos is deliberately clamped for stable gradients, so identity is ~0.08°.
    assert float(rotation_error.max()) < 2e-3
