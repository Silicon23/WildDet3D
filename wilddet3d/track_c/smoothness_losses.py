"""Pattern-resolving trajectory smoothness losses (trajectory_smoothness_losses.md).

Implements the portfolio's first wave on top of the existing base losses
(per-frame L1 + GT-derivative matching = L_pos + L_vel):

  R2   rotvel matching on UNFOLDED world-frame relative rotations
       (chordal by default; atan2-geodesic variant for the pi dead spot).
       Replaces the magnitude-only, symmetry-folded rotation term in
       ``derivative_matching_loss`` (which is blind to axis wobble and to
       180-degree flips between consecutive frames).
  L3h  acceleration hinge: ReLU(|acc_pred|_c - |acc_gt|_c - margin),
       Charbonnier-smoothed norms -> penalty LINEAR in jitter amplitude
       (the squared losses fade quadratically and go silent at small eps).
  L5/6 lag-k flip loss on the velocity deviation d_t = v_pred - v_gt:
       ReLU(-cos(d_t, d_{t+k}) - margin), delta-gated. Scale-invariant:
       O(1) per flipping frame at ANY amplitude. lags=(1,) -> L5; octave
       ladder (1,2,4,8) -> L6.
  Rotation pattern terms: same L5/L6 machinery on so(3) log-increment
       deviations, gated at ~150 deg (the log's conditioning boundary).

All terms are computed per trajectory on (concat, sizes)-packed sequences,
matching ``derivative_matching_loss``'s interface. Every term returns
monitors (detached) per the doc's monitoring sections.

delta units: velocity^2 (position, m^2/frame^2) / rad^2/frame^2 (rotation).
Anchor: delta ~ (noise-floor velocity deviation)^2 — see smoothness_eval's
dmag percentiles + gt_sigma.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
from torch import Tensor


# --------------------------------------------------------------------------
# so(3) log, differentiable, atan2 form, safe branches (no NaN through where)
# --------------------------------------------------------------------------

def so3_log_torch(R: Tensor, eps: float = 1e-7) -> tuple[Tensor, Tensor]:
    """R [...,3,3] -> (omega [...,3], theta [...]). atan2(|skew|, (tr-1)/2):
    stable through pi for the angle; direction Taylor-guarded near 0.
    Both where-branches are computed finite so gradients can't go NaN."""
    s = 0.5 * torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1]], dim=-1)                    # sin(th)*axis
    sn = s.norm(dim=-1)                                          # |sin th| >= 0
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cs = 0.5 * (tr - 1.0)
    th = torch.atan2(sn, cs)                                     # [0, pi]
    big = sn > eps
    scale_big = th / sn.clamp(min=eps)
    scale_small = 1.0 + th * th / 6.0                            # th/sin(th) Taylor
    scale = torch.where(big, scale_big, scale_small)
    return s * scale.unsqueeze(-1), th


def charbonnier(x: Tensor, delta: float) -> Tensor:
    """sqrt(|x|^2 + delta^2) over the last dim — smooth |.| (doc §0.5)."""
    return torch.sqrt((x * x).sum(-1) + delta * delta)


def _iter_trajs(sizes: Sequence[int]) -> Iterable[tuple[int, int]]:
    off = 0
    for n in sizes:
        yield off, off + int(n)
        off += int(n)


# --------------------------------------------------------------------------
# R2 — unfolded relative-rotation matching (anti-flip / anti-wobble workhorse)
# --------------------------------------------------------------------------

def rotvel_matching_loss(
    pred_R: Tensor,          # [N,3,3] concat over trajectories
    gt_R: Tensor,            # [N,3,3]
    sizes: Sequence[int],
    valid: Tensor,           # [N] bool
    form: str = "chordal",   # "chordal" | "geodesic" (atan2; unit grad through pi)
) -> dict:
    """R2: || (R^_{t+1} R^_t^T) - (R_{t+1} R_t^T) ||_F^2  (or geodesic angle of
    the relative-relative rotation). UNFOLDED: no symmetry min on increments —
    a 180-degree prediction flip against smooth GT costs full price. Increments
    are invariant to any constant per-track relabeling, so GT canonicalization
    only needs track-consistency (doc §8.2)."""
    terms, flips = [], []
    for a, b in _iter_trajs(sizes):
        if b - a < 2 or int(valid[a:b].sum()) < 2:
            continue
        m = valid[a:b]
        pair = m[:-1] & m[1:]
        if int(pair.sum()) < 1:
            continue
        relP = pred_R[a + 1:b] @ pred_R[a:b - 1].transpose(-1, -2)   # [T-1,3,3]
        relG = gt_R[a + 1:b] @ gt_R[a:b - 1].transpose(-1, -2)
        if form == "chordal":
            diff = (relP - relG)[pair]
            terms.append((diff * diff).sum((-1, -2)).mean())
        else:  # geodesic angle of relP relG^T via atan2 — unit grad through pi
            _, th = so3_log_torch(relP @ relG.transpose(-1, -2))
            terms.append(th[pair].mean())
        with torch.no_grad():
            _, thp = so3_log_torch(relP)
            flips.append((thp[pair] > math.radians(120.0)).float().mean())
    z = pred_R.new_zeros(())
    loss = torch.stack(terms).mean() if terms else z
    return {
        "loss": loss,
        "mon_flip_rate": (torch.stack(flips).mean().detach() if flips else z),
    }


# --------------------------------------------------------------------------
# L3 hinge — "no jerkier than reality", linear in jitter amplitude
# --------------------------------------------------------------------------

def accel_hinge_loss(
    pred_c: Tensor,          # [N,3]
    gt_c: Tensor,            # [N,3]
    sizes: Sequence[int],
    valid: Tensor,           # [N] bool
    margin: float = 0.0,     # metres/frame^2 of free slack above GT's envelope
    delta: float = 1e-3,     # Charbonnier smoothing (m)
) -> dict:
    terms, active = [], []
    for a, b in _iter_trajs(sizes):
        if b - a < 3 or int(valid[a:b].sum()) < 3:
            continue
        m = valid[a:b]
        trip = m[:-2] & m[1:-1] & m[2:]
        if int(trip.sum()) < 1:
            continue
        accP = pred_c[a + 2:b] - 2 * pred_c[a + 1:b - 1] + pred_c[a:b - 2]
        accG = gt_c[a + 2:b] - 2 * gt_c[a + 1:b - 1] + gt_c[a:b - 2]
        h = torch.relu(charbonnier(accP, delta) - charbonnier(accG, delta) - margin)
        terms.append(h[trip].mean())
        active.append((h[trip] > 0).float().mean().detach())
    z = pred_c.new_zeros(())
    return {
        "loss": torch.stack(terms).mean() if terms else z,
        "mon_hinge_active": torch.stack(active).mean() if active else z,
    }


# --------------------------------------------------------------------------
# L5 / L6 — lag-k flip loss on deviation sequences (scale-invariant)
# --------------------------------------------------------------------------

def lag_flip_loss_on_dev(
    d: Tensor,               # [M,3] deviation sequence of ONE trajectory
    lags: Sequence[int] = (1,),
    delta: float = 25e-6,    # velocity^2; sqrt(delta)=5mm/frame default anchor
    margin: float = 0.2,
    weights: Sequence[float] | None = None,
) -> tuple[Tensor, dict]:
    """Core L5/L6 on a single deviation sequence. Returns (loss, monitors)."""
    z = d.new_zeros(())
    if len(d) < 2:
        return z, {}
    w = list(weights) if weights is not None else [1.0] * len(lags)
    total, mons = z, {}
    for k, wk in zip(lags, w):
        if len(d) <= k:
            continue
        a, b = d[:-k], d[k:]
        dot = (a * b).sum(-1)
        denom = a.norm(dim=-1) * b.norm(dim=-1) + delta
        cos = dot / denom
        hinge = torch.relu(-cos - margin)
        total = total + wk * hinge.mean()
        mons[f"mon_lag{k}_cos"] = cos.mean().detach()
        mons[f"mon_lag{k}_active"] = (hinge > 0).float().mean().detach()
    return total, mons


def position_flip_loss(
    pred_c: Tensor, gt_c: Tensor, sizes: Sequence[int], valid: Tensor,
    lags: Sequence[int] = (1,), delta: float = 25e-6, margin: float = 0.2,
) -> dict:
    """L5/L6 on position velocity deviation d_t = (v^_t - v_t)."""
    terms, mon_acc = [], {}
    for a, b in _iter_trajs(sizes):
        if b - a < 3 or int(valid[a:b].sum()) < 3:
            continue
        m = valid[a:b]
        pair = m[:-1] & m[1:]
        d = ((pred_c[a + 1:b] - pred_c[a:b - 1])
             - (gt_c[a + 1:b] - gt_c[a:b - 1]))
        d = d[pair]
        loss_k, mons = lag_flip_loss_on_dev(d, lags, delta, margin)
        terms.append(loss_k)
        for k, v in mons.items():
            mon_acc.setdefault(k, []).append(v)
    z = pred_c.new_zeros(())
    out = {"loss": torch.stack(terms).mean() if terms else z}
    for k, vs in mon_acc.items():
        out[k] = torch.stack(vs).mean()
    return out


def rotation_flip_loss(
    pred_R: Tensor, gt_R: Tensor, sizes: Sequence[int], valid: Tensor,
    lags: Sequence[int] = (1,), delta: float = 7.6e-7,   # (0.05 deg/frame)^2 rad^2
    margin: float = 0.2, gate_deg: float = 150.0,
) -> dict:
    """L5/L6 on rotational increment deviation d_t = omega^_t - omega_t (world-
    frame logs, gated at the log's conditioning boundary). Frames beyond the
    gate are near-flips already hammered by R2 (doc §8.4 R4)."""
    gate = math.radians(gate_deg)
    terms, mon_acc, pass_rates = [], {}, []
    for a, b in _iter_trajs(sizes):
        if b - a < 3 or int(valid[a:b].sum()) < 3:
            continue
        m = valid[a:b]
        pair = m[:-1] & m[1:]
        wp, thp = so3_log_torch(pred_R[a + 1:b] @ pred_R[a:b - 1].transpose(-1, -2))
        wg, thg = so3_log_torch(gt_R[a + 1:b] @ gt_R[a:b - 1].transpose(-1, -2))
        g = pair & (thp < gate) & (thg < gate)
        pass_rates.append(g.float().mean().detach())
        if int(g.sum()) < 2:
            continue
        d = (wp - wg)[g]
        loss_k, mons = lag_flip_loss_on_dev(d, lags, delta, margin)
        terms.append(loss_k)
        for k, v in mons.items():
            mon_acc.setdefault(k, []).append(v)
    z = pred_R.new_zeros(())
    out = {"loss": torch.stack(terms).mean() if terms else z,
           "mon_gate_pass": torch.stack(pass_rates).mean() if pass_rates else z}
    for k, vs in mon_acc.items():
        out[k] = torch.stack(vs).mean()
    return out


# --------------------------------------------------------------------------
# Bundle — one call from the trainer
# --------------------------------------------------------------------------

def pattern_smoothness_loss(
    pred_center: Tensor, pred_R: Tensor,
    gt_center: Tensor, gt_R: Tensor,
    sizes: Sequence[int], valid: Tensor,
    w_rotvel2: float = 0.0,      # R2 (chordal). Replaces old magnitude rotvel.
    rotvel2_form: str = "chordal",
    w_acc_hinge: float = 0.0,    # L3h
    acc_hinge_margin: float = 0.0,
    w_pos_flip: float = 0.0,     # L5/L6 position
    w_rot_flip: float = 0.0,     # L5/L6 rotation (gated)
    flip_lags: Sequence[int] = (1,),
    pos_delta: float = 25e-6,
    rot_delta: float = 7.6e-7,
    flip_margin: float = 0.2,
) -> dict:
    """Compose the first-wave pattern terms. Returns total + per-term values +
    monitors (all monitors prefixed mon_)."""
    out = {}
    total = pred_center.new_zeros(())
    if w_rotvel2 > 0:
        r2 = rotvel_matching_loss(pred_R, gt_R, sizes, valid, rotvel2_form)
        total = total + w_rotvel2 * r2["loss"]
        out["loss_rotvel2"] = r2["loss"].detach()
        out["mon_flip_rate"] = r2["mon_flip_rate"]
    if w_acc_hinge > 0:
        l3 = accel_hinge_loss(pred_center, gt_center, sizes, valid,
                              margin=acc_hinge_margin)
        total = total + w_acc_hinge * l3["loss"]
        out["loss_acc_hinge"] = l3["loss"].detach()
        out["mon_acc_hinge_active"] = l3["mon_hinge_active"]
    if w_pos_flip > 0:
        l5 = position_flip_loss(pred_center, gt_center, sizes, valid,
                                flip_lags, pos_delta, flip_margin)
        total = total + w_pos_flip * l5["loss"]
        out["loss_pos_flip"] = l5["loss"].detach()
        out.update({f"pos_{k}": v for k, v in l5.items() if k.startswith("mon_")})
    if w_rot_flip > 0:
        r5 = rotation_flip_loss(pred_R, gt_R, sizes, valid,
                                flip_lags, rot_delta, flip_margin)
        total = total + w_rot_flip * r5["loss"]
        out["loss_rot_flip"] = r5["loss"].detach()
        out.update({f"rot_{k}": v for k, v in r5.items() if k.startswith("mon_")})
    out["loss"] = total
    return out
