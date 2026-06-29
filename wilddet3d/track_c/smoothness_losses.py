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
# Spectral (frequency-domain) losses — L7, L8, L9a, L9b (doc §5)
# --------------------------------------------------------------------------

def _windowed_rfft_power(x: Tensor, window: int, hop: int, taper: Tensor) -> Tensor:
    """Hann-tapered windowed rFFT power. x: [T, C] -> P: [n_w, n_freq, C].
    Per-window mean-subtract, multiply by Hann, rFFT along time, |D|^2.
    n_freq = window // 2 + 1. Returns shape [0, n_freq, C] if T < window."""
    T, C = x.shape
    n_freq = window // 2 + 1
    if T < window:
        return x.new_zeros((0, n_freq, C))
    n_w = (T - window) // hop + 1
    starts = torch.arange(n_w, device=x.device) * hop
    idx = starts[:, None] + torch.arange(window, device=x.device)[None, :]  # [n_w, W]
    w = x[idx]                                            # [n_w, W, C]
    w = w - w.mean(dim=1, keepdim=True)
    w = w * taper[None, :, None]
    D = torch.fft.rfft(w, n=window, dim=1)                # [n_w, n_freq, C]
    return D.real ** 2 + D.imag ** 2


def spectral_smoothness_loss(
    pred_center: Tensor, gt_center: Tensor,
    sizes: Sequence[int], valid: Tensor,
    window: int = 16, hop: int = 8,
    cutoff_period_frames: float = 4.0,
    delta_mm: float = 50.0,
    w_l7: float = 0.0, w_l8: float = 0.0,
    w_l9a: float = 0.0, w_l9b: float = 0.0,
    l9b_margin: float = 0.05,
) -> dict:
    """Frequency-domain smoothness losses on velocity deviation d_t = v_pred - v_gt.

    Implements doc §5:
      L7  = E_high(d) / (E_total(d) + delta)               self-normalized HF fraction
      L8  = sum_k w_k P_k(d) / (E_total(d) + delta)        soft band edge (Hann ramp)
      L9a = E_high(d) / (E_total(v_gt) + delta)            GT-energy denominator (anti-gaming)
      L9b = ReLU(rho(v_pred) - rho(v_gt) - margin)         GT-referenced HF-ratio hinge

    Hann window (length ``window``) with 50% overlap (``hop=window/2``); per-window
    mean is subtracted before tapering. ``cutoff_period_frames`` sets the high-
    band start: k_c = window / cutoff_period_frames (e.g. window=16, cutoff=4 ->
    k_c=4 -> high band = periods < 4 frames). ``delta_mm`` is the eps stabilizer
    in mm/frame velocity units (squared internally to m^2).

    Skip trajectories with <window+1 frames or any invalid frames in their span.
    Returns zero loss if no trajectory contributes any window (no NaN).

    Monitors (doc §5 + §7): ``mon_e_low``, ``mon_e_high`` (raw deviation band
    energies — diverging while L_spec decreases = spectral gaming),
    ``mon_rho_pred``, ``mon_rho_gt`` (for L9b hinge interpretation).
    """
    if max(w_l7, w_l8, w_l9a, w_l9b) <= 0:
        z = pred_center.new_zeros(())
        return {"loss": z, "mon_e_low": z, "mon_e_high": z,
                "mon_rho_pred": z, "mon_rho_gt": z}

    dev, dtype = pred_center.device, pred_center.dtype
    taper = torch.hann_window(window, periodic=False, device=dev, dtype=dtype)
    n_freq = window // 2 + 1
    k_c = max(1, int(window / cutoff_period_frames))
    # L8 Hann-ramped band weights: 0->1 over one octave [k_c, 2*k_c], then 1.
    band_w = torch.zeros(n_freq, device=dev, dtype=dtype)
    ramp_end = min(2 * k_c, n_freq - 1)
    if ramp_end > k_c:
        for k in range(k_c + 1, ramp_end + 1):
            band_w[k] = (k - k_c) / (ramp_end - k_c)
    band_w[ramp_end + 1:] = 1.0
    delta = (delta_mm * 1e-3) ** 2

    terms_l7, terms_l8, terms_l9a, terms_l9b = [], [], [], []
    mon_e_low, mon_e_high, mon_rho_p, mon_rho_g = [], [], [], []
    for a, b in _iter_trajs(sizes):
        if b - a < window + 1:
            continue
        if valid is not None and not bool(valid[a:b].all()):
            continue
        v_pred = pred_center[a + 1:b] - pred_center[a:b - 1]
        v_gt = gt_center[a + 1:b] - gt_center[a:b - 1]
        d = v_pred - v_gt
        if d.shape[0] < window:
            continue
        P_d = _windowed_rfft_power(d, window, hop, taper)
        P_p = _windowed_rfft_power(v_pred, window, hop, taper)
        P_g = _windowed_rfft_power(v_gt, window, hop, taper)
        if P_d.shape[0] == 0:
            continue
        # exclude DC (k=0) from totals; high band = k > k_c
        E_total_d = P_d[:, 1:, :].sum(dim=1)            # [n_w, C]
        E_high_d  = P_d[:, k_c + 1:, :].sum(dim=1)
        E_low_d   = E_total_d - E_high_d
        E_high_w  = (P_d * band_w[None, :, None]).sum(dim=1)
        E_total_g = P_g[:, 1:, :].sum(dim=1)
        E_total_p = P_p[:, 1:, :].sum(dim=1)
        E_high_p  = P_p[:, k_c + 1:, :].sum(dim=1)
        E_high_g  = P_g[:, k_c + 1:, :].sum(dim=1)
        l7 = (E_high_d / (E_total_d + delta)).mean()
        l8 = (E_high_w / (E_total_d + delta)).mean()
        l9a = (E_high_d / (E_total_g + delta)).mean()
        rho_p = E_high_p / (E_total_p + delta)
        rho_g = E_high_g / (E_total_g + delta)
        l9b = torch.clamp(rho_p - rho_g - l9b_margin, min=0).mean()
        terms_l7.append(l7); terms_l8.append(l8)
        terms_l9a.append(l9a); terms_l9b.append(l9b)
        mon_e_low.append(E_low_d.mean().detach())
        mon_e_high.append(E_high_d.mean().detach())
        mon_rho_p.append(rho_p.mean().detach())
        mon_rho_g.append(rho_g.mean().detach())

    z = pred_center.new_zeros(())
    if not terms_l7:
        return {"loss": z, "mon_e_low": z, "mon_e_high": z,
                "mon_rho_pred": z, "mon_rho_gt": z}
    L7 = torch.stack(terms_l7).mean()
    L8v = torch.stack(terms_l8).mean()
    L9A = torch.stack(terms_l9a).mean()
    L9B = torch.stack(terms_l9b).mean()
    loss = w_l7 * L7 + w_l8 * L8v + w_l9a * L9A + w_l9b * L9B
    return {"loss": loss,
            "loss_l7": L7.detach(), "loss_l8": L8v.detach(),
            "loss_l9a": L9A.detach(), "loss_l9b": L9B.detach(),
            "mon_e_low": torch.stack(mon_e_low).mean(),
            "mon_e_high": torch.stack(mon_e_high).mean(),
            "mon_rho_pred": torch.stack(mon_rho_p).mean(),
            "mon_rho_gt": torch.stack(mon_rho_g).mean()}


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
    # Spectral family (doc §5) on velocity deviation d_t = v_pred - v_gt
    w_l7: float = 0.0,           # self-normalized HF energy fraction
    w_l8: float = 0.0,           # soft band edge (Hann ramp)
    w_l9a: float = 0.0,          # GT-energy denominator (anti-gaming)
    w_l9b: float = 0.0,          # GT-referenced HF-ratio hinge
    spec_window: int = 16,
    spec_hop: int = 8,
    spec_cutoff_period: float = 4.0,
    spec_delta_mm: float = 50.0,
    spec_l9b_margin: float = 0.05,
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
    if max(w_l7, w_l8, w_l9a, w_l9b) > 0:
        sp = spectral_smoothness_loss(
            pred_center, gt_center, sizes, valid,
            window=spec_window, hop=spec_hop,
            cutoff_period_frames=spec_cutoff_period,
            delta_mm=spec_delta_mm,
            w_l7=w_l7, w_l8=w_l8, w_l9a=w_l9a, w_l9b=w_l9b,
            l9b_margin=spec_l9b_margin,
        )
        total = total + sp["loss"]
        if "loss_l7" in sp:
            out["loss_l7"] = sp["loss_l7"]; out["loss_l8"] = sp["loss_l8"]
            out["loss_l9a"] = sp["loss_l9a"]; out["loss_l9b"] = sp["loss_l9b"]
        out["mon_spec_e_low"] = sp["mon_e_low"]
        out["mon_spec_e_high"] = sp["mon_e_high"]
        out["mon_spec_rho_pred"] = sp["mon_rho_pred"]
        out["mon_spec_rho_gt"] = sp["mon_rho_gt"]
    out["loss"] = total
    return out
