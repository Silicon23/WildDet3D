"""Pattern-resolving smoothness eval for Track C (trajectory_smoothness_losses.md §7).

Extends jitter_eval's energy metrics with loss-independent *pattern* metrics on the
velocity deviation d_t = v̂_t − v_t (first difference of the residual):

  position (and rotation, via world-frame so(3) log increments):
    - lag-k mean cosine of consecutive deviations, k ∈ {1,2,4,8}   (anticorrelation
      = jitter signature; ~0 = healthy; negative = zigzag at that timescale)
    - lag-1 flip fraction (cos < -0.2)
    - high-band energy fraction of d_t (Hann window, 50% overlap) at period
      cutoffs <=4 and <=8 frames
    - ||d_t|| percentiles (the empirical noise floor -> delta anchors)
  rotation extras:
    - prediction flip rate (||omega_hat|| > 120 deg), gate pass rate (< 150 deg)
    - GT flip rate (M3: must be ~0)
  GT-noise anchor:
    - sigma_gt estimate: residual of GT center after light smoothing (per axis)

Legacy energy metrics (2nd-diff center/dims, |Delta rot step|) are kept verbatim
for continuity with all previous runs.

Usage (one checkpoint):
  python -m wilddet3d.track_c.smoothness_eval --cache_dir ... --ckpt .../best.pt \
      [arch flags as in jitter_eval] --out /path/metrics.json
"""
import sys, os, json, argparse, math
import numpy as np
import torch

WD = "/weka/oe-training-default/weikaih/3d_boundingbox_detection/video_3d_box/itw_3dbox_det/WildDet3D"
sys.path.insert(0, WD)
from vis4d.op.geometry.rotation import quaternion_to_matrix
from wilddet3d.ops.rotation import rotation_6d_to_matrix
from wilddet3d.track_c import TrackCRefiner
from wilddet3d.track_c.dataset import (
    CachedTrackCDataset, list_cached_trajectories, split_by_video)

LAGS = (1, 2, 4, 8)
POS_FLOOR = 1e-3        # 1 mm/frame: below this, deviation direction is noise
ROT_FLOOR = math.radians(0.05)   # 0.05 deg/frame
GATE_DEG, FLIP_DEG = 150.0, 120.0


# ---------- so(3) log, atan2 form, Taylor-guarded (magnitude + vector) ----------

def so3_log(R):
    """R [...,3,3] -> omega [...,3] (axis*angle, radians). atan2 form, stable
    through pi for magnitude; vector direction Taylor-guarded near 0."""
    s = 0.5 * np.stack([R[..., 2, 1] - R[..., 1, 2],
                        R[..., 0, 2] - R[..., 2, 0],
                        R[..., 1, 0] - R[..., 0, 1]], axis=-1)        # sin(th)*axis
    sn = np.linalg.norm(s, axis=-1)                                    # |sin th|
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cs = 0.5 * (tr - 1.0)
    th = np.arctan2(sn, cs)                                            # [0, pi]
    # omega = s * th/sin(th); Taylor near 0; near pi magnitude is right but the
    # s-based axis degrades -- fine here (gated frames exclude >150 deg patterns).
    scale = np.where(sn > 1e-7, th / np.maximum(sn, 1e-12), 1.0 + th * th / 6.0)
    return s * scale[..., None], th


# ---------- pattern metrics on a deviation sequence d [T,3] ----------

def lag_cosines(d, floor):
    out = {}
    n = np.linalg.norm(d, axis=-1)
    for k in LAGS:
        if len(d) <= k:
            out[f"lag{k}_cos"] = np.nan
            if k == 1:
                out["lag1_flip_frac"] = np.nan
            continue
        a, b = d[:-k], d[k:]
        na, nb = n[:-k], n[k:]
        m = (na > floor) & (nb > floor)
        if m.sum() < 2:
            out[f"lag{k}_cos"] = np.nan
            if k == 1:
                out["lag1_flip_frac"] = np.nan
            continue
        cos = (a[m] * b[m]).sum(-1) / (na[m] * nb[m])
        out[f"lag{k}_cos"] = float(cos.mean())
        if k == 1:
            out["lag1_flip_frac"] = float((cos < -0.2).mean())
    return out


def highband_fracs(d, w=32, return_per_window=False):
    """Energy fraction of d above period cutoffs (<=4, <=8 frames).
    Hann window, 50% overlap; full-length window if T < w.

    If ``return_per_window=True``, also returns a list of per-window p4 ratios
    (one per analysis window across the trajectory). Long tail in the aggregate
    distribution of these ratios (across all val trajectories) = burst-concentrated
    jitter, signaling that wavelet (L10) localization would help.
    """
    T = len(d)
    if T < 6:
        if return_per_window:
            return {"hb_frac_p4": np.nan, "hb_frac_p8": np.nan, "per_window_p4": []}
        return {"hb_frac_p4": np.nan, "hb_frac_p8": np.nan}
    W = min(w, T)
    hop = max(W // 2, 1)
    hann = np.hanning(W)
    e_tot = e_p4 = e_p8 = 0.0
    per_window_p4 = []   # per-window ratio (each: window's HF energy / window's total)
    for s0 in range(0, max(T - W, 0) + 1, hop):
        seg = d[s0:s0 + W]                                  # [W,3]
        seg = seg - seg.mean(0, keepdims=True)
        spec = np.fft.rfft(seg * hann[:, None], axis=0)     # [W//2+1, 3]
        P = (spec.real ** 2 + spec.imag ** 2).sum(-1)       # [K]
        k = np.arange(len(P))
        win_tot = float(P[1:].sum())
        win_p4 = float(P[k >= max(W / 4.0, 1)].sum())
        e_tot += win_tot
        e_p4 += win_p4
        e_p8 += P[k >= max(W / 8.0, 1)].sum()               # period <= 8 frames
        if win_tot > 1e-12:
            per_window_p4.append(win_p4 / win_tot)
    if e_tot <= 1e-12:
        if return_per_window:
            return {"hb_frac_p4": np.nan, "hb_frac_p8": np.nan, "per_window_p4": []}
        return {"hb_frac_p4": np.nan, "hb_frac_p8": np.nan}
    out = {"hb_frac_p4": float(e_p4 / e_tot), "hb_frac_p8": float(e_p8 / e_tot)}
    if return_per_window:
        out["per_window_p4"] = per_window_p4
    return out


def pos_pattern(pred_c, gt_c):
    d = np.diff(pred_c - gt_c, axis=0)                      # [T-1,3] vel deviation
    out = lag_cosines(d, POS_FLOOR)
    hb = highband_fracs(d, return_per_window=True)
    per_window_p4 = hb.pop("per_window_p4")
    out.update(hb)
    out["_per_window_p4"] = per_window_p4   # consumed by main() for the L10-go-signal histogram
    n = np.linalg.norm(d, axis=-1)
    for p in (10, 25, 50, 75, 90):
        out[f"dmag_p{p}_mm"] = float(np.percentile(n, p) * 1000) if len(n) else np.nan
    return out


def rot_pattern(pred_R, gt_R):
    """World-frame UNFOLDED increments; pattern on gated frames."""
    wp, thp = so3_log(pred_R[1:] @ np.swapaxes(pred_R[:-1], -1, -2))
    wg, thg = so3_log(gt_R[1:] @ np.swapaxes(gt_R[:-1], -1, -2))
    out = {
        "flip_rate": float((np.degrees(thp) > FLIP_DEG).mean()),
        "gate_pass": float((np.degrees(thp) < GATE_DEG).mean()),
        "gt_flip_rate": float((np.degrees(thg) > 90.0).mean()),
    }
    gate = (np.degrees(thp) < GATE_DEG) & (np.degrees(thg) < GATE_DEG)
    d = (wp - wg)[gate]
    out.update(lag_cosines(d, ROT_FLOOR))
    out.update(highband_fracs(d))
    n = np.degrees(np.linalg.norm(d, axis=-1))
    for p in (10, 25, 50, 75, 90):
        out[f"dmag_p{p}_deg"] = float(np.percentile(n, p)) if len(n) else np.nan
    return out


# ---------- legacy energy metrics (jitter_eval, verbatim semantics) ----------

def geodesic_deg_seq(R):
    rel = np.swapaxes(R[:-1], -1, -2) @ R[1:]
    tr = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    return np.degrees(np.arccos(np.clip((tr - 1) * 0.5, -1 + 1e-9, 1 - 1e-9)))


def energy_jitter(c, dims, R):
    T = len(c)
    if T < 3:
        return np.nan, np.nan, np.nan
    cj = float(np.linalg.norm(c[2:] - 2 * c[1:-1] + c[:-2], axis=-1).mean())
    dj = float(np.linalg.norm(dims[2:] - 2 * dims[1:-1] + dims[:-2], axis=-1).mean())
    w = geodesic_deg_seq(R)
    rj = float(np.abs(np.diff(w)).mean()) if len(w) >= 2 else np.nan
    return cj, rj, dj


def gt_sigma(gt_c):
    """sigma_gt per frame: residual std of GT center after light smoothing (mm)."""
    T = len(gt_c)
    if T < 9:
        return np.nan
    k = np.hanning(7); k /= k.sum()
    sm = np.stack([np.convolve(gt_c[:, a], k, mode="same") for a in range(3)], -1)
    r = (gt_c - sm)[3:-3]                                   # drop edge effects
    return float(np.sqrt((r ** 2).sum(-1).mean()) * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--reg_residual_from_prior", type=int, default=0)
    ap.add_argument("--no_traj_encoder", type=int, default=0)
    ap.add_argument("--use_layer_bias", type=int, default=0)
    ap.add_argument("--temporal_kv_norm", type=int, default=0)
    ap.add_argument("--temporal_multi_token", type=int, default=0)
    ap.add_argument("--val_frac", type=float, default=0.08)
    ap.add_argument("--val_split_file", default=None,
                    help="override CA-1M canonical split (pass Waymo/ADT split or '' for RNG)")
    ap.add_argument("--categories", default="", help="comma list to keep; needs --pairing_index")
    ap.add_argument("--pairing_index", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="", help="write metrics JSON here")
    args = ap.parse_args()
    dev = args.device

    paths = list_cached_trajectories(args.cache_dir)
    if args.categories and args.pairing_index:
        keep = set(c.strip() for c in args.categories.split(",") if c.strip())
        cat_of = {}
        for line in open(args.pairing_index):
            d = json.loads(line); cat_of[f"{d['seg']}__{d['track_id']}"] = d["category"]
        paths = [p for p in paths if cat_of.get(os.path.basename(p)[:-3]) in keep]
    split_kw = {} if args.val_split_file is None else {"split_file": args.val_split_file}
    _, val_paths, _ = split_by_video(paths, args.val_frac, **split_kw)
    ds = CachedTrackCDataset(args.cache_dir, val_paths, preload=True)

    refiner = TrackCRefiner(
        reg_residual_from_prior=bool(args.reg_residual_from_prior),
        use_temporal_modules=not bool(args.no_traj_encoder),
        use_layer_bias=bool(args.use_layer_bias),
        use_temporal_kv_norm=bool(args.temporal_kv_norm),
        temporal_multi_token=bool(args.temporal_multi_token),
    ).to(dev)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["refiner"]
    refiner.load_state_dict(sd, strict=True)
    refiner.eval()

    rows = {s: [] for s in ("input", "track_c")}
    legacy = {s: [] for s in ("input", "track_c", "gt")}
    sigmas = []
    with torch.no_grad():
        for i in range(len(ds)):
            p = ds[i]
            if p["box_repr"].shape[0] < 6:
                continue
            g = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in p.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = refiner(
                    hidden_states=g["hidden"].float(), ray_embeddings=g["ray"].float(),
                    depth_latents=g["depth"].float(), pred_box_2d=g["box2d"],
                    intrinsics=g["K"], box_repr=g["box_repr"], timestamps=g["ts"],
                    measured_mask=g["measured"], input_hw=g["input_hw"])
            dec = refiner.decode_layer(out["reg"][-1, :, 0, :].float(),
                                       g["box2d"], g["K"], g["input_hw"])
            tc_c = dec[:, 0:3].cpu().numpy(); tc_d = dec[:, 3:6].cpu().numpy()
            tc_R = quaternion_to_matrix(dec[:, 6:10]).cpu().numpy()
            br = g["box_repr"].float()
            in_c = br[:, 0:3].cpu().numpy(); in_d = torch.exp(br[:, 3:6]).cpu().numpy()
            in_R = rotation_6d_to_matrix(br[:, 6:12]).cpu().numpy()
            gt_c = g["gt_center"].float().cpu().numpy()
            gt_d = g["gt_dims"].float().cpu().numpy()
            gt_R = quaternion_to_matrix(g["gt_quat"].float()).cpu().numpy()

            sigmas.append(gt_sigma(gt_c))
            for s, (c, d, R) in (("input", (in_c, in_d, in_R)),
                                 ("track_c", (tc_c, tc_d, tc_R))):
                row = {f"pos_{k}": v for k, v in pos_pattern(c, gt_c).items()}
                row.update({f"rot_{k}": v for k, v in rot_pattern(R, gt_R).items()})
                rows[s].append(row)
            for s, (c, d, R) in (("input", (in_c, in_d, in_R)),
                                 ("track_c", (tc_c, tc_d, tc_R)),
                                 ("gt", (gt_c, gt_d, gt_R))):
                cj, rj, dj = energy_jitter(c, d, R)
                legacy[s].append((cj, rj, dj))

    res = {"n_traj": len(rows["track_c"]),
           "gt_sigma_mm_median": float(np.nanmedian(sigmas))}
    for s in ("input", "track_c", "gt"):
        L = np.array(legacy[s], dtype=float)
        res[f"{s}/center_jit_m"] = float(np.nanmean(L[:, 0]))
        res[f"{s}/rot_jit_deg"] = float(np.nanmean(L[:, 1]))
        res[f"{s}/dims_jit_m"] = float(np.nanmean(L[:, 2]))
    # Collect per-window HF ratios from every trajectory (the L10-go-signal:
    # long tail in this distribution = burst-concentrated jitter; flat = persistent)
    per_window_all = {"input": [], "track_c": []}
    for s in ("input", "track_c"):
        for r in rows[s]:
            per_window_all[s].extend(r.pop("_per_window_p4", []))
        keys = rows[s][0].keys() if rows[s] else []
        for k in keys:
            res[f"{s}/{k}"] = float(np.nanmean([r[k] for r in rows[s]]))
        pw = np.array(per_window_all[s], dtype=float)
        if pw.size > 0:
            for p in (50, 75, 90, 95, 99):
                res[f"{s}/pw_hb_p4_p{p}"] = float(np.percentile(pw, p))
            res[f"{s}/pw_hb_p4_max"] = float(pw.max())
            res[f"{s}/pw_hb_p4_n_windows"] = int(pw.size)
            # tail metric: p95 / p50. Persistent jitter -> ~1.0-1.5; bursty -> > 2.5
            p50 = float(np.percentile(pw, 50))
            res[f"{s}/pw_hb_p4_p95_over_p50"] = (float(np.percentile(pw, 95)) /
                                                 max(p50, 1e-9))

    print(f"\n=== smoothness eval: {args.ckpt} ({res['n_traj']} val trajs) ===")
    print(f"GT center noise floor sigma_gt ~ {res['gt_sigma_mm_median']:.2f} mm/frame")
    print(f"\n{'metric':34} {'input':>10} {'track_c':>10}")
    show = (["pos_lag1_cos", "pos_lag2_cos", "pos_lag4_cos", "pos_lag8_cos",
             "pos_lag1_flip_frac", "pos_hb_frac_p4", "pos_hb_frac_p8",
             "pos_dmag_p50_mm", "pos_dmag_p25_mm"]
            + ["rot_lag1_cos", "rot_lag2_cos", "rot_hb_frac_p4",
               "rot_flip_rate", "rot_gate_pass", "rot_gt_flip_rate",
               "rot_dmag_p50_deg", "rot_dmag_p25_deg"])
    for k in show:
        print(f"{k:34} {res.get('input/'+k, np.nan):>10.4f} "
              f"{res.get('track_c/'+k, np.nan):>10.4f}")
    # L10-go-signal: per-window HF-ratio distribution. p95/p50 ratio is the
    # tail-heaviness indicator — flat (~1) = persistent jitter (L7-L9 work);
    # heavy tail (>2.5) = burst-concentrated jitter (L10 wavelet would localize).
    print(f"\nper-window HF(p4) ratio distribution (L10-go-signal):")
    print(f"  {'source':10} {'n_win':>8} {'p50':>7} {'p75':>7} {'p90':>7} {'p95':>7} {'p99':>7} "
          f"{'max':>7} {'p95/p50':>8}")
    for s in ("input", "track_c"):
        n_win = res.get(f"{s}/pw_hb_p4_n_windows", 0)
        if n_win > 0:
            print(f"  {s:10} {n_win:>8d} "
                  f"{res[f'{s}/pw_hb_p4_p50']:>7.3f} {res[f'{s}/pw_hb_p4_p75']:>7.3f} "
                  f"{res[f'{s}/pw_hb_p4_p90']:>7.3f} {res[f'{s}/pw_hb_p4_p95']:>7.3f} "
                  f"{res[f'{s}/pw_hb_p4_p99']:>7.3f} {res[f'{s}/pw_hb_p4_max']:>7.3f} "
                  f"{res[f'{s}/pw_hb_p4_p95_over_p50']:>8.2f}")
    print(f"\nlegacy energy:  center_jit  rot_jit  dims_jit")
    for s in ("input", "track_c", "gt"):
        print(f"  {s:8} {res[f'{s}/center_jit_m']:>10.4f} "
              f"{res[f'{s}/rot_jit_deg']:>8.3f} {res[f'{s}/dims_jit_m']:>9.4f}")
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=1)
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
