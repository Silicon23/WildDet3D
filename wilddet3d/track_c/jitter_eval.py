"""Jitter / smoothness eval for Track C, aligned with Track B's trajectory_jitter.

Per held-out trajectory (frame-ordered) compute the 2nd-difference (acceleration)
magnitude of center (m), rotation step (deg), and dims (m), for each box source:
  input (Step-4 prior) / track_c (model) / GT (the smoothness floor).
Averaged over trajectories. Run for a given checkpoint.
"""
import sys, argparse, numpy as np, torch
WD = "/weka/oe-training-default/weikaih/3d_boundingbox_detection/video_3d_box/itw_3dbox_det/WildDet3D"
sys.path.insert(0, WD)
from vis4d.op.geometry.rotation import quaternion_to_matrix
from wilddet3d.ops.rotation import rotation_6d_to_matrix
from wilddet3d.track_c import TrackCRefiner
from wilddet3d.track_c.dataset import CachedTrackCDataset, list_cached_trajectories, split_by_video


def geodesic_deg(Ra, Rb):
    rel = Ra.T @ Rb
    cos = (np.trace(rel) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(cos, -1 + 1e-9, 1 - 1e-9))))


def traj_jitter(center, dims, R):
    """center[T,3], dims[T,3], R[T,3,3] -> (center_jit_m, rot_jit_deg, dims_jit_m, n)."""
    T = len(center)
    if T < 3:
        return np.nan, np.nan, np.nan, 0
    acc = center[2:] - 2 * center[1:-1] + center[:-2]
    center_jit = float(np.linalg.norm(acc, axis=-1).mean())
    dacc = dims[2:] - 2 * dims[1:-1] + dims[:-2]
    dims_jit = float(np.linalg.norm(dacc, axis=-1).mean())
    w = np.array([geodesic_deg(R[t], R[t + 1]) for t in range(T - 1)])
    rot_jit = float(np.abs(np.diff(w)).mean()) if len(w) >= 2 else np.nan
    return center_jit, rot_jit, dims_jit, T - 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--ckpt", required=True)              # best.pt with ["refiner"]
    ap.add_argument("--reg_residual_from_prior", type=int, default=0)
    ap.add_argument("--val_frac", type=float, default=0.08)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device

    paths = list_cached_trajectories(args.cache_dir)
    _, val_paths, _ = split_by_video(paths, args.val_frac)   # seed=0 default -> same val set
    ds = CachedTrackCDataset(args.cache_dir, val_paths, preload=True)

    refiner = TrackCRefiner(reg_residual_from_prior=bool(args.reg_residual_from_prior)).to(dev)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["refiner"]
    refiner.load_state_dict(sd, strict=True)
    refiner.eval()

    agg = {s: {"c": [], "r": [], "d": []} for s in ("input", "track_c", "gt")}
    with torch.no_grad():
        for i in range(len(ds)):
            p = ds[i]
            if p["box_repr"].shape[0] < 3:
                continue
            g = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in p.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = refiner(
                    hidden_states=g["hidden"].float(), ray_embeddings=g["ray"].float(),
                    depth_latents=g["depth"].float(), pred_box_2d=g["box2d"],
                    intrinsics=g["K"], box_repr=g["box_repr"], timestamps=g["ts"],
                    measured_mask=g["measured"], input_hw=g["input_hw"])
            dec = refiner.decode_layer(out["reg"][-1, :, 0, :].float(), g["box2d"], g["K"], g["input_hw"])
            tc_c = dec[:, 0:3].cpu().numpy(); tc_d = dec[:, 3:6].cpu().numpy()
            tc_R = quaternion_to_matrix(dec[:, 6:10]).cpu().numpy()
            # input prior from box_repr
            br = g["box_repr"].float()
            in_c = br[:, 0:3].cpu().numpy(); in_d = torch.exp(br[:, 3:6]).cpu().numpy()
            in_R = rotation_6d_to_matrix(br[:, 6:12]).cpu().numpy()
            # GT
            gt_c = g["gt_center"].float().cpu().numpy(); gt_d = g["gt_dims"].float().cpu().numpy()
            gt_R = quaternion_to_matrix(g["gt_quat"].float()).cpu().numpy()
            for s, (c, d, R) in (("input", (in_c, in_d, in_R)),
                                 ("track_c", (tc_c, tc_d, tc_R)),
                                 ("gt", (gt_c, gt_d, gt_R))):
                cj, rj, dj, n = traj_jitter(c, d, R)
                if n > 0:
                    agg[s]["c"].append(cj); agg[s]["r"].append(rj); agg[s]["d"].append(dj)
    print(f"jitter over {len(agg['gt']['c'])} val trajectories (mean +/- per-traj):")
    print(f"{'source':9} {'center_jit_m':>13} {'rot_jit_deg':>12} {'dims_jit_m':>11}")
    for s in ("input", "track_c", "gt"):
        c = np.nanmean(agg[s]["c"]); r = np.nanmean(agg[s]["r"]); d = np.nanmean(agg[s]["d"])
        print(f"{s:9} {c:13.4f} {r:12.3f} {d:11.4f}")


if __name__ == "__main__":
    main()
