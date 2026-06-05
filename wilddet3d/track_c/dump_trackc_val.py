"""Dump Track C per-frame val predictions for the Track A (Kalman/RTS) smoother.

For each held-out trajectory (same by-video val split, seed 0, val_frac 0.08),
save the per-frame Track C predicted box, the Step-4 prior box, and the GT box —
all CAMERA frame (center, dims, R) — plus frame_index (-> Step-1 extrinsics for
cam<->world) and ts. Track A: cam->world via step1 extrinsics, RTS-smooth in
world, world->cam, eval accuracy + jitter vs GT.
"""
import sys, argparse, numpy as np, torch
WD = "/weka/oe-training-default/weikaih/3d_boundingbox_detection/video_3d_box/itw_3dbox_det/WildDet3D"
sys.path.insert(0, WD)
from vis4d.op.geometry.rotation import quaternion_to_matrix
from wilddet3d.ops.rotation import rotation_6d_to_matrix
from wilddet3d.track_c import TrackCRefiner
from wilddet3d.track_c.dataset import CachedTrackCDataset, list_cached_trajectories, split_by_video

ap = argparse.ArgumentParser()
ap.add_argument("--cache_dir", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--val_frac", type=float, default=0.08)
ap.add_argument("--reg_residual_from_prior", type=int, default=0)
a = ap.parse_args()
dev = "cuda"

paths = list_cached_trajectories(a.cache_dir)
_, val_paths, val_vids = split_by_video(paths, a.val_frac)
ds = CachedTrackCDataset(a.cache_dir, val_paths, preload=True)
r = TrackCRefiner(reg_residual_from_prior=bool(a.reg_residual_from_prior)).to(dev)
r.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=False)["refiner"], strict=True)
r.eval()

out = []
with torch.no_grad():
    for i in range(len(ds)):
        p = ds[i]; traj = ds.traj_cache[i]
        g = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in p.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = r(hidden_states=g["hidden"].float(), ray_embeddings=g["ray"].float(),
                  depth_latents=g["depth"].float(), pred_box_2d=g["box2d"], intrinsics=g["K"],
                  box_repr=g["box_repr"], timestamps=g["ts"], measured_mask=g["measured"],
                  input_hw=g["input_hw"])
        dec = r.decode_layer(o["reg"][-1, :, 0, :].float(), g["box2d"], g["K"], g["input_hw"])
        br = g["box_repr"].float()
        out.append(dict(
            video_id=p["video_id"], object_id=p["object_id"],
            frame_index=traj["frame_index"].cpu().numpy(),
            ts_sec=p["ts"].cpu().numpy(),
            measured=p["measured"].cpu().numpy(),
            pred_center=dec[:, 0:3].cpu().numpy(),
            pred_dims=dec[:, 3:6].cpu().numpy(),
            pred_R=quaternion_to_matrix(dec[:, 6:10]).cpu().numpy(),
            input_center=br[:, 0:3].cpu().numpy(),
            input_dims=torch.exp(br[:, 3:6]).cpu().numpy(),
            input_R=rotation_6d_to_matrix(br[:, 6:12]).cpu().numpy(),
            gt_center=p["gt_center"].cpu().numpy(),
            gt_dims=p["gt_dims"].cpu().numpy(),
            gt_R=quaternion_to_matrix(p["gt_quat"].float()).cpu().numpy(),
        ))
torch.save({"trajectories": out, "val_videos": val_vids,
            "source_ckpt": a.ckpt, "frame": "camera (OpenCV); R maps box-local->camera",
            "note": "frame_index indexes Step-1 arrays (extrinsics/intrinsics/depth) per video"}, a.out)
print(f"dumped {len(out)} val trajectories -> {a.out}")
