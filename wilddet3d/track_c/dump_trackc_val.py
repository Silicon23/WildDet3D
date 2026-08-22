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

import os, json
ap = argparse.ArgumentParser()
ap.add_argument("--cache_dir", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--val_frac", type=float, default=0.08)
ap.add_argument("--reg_residual_from_prior", type=int, default=0)
ap.add_argument("--no_traj_encoder", type=int, default=0)
ap.add_argument("--temporal_kv_norm", type=int, default=0)
ap.add_argument("--temporal_multi_token", type=int, default=0)
ap.add_argument("--temporal_block", default="prompt3d",
                choices=("prompt3d", "xattn_only"),
                help="'xattn_only' for WX3-era checkpoints")
ap.add_argument("--use_layer_bias", type=int, default=0)
ap.add_argument("--device", default="cuda", help="cpu fallback when GPU is incompatible")
ap.add_argument("--val_split_file", default=None, help="Waymo/ADT split override; ''=RNG")
ap.add_argument("--categories", default="", help="comma list to keep; needs --pairing_index")
ap.add_argument("--pairing_index", default="")
ap.add_argument("--extrinsics_note", default="frame_index indexes Step-1 arrays per video")
a = ap.parse_args()
dev = a.device

paths = list_cached_trajectories(a.cache_dir)
if a.categories and a.pairing_index:
    keep = set(c.strip() for c in a.categories.split(",") if c.strip())
    cat_of = {}
    for line in open(a.pairing_index):
        d = json.loads(line); cat_of[f"{d['seg']}__{d['track_id']}"] = d["category"]
    paths = [p for p in paths if cat_of.get(os.path.basename(p)[:-3]) in keep]
split_kw = {} if a.val_split_file is None else {"split_file": a.val_split_file}
_, val_paths, val_vids = split_by_video(paths, a.val_frac, **split_kw)
ds = CachedTrackCDataset(a.cache_dir, val_paths, preload=True)
r = TrackCRefiner(reg_residual_from_prior=bool(a.reg_residual_from_prior),
                  use_temporal_modules=not bool(a.no_traj_encoder),
                  use_layer_bias=bool(a.use_layer_bias),
                  use_temporal_kv_norm=bool(a.temporal_kv_norm),
                  temporal_multi_token=bool(a.temporal_multi_token),
                  temporal_block=a.temporal_block).to(dev)
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
            "note": a.extrinsics_note}, a.out)
print(f"dumped {len(out)} val trajectories -> {a.out}")
