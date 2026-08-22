"""Rewrite the `box_repr` array in cached trajectory files to use GT boxes.

Motivation: on Waymo the Step-4 FoundationPose prior has median center error
5-13 m per frame — the temporal encoder gets noise. This script substitutes the
trajectory encoder's input with the CAMERA-FRAME GT box, keeping every other
field of the cached traj file unchanged. Purpose: validate whether temporal
engages when given a clean trajectory. Not for shipping.

Reads the existing traj_*.pt (fast — the cache already exists) and writes a
sibling cache dir with `box_repr` replaced by an encoding of (gt_center,
gt_dims, gt_R). Also flips the `measured` mask to True everywhere (GT is
always "measured" in this substituted regime). Every other tensor (hidden,
box2d, ts_sec, K, gt_*, category, video_id, object_id) copies through.

Result: a cache identical to the original except the temporal-encoder input
is the ground-truth trajectory. Retraining on this and running the
`temporal_diagnostic.py` intervention tests is Option B.

Usage:
  python -m wilddet3d.track_c.rewrite_traj_with_gt \
      --src  outputs/track_c_waymo_gt2d_gtk_lidar_feature_cache \
      --dst  outputs/track_c_waymo_gt2d_gtk_lidar_GTTRAJ_feature_cache
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil

import numpy as np
import torch

from wilddet3d.track_c.data import _camera_box_repr


def rewrite_one_traj(src_path: str, dst_path: str) -> tuple[int, int]:
    """Copy src -> dst with box_repr replaced by GT encoding. Returns (T, n_masked)."""
    t = torch.load(src_path, weights_only=False)
    T = int(t["frame_index"].shape[0])

    gt_c = t["gt_center"].float().numpy()      # [T,3]
    gt_d = t["gt_dims"].float().numpy()        # [T,3]
    gt_q = t["gt_quat"].float().numpy()        # [T,4] wxyz

    # wxyz -> R
    def _R_from_q(q):
        w, x, y, z = q
        n = w*w + x*x + y*y + z*z
        if n < 1e-12:
            return np.eye(3, dtype=np.float32)
        s = 2.0 / n
        return np.array([
            [1 - s*(y*y + z*z), s*(x*y - w*z),     s*(x*z + w*y)],
            [s*(x*y + w*z),     1 - s*(x*x + z*z), s*(y*z - w*x)],
            [s*(x*z - w*y),     s*(y*z + w*x),     1 - s*(x*x + y*y)],
        ], dtype=np.float32)

    # Rebuild box_repr from GT for every frame. If GT is invalid (z <= 0.1)
    # we fall back to the original box_repr (no signal to inject either way).
    box_repr = np.zeros((T, 12), dtype=np.float32)
    n_gt = 0
    for i in range(T):
        if gt_c[i, 2] > 0.1 and (gt_d[i] > 1e-4).all():
            box_repr[i] = _camera_box_repr(gt_c[i], gt_d[i], _R_from_q(gt_q[i]))
            n_gt += 1
        else:
            box_repr[i] = t["box_repr"][i].numpy()

    t["box_repr"] = torch.from_numpy(box_repr)
    # GT is always "measured" wherever it's valid; keep the original measured
    # mask for GT-invalid frames.
    orig_measured = t["measured"].bool().numpy()
    new_measured = np.array([True if gt_c[i, 2] > 0.1 else orig_measured[i]
                             for i in range(T)], dtype=bool)
    t["measured"] = torch.from_numpy(new_measured)
    t["variant"] = "GTTRAJ"     # provenance stamp

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    tmp = dst_path + ".tmp"
    torch.save(t, tmp)
    os.replace(tmp, dst_path)
    return T, n_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="existing feature-cache dir")
    ap.add_argument("--dst", required=True, help="new cache dir (created)")
    args = ap.parse_args()

    if os.path.exists(args.dst):
        raise SystemExit(f"{args.dst} exists — refusing to overwrite")
    os.makedirs(f"{args.dst}/traj", exist_ok=True)

    # frame files: symlink (they're identical — we don't touch the visual features)
    os.makedirs(f"{args.dst}/frames", exist_ok=True)
    for f in glob.glob(f"{args.src}/frames/*.pt"):
        rel = os.path.basename(f)
        try:
            os.symlink(f, f"{args.dst}/frames/{rel}")
        except FileExistsError:
            pass

    # rewrite every traj
    trajs = sorted(glob.glob(f"{args.src}/traj/*.pt"))
    print(f"[gttraj] rewriting {len(trajs)} traj files from {args.src} -> {args.dst}", flush=True)
    total_T, total_gt = 0, 0
    for i, src in enumerate(trajs):
        rel = os.path.basename(src)
        dst = f"{args.dst}/traj/{rel}"
        T, n_gt = rewrite_one_traj(src, dst)
        total_T += T; total_gt += n_gt
        if (i + 1) % 100 == 0:
            print(f"[gttraj] {i+1}/{len(trajs)} ({100*total_gt/max(total_T,1):.1f}% GT-valid frames so far)", flush=True)

    # mirror .done markers so dataset.split_by_video works
    os.makedirs(f"{args.dst}/.done", exist_ok=True)
    for f in glob.glob(f"{args.src}/.done/*"):
        rel = os.path.basename(f)
        try:
            open(f"{args.dst}/.done/{rel}", "w").close()
        except FileExistsError:
            pass

    print(f"[gttraj] DONE — {len(trajs)} trajs, "
          f"{100*total_gt/max(total_T,1):.1f}% frames use GT box_repr")


if __name__ == "__main__":
    main()
