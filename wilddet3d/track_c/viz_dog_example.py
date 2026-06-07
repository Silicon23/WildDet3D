"""Render single-color 3D-box overlays on the dog-example.

Writes two mp4s at the native video fps:
  viz_raw.mp4       : red box   = Track C v11 per-frame (pre-smoother)
  viz_smoothed.mp4  : green box = Track C v11 + Track A Kalman/RTS

Each draws ONLY one box per frame (no two-color overlay) so the two videos
can be compared side-by-side or used as standalones.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


_OBB_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),
              (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7)]


def project(pts_cam: np.ndarray, K: np.ndarray):
    z = np.clip(pts_cam[:, 2], 1e-6, None)
    u = K[0, 0] * pts_cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * pts_cam[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=1), pts_cam[:, 2] > 1e-6


def draw_box(img, corners_cam: np.ndarray, K: np.ndarray, color):
    uv, valid = project(corners_cam, K)
    for a, b in _OBB_EDGES:
        if valid[a] and valid[b]:
            cv2.line(img, tuple(np.round(uv[a]).astype(int)),
                     tuple(np.round(uv[b]).astype(int)),
                     color, 2, cv2.LINE_AA)


def render(meta_path: Path, step1_dir: Path, out_mp4: Path, color, label: str,
           fps: float):
    """Render a single-color overlay video from a step-4-shaped meta.json."""
    meta = json.loads(meta_path.read_text())
    frames = meta["frames"]
    extr = np.load(step1_dir / "extrinsics.npy")  # world->cam, [N,4,4]
    K_all = np.load(step1_dir / "intrinsics.npy")  # [N,3,3]

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    n_drawn = 0
    for f in frames:
        if f.get("status") and f["status"] != "ok":
            continue
        # Raw (Track C v11) frames have step1_index / source_frame_index +
        # corners under f["box"]. Smoothed (Track A) frames have frame_index
        # at the top level + corners directly on f.
        i = f.get("step1_index", f.get("source_frame_index", f.get("frame_index")))
        if i is None:
            continue
        i = int(i)
        img_p = step1_dir / "frames" / f"{i:06d}.jpg"
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        K = K_all[i].astype(np.float64)
        w2c = extr[i].astype(np.float64)
        # corners_world is the canonical key on both raw and smoothed metas
        box = f.get("box", f)
        if box.get("corners_world") is None:
            continue
        corners_world = np.asarray(box["corners_world"], dtype=np.float64)
        corners_cam = (np.concatenate([corners_world, np.ones((8, 1))], 1)
                       @ w2c.T)[:, :3]
        draw_box(img, corners_cam, K, color)
        cv2.putText(img, f"{i:3d}  {label}", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if writer is None:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(str(out_mp4),
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     max(1.0, fps), (w, h))
        writer.write(img)
        n_drawn += 1
    if writer is not None:
        writer.release()
    print(f"wrote {out_mp4}  ({n_drawn} frames @ {fps:g} fps)")


def main():
    ap = argparse.ArgumentParser()
    DEF_INNER = "/weka/oe-training-default/weikaih/3d_boundingbox_detection/video_3d_box/itw_3dbox_det/open-vocab-3d-tracker"
    ap.add_argument("--inner_repo", default=DEF_INNER)
    ap.add_argument("--video_name", default="dog-example")
    ap.add_argument("--raw_meta", default=None,
                    help="step4-shaped meta.json with the raw Track C v11 boxes")
    ap.add_argument("--smoothed_meta", default=None,
                    help="step4-shaped meta.json with the Track A smoothed boxes")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--fps", type=float, default=None,
                    help="output fps; defaults to step1 meta's 'fps' (native).")
    args = ap.parse_args()

    inner = Path(args.inner_repo)
    step1 = inner / "output" / "step1" / args.video_name
    raw_meta = Path(args.raw_meta or inner / "output" / "step5_trackc" / args.video_name / "meta.json")
    sm_meta = Path(args.smoothed_meta or inner / "output" / "step5_trackc_kalman" / args.video_name / "meta.json")
    out_dir = Path(args.out_dir or inner / "output" / "step5_trackc_viz" / args.video_name)

    step1_meta = json.loads((step1 / "meta.json").read_text())
    fps = float(args.fps) if args.fps is not None else float(step1_meta.get("fps") or 29.97)
    print(f"native fps from step1 meta: {fps:g}")

    render(raw_meta, step1, out_dir / "viz_raw.mp4",
           color=(0, 0, 255),  # BGR -> red
           label="Track C v11 (raw, per-frame)", fps=fps)
    render(sm_meta, step1, out_dir / "viz_smoothed.mp4",
           color=(0, 255, 0),  # BGR -> green
           label="Track C v11 + Track A (smoothed)", fps=fps)


if __name__ == "__main__":
    main()
