"""End-to-end Track C v11 on the open-vocab-3d-tracker dog-example.

Per frame: decode the SAM 3 mask (Step 2) -> 2D bounding box -> use as
geometric prompt to the frozen WildDet3D stack -> v11 head -> 12-d camera-frame
3D box. Writes a Step-4-shaped meta.json that the Track A (Kalman/RTS) script
can consume directly.

Inputs:
  --inner_repo  open-vocab-3d-tracker root (defaults to the canonical path).
  --video_name  defaults to "dog-example".
  --ckpt        v11 checkpoint, defaults to outputs/track_c/runs/v11_no_traj_encoder/best.pt
  --out_dir     output directory (defaults to inner_repo/output/step5_trackc/<video>)

After running this, smooth + visualize with:
  python scripts/step5_kalman_smoother.py \\
      --step4_dir output/step5_trackc/dog-example \\
      --step1_dir output/step1/dog-example \\
      --output_dir output/step5_trackc_kalman \\
      --save_viz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as pycoco_mask

from vis4d.op.geometry.rotation import quaternion_to_matrix

from wilddet3d.track_c import TrackCRefiner
from wilddet3d.track_c.feature_extractor import FrozenFeatureExtractor


# Standard 8-corner box layout in box-local axes (extents are full, so half = e/2).
def box_corners_local(extents: np.ndarray) -> np.ndarray:
    """Return 8x3 corners in box-local frame; extents are full."""
    hx, hy, hz = extents / 2.0
    return np.array([
        [-hx, -hy, -hz], [+hx, -hy, -hz], [+hx, +hy, -hz], [-hx, +hy, -hz],
        [-hx, -hy, +hz], [+hx, -hy, +hz], [+hx, +hy, +hz], [-hx, +hy, +hz],
    ], dtype=np.float64)


def mask_bbox_xyxy(rle_dict: dict, h: int, w: int) -> list[float] | None:
    """Decode a COCO RLE dict and return tight xyxy bbox in pixel coords."""
    rle = rle_dict.copy()
    if isinstance(rle.get("counts"), str):
        rle["counts"] = rle["counts"].encode("utf-8")
    m = pycoco_mask.decode(rle)
    if m.ndim == 3:
        m = m[..., 0]
    ys, xs = np.where(m > 0)
    if xs.size == 0:
        return None
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    # tiny dilation guards against the mask kissing the object edge
    x1, y1 = max(0.0, x1 - 1), max(0.0, y1 - 1)
    x2, y2 = min(w - 1.0, x2 + 1), min(h - 1.0, y2 + 1)
    return [x1, y1, x2, y2]


def main():
    ap = argparse.ArgumentParser()
    DEF_INNER = "/weka/oe-training-default/weikaih/3d_boundingbox_detection/video_3d_box/itw_3dbox_det/open-vocab-3d-tracker"
    DEF_CKPT = "/weka/oe-training-default/weikaih/3d_boundingbox_detection/video_3d_box/itw_3dbox_det/outputs/track_c/runs/v11_no_traj_encoder/best.pt"
    DEF_WD = "/weka/oe-training-default/weikaih/3d_boundingbox_detection/video_3d_box/itw_3dbox_det/WildDet3D"
    ap.add_argument("--inner_repo", default=DEF_INNER)
    ap.add_argument("--video_name", default="dog-example")
    ap.add_argument("--ckpt", default=DEF_CKPT)
    ap.add_argument("--wilddet3d_ckpt", default=f"{DEF_WD}/ckpt/wilddet3d_alldata_all_prompt_v1.0.pt",
                    help="WildDet3D pretrained checkpoint to seed the frozen stack")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap #frames")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    inner = Path(args.inner_repo)
    step1 = inner / "output" / "step1" / args.video_name
    step2 = inner / "output" / "step2" / args.video_name
    out_dir = Path(args.out_dir or inner / "output" / "step5_trackc" / args.video_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[dog] inner={inner}\n      step1={step1}\n      step2={step2}\n      out={out_dir}", flush=True)

    # ---- load Step 1 ----
    K_all = np.load(step1 / "intrinsics.npy").astype(np.float32)   # [N,3,3]
    E_all = np.load(step1 / "extrinsics.npy").astype(np.float32)   # [N,4,4] world->cam
    c2w_all = np.load(step1 / "poses_c2w.npy").astype(np.float32)  # [N,4,4] cam->world
    step1_meta = json.loads((step1 / "meta.json").read_text())
    n_frames = int(step1_meta["num_frames"])
    H, W = int(step1_meta["image_height"]), int(step1_meta["image_width"])
    src_idx = step1_meta.get("source_frame_indices") or list(range(n_frames))
    print(f"[dog] step1: {n_frames} frames @ {W}x{H}; intrinsics constant? "
          f"{bool((K_all - K_all[0]).std() < 1e-3)}", flush=True)

    # ---- load Step 2 masks ----
    step2_masks = json.loads((step2 / "masks_rle.json").read_text())
    step2_meta = json.loads((step2 / "meta.json").read_text())
    cat = step2_meta.get("text") or "object"
    print(f"[dog] step2: {len(step2_masks)} mask entries; prompt text='{cat}'", flush=True)

    # ---- build frozen feature extractor + v11 refiner ----
    ext = FrozenFeatureExtractor(checkpoint=args.wilddet3d_ckpt, device=args.device,
                                 use_depth_input=True)
    refiner = TrackCRefiner(
        reg_residual_from_prior=False,
        use_temporal_modules=False,   # v11
        use_layer_bias=False,
    ).to(args.device)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["refiner"]
    refiner.load_state_dict(sd, strict=True)
    refiner.eval()
    print(f"[dog] models ready (v11 head, modules removed, 21.07M params)", flush=True)

    # ---- per-frame inference ----
    frames_dir = step1 / "frames"
    depth_dir = step1 / "depth"
    n_run = min(args.limit, n_frames) if args.limit else n_frames

    out_frames = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(n_run):
            orig = int(src_idx[i])
            mkey = str(i)
            rle = step2_masks.get(mkey)
            if rle is None or rle.get("mask_rle") is None:
                continue
            bbox = mask_bbox_xyxy(rle["mask_rle"], H, W)
            if bbox is None:
                continue

            img_path = frames_dir / f"{i:06d}.jpg"
            depth_path = depth_dir / f"{i:06d}.npy"
            img = np.array(Image.open(img_path).convert("RGB")).astype(np.float32)
            dep = np.load(depth_path).astype(np.float32) if depth_path.exists() else None
            K_i = K_all[i]

            feat = ext.extract(img, K_i, [bbox], depth=dep)
            hidden = feat["hidden_states"].unsqueeze(1).to(args.device)        # [L,1,1,256]
            ray = feat["ray_embeddings"].unsqueeze(0).float().to(args.device)  # [1,N_tok,81]
            depth_l = feat["depth_latents"].unsqueeze(0).float().to(args.device)  # [1,N_tok,256]
            box2d = feat["pred_box_2d"].to(args.device)                        # [1,4] norm xyxy
            K_t = feat["intrinsics"].to(args.device)                           # [3,3] model space
            ihw = feat["input_hw"]

            # v11 forward (no traj_encoder, no temporal_box)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = refiner(
                    hidden_states=hidden.float(),
                    ray_embeddings=ray, depth_latents=depth_l,
                    pred_box_2d=box2d, intrinsics=K_t,
                    box_repr=torch.zeros(1, 12, device=args.device),  # unused (no modules)
                    timestamps=torch.zeros(1, device=args.device),
                    measured_mask=torch.zeros(1, dtype=torch.bool, device=args.device),
                    input_hw=ihw,
                )
            dec = refiner.decode_layer(out["reg"][-1, :, 0, :].float(),
                                       box2d, K_t, ihw).cpu().numpy()[0]
            center_cam = dec[0:3].astype(np.float64)
            dims = np.maximum(dec[3:6], 1e-3).astype(np.float64)
            quat = dec[6:10]
            R_cam = quaternion_to_matrix(torch.from_numpy(quat).float()).numpy().astype(np.float64)

            # camera->world via Step-1 c2w at this frame
            c2w = c2w_all[i].astype(np.float64)
            R_w2c = c2w[:3, :3].T               # world->camera rotation (= extrinsics rotation)
            R_world = c2w[:3, :3] @ R_cam
            center_world = c2w[:3, :3] @ center_cam + c2w[:3, 3]

            corners_local = box_corners_local(dims)
            corners_cam = (R_cam @ corners_local.T).T + center_cam
            corners_world = (R_world @ corners_local.T).T + center_world

            out_frames.append({
                "source_frame_index": orig,
                "step1_index": i,
                "step2_index": i,
                "status": "ok",
                "sam3_score": float(rle.get("sam3_score") or 1.0),
                "proj_box_mask_iou": float(feat["sel_iou"][0]),
                "intrinsics": K_i.tolist(),
                "camera_to_world": c2w_all[i].astype(np.float64).tolist(),
                "T_obj_in_cam": np.r_[np.c_[R_cam, center_cam[:, None]],
                                      [[0, 0, 0, 1]]].tolist(),
                "T_obj_in_world": np.r_[np.c_[R_world, center_world[:, None]],
                                        [[0, 0, 0, 1]]].tolist(),
                "box": {
                    "center_cam": center_cam.tolist(),
                    "center_world": center_world.tolist(),
                    "R_cam": R_cam.tolist(),
                    "R_world": R_world.tolist(),
                    "obb_extents": dims.tolist(),
                    "size": dims.tolist(),
                    "corners_cam": corners_cam.tolist(),
                    "corners_world": corners_world.tolist(),
                },
            })
            if (i + 1) % 20 == 0:
                el = time.time() - t0
                print(f"[dog] {i+1}/{n_run} frames, {el:.1f}s elapsed "
                      f"(IoU sel {float(feat['sel_iou'][0]):.3f})", flush=True)

    print(f"[dog] {len(out_frames)} per-frame boxes in {time.time()-t0:.1f}s", flush=True)

    # ---- write Step-4-shaped meta.json ----
    # Track A reads target_fps + frame_stride to set output_fps for its viz.
    # We process every frame (frame_stride = 1), so output_fps = native video fps.
    native_fps = float(step1_meta.get("fps") or 29.97)
    meta = {
        "step": "step5_trackc_v11",
        "generated_by": "wilddet3d.track_c.run_dog_example",
        "video_name": args.video_name,
        "model": "Track C v11 (WildDet3D head, no temporal modules)",
        "source_ckpt": args.ckpt,
        "num_frames": len(out_frames),
        "camera_frame": "OpenCV (x right, y down, z forward)",
        "target_fps": native_fps,
        "frame_stride": 1,
        "frames": out_frames,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[dog] wrote {out_dir/'meta.json'}", flush=True)
    print(f"\nNext step: render mp4 via Track A's smoother (which also smooths):")
    print(f"  python {inner}/scripts/step5_kalman_smoother.py \\")
    print(f"    --step4_dir {out_dir} \\")
    print(f"    --step1_dir {step1} \\")
    print(f"    --output_dir {inner}/output/step5_trackc_kalman \\")
    print(f"    --save_viz")


if __name__ == "__main__":
    main()
