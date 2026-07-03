"""Track C feature precompute for WAYMO (dynamic-object diagnostic).

Same frozen-feature cache format as precompute.py (CA-1M) so dataset.py reads it
unchanged, but adapted to the Waymo production contract (dataset-scout 2026-06-14):

  RGB        : step1_waymo/<SEG>/frames/<idx:06d>.jpg
  depth(ViPE): step1_waymo/<SEG>/depth/<idx:06d>.npy   (pipeline branch; ~1.5x short
               systematic bias = the learnable depth correction)
  intrinsics : step1_waymo/<SEG>/intrinsics.npy        ([N,3,3] or [3,3])
  geo-prompt : step2_waymo/<SEG>/<OBJ>/masks_rle.json  (per-frame mask -> tight bbox;
               honest deployment path, same as the dog demo)
  noisy prior: step4_waymo/<SEG>/<OBJ>/meta.json  f['box'] (skip None = FP fail)
  GT target  : waymo_production/<SEG>/gt/boxes.json  frames[].objects[] matched by
               id==OBJ(track_id) and frame_index==step1_index. dims_lhw = full extents.

Pairing per scout: (SEG, OBJ=track_id, step1_index==GT frame_index), native 10 Hz,
no Step 4.5 needed (manifest geo-pinned SAM3 to GT tracks => exact correspondence).

Cache layout (under --cache_dir):
  frames/<SEG>.pt   {step1_index: {depth_latents,ray,K,input_hw}}
  traj/<SEG>__<OBJ>.pt  {hidden[T,L,256], box2d[T,4], sel_iou[T], frame_index[T],
                         ts_sec[T], measured[T], box_repr[T,12], gt_center[T,3],
                         gt_dims[T,3], gt_quat[T,4], category, video_id, object_id,
                         input_hw}
  .done/<SEG>

Run one shard (vehicles+cyclists by default):
  python -m wilddet3d.track_c.waymo_precompute --outputs <outputs> --ckpt <ckpt>
    --cache_dir <out> --shard <i> --num_shards 4 [--categories vehicle,cyclist]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as pycoco_mask

from wilddet3d.track_c.data import _camera_box_repr, _interp_box_repr, _quat_wxyz_from_R
from wilddet3d.track_c.feature_extractor import FrozenFeatureExtractor
from wilddet3d.track_c.precompute import _atomic_save


def _lidar_depth_map(outputs_dir, seg, idx, H, W):
    """Sparse GT-LiDAR depth map in METERS, matching WildDet3D's training format
    (generate_waymo_depth_maps.py): zeros(H,W) with first-return points splatted,
    0=invalid, keep-closer on pixel collisions. NO densification/interpolation.
    Returns None if the npz is missing."""
    p = f"{outputs_dir}/waymo_production/{seg}/gt/depth/{idx:06d}.npz"
    if not os.path.exists(p):
        return None
    z = np.load(p)
    u, v, d = z["u"].astype(np.int64), z["v"].astype(np.int64), z["z"].astype(np.float32)
    m = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (d > 0)
    u, v, d = u[m], v[m], d[m]
    dm = np.zeros((H, W), dtype=np.float32)
    # keep-closer: sort by descending depth so nearer points overwrite last
    order = np.argsort(-d)
    dm[v[order], u[order]] = d[order]
    return dm


def _mask_bbox_xyxy(rle_dict, h, w):
    """COCO RLE -> tight xyxy pixel bbox (+1px dilation), or None if empty."""
    rle = dict(rle_dict)
    if isinstance(rle.get("counts"), str):
        rle["counts"] = rle["counts"].encode("utf-8")
    m = pycoco_mask.decode(rle)
    if m.ndim == 3:
        m = m[..., 0]
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return None
    x0, x1 = xs.min() - 1, xs.max() + 1
    y0, y1 = ys.min() - 1, ys.max() + 1
    return [float(max(0, x0)), float(max(0, y0)), float(min(w - 1, x1)), float(min(h - 1, y1))]


def _gt_box2d_from_obj(gt_obj, K, H, W):
    """Project the 8 GT 3D corners through K -> tight xyxy 2D box (+2px pad).

    Uses gt_obj['corners_cam'] if present; otherwise reconstructs from
    (center_cam, dims_lhw, R_cam). Corners with z <= 0.1m are dropped (behind
    camera). Returns None if < 4 corners project in front.

    IMPORTANT: K here must be the SAME intrinsics the model will see (GT-K
    when intrinsics_source='gt'), so the geo prompt lands in the same pixel
    frame as the image the frozen encoder consumes.
    """
    corners = None
    if "corners_cam" in gt_obj:
        arr = np.asarray(gt_obj["corners_cam"], dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] == 8 and arr.shape[1] == 3:
            corners = arr
    if corners is None:
        c = np.asarray(gt_obj["center_cam"], np.float32)
        d = np.asarray(gt_obj.get("dims_lhw") or gt_obj["dims_xyz_obj"], np.float32)
        R = np.asarray(gt_obj["R_cam"], np.float32)
        hx, hy, hz = d / 2
        local = np.array([[-hx,-hy,-hz],[+hx,-hy,-hz],[+hx,+hy,-hz],[-hx,+hy,-hz],
                          [-hx,-hy,+hz],[+hx,-hy,+hz],[+hx,+hy,+hz],[-hx,+hy,+hz]],
                         dtype=np.float32)
        corners = (R @ local.T).T + c
    xs, ys = [], []
    for x, y, z in corners:
        if z > 0.1:
            xs.append(float(K[0, 0] * x / z + K[0, 2]))
            ys.append(float(K[1, 1] * y / z + K[1, 2]))
    if len(xs) < 4:
        return None
    x0 = max(0.0, min(xs) - 2)
    y0 = max(0.0, min(ys) - 2)
    x1 = min(float(W - 1), max(xs) + 2)
    y1 = min(float(H - 1), max(ys) + 2)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return [x0, y0, x1, y1]


def _load_gt(outputs_dir, seg):
    """boxes.json -> ({frame_index: {obj_id: obj}}, {obj_id: category},
    {frame_index: ts_micros})."""
    g = json.load(open(f"{outputs_dir}/waymo_production/{seg}/gt/boxes.json"))
    by_frame, cat, ts = {}, {}, {}
    for fr in g["frames"]:
        fi = int(fr["frame_index"])
        ts[fi] = int(fr.get("timestamp_micros", fi * 100000))
        d = {}
        for o in fr["objects"]:
            d[o["id"]] = o
            cat.setdefault(o["id"], o["category"])
        by_frame[fi] = d
    return by_frame, cat, ts


def _noisy_boxes(outputs_dir, seg, obj, step4_subdir="step4_waymo"):
    """step4 meta -> {step1_index: box_repr[12]} for frames with a valid FP box.
    step4_subdir picks the trajectory-prior source: 'step4_waymo' (ViPE-K pipeline,
    default) or 'step4_waymo_gt' (regenerated GT-K+GT-LiDAR pipeline, task #41)."""
    m = json.load(open(f"{outputs_dir}/{step4_subdir}/{seg}/{obj}/meta.json"))
    out = {}
    for f in m["frames"]:
        b = f.get("box")
        if b is None:
            continue
        idx = int(f["step1_index"])
        ext = b.get("obb_extents") or b.get("size")
        out[idx] = _camera_box_repr(np.array(b["center_cam"], np.float32),
                                    np.array(ext, np.float32),
                                    np.array(b["R_cam"], np.float32))
    return out


def _load_pairing_index(outputs_dir, path):
    """pairing_index.jsonl -> {seg: {obj: {category, frames:set(step1_index)}}}."""
    by_seg = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        frames = set(int(f["step1_index"]) for f in d["frames"])
        by_seg.setdefault(d["seg"], {})[d["track_id"]] = {
            "category": d["category"], "frames": frames}
    return by_seg


def precompute_segment(ext, outputs_dir, seg, cache_dir, categories, index_objs=None,
                       depth_source="vipe", intrinsics_source="vipe",
                       geo_prompt_source="mask", step4_subdir="step4_waymo"):
    done_marker = f"{cache_dir}/.done/{seg}"
    if os.path.exists(done_marker):
        return {"seg": seg, "skipped": True}

    step1 = f"{outputs_dir}/step1_waymo/{seg}"
    frames_dir = f"{step1}/frames"
    depth_dir = f"{step1}/depth"
    if not os.path.isdir(frames_dir):
        return {"seg": seg, "error": "no step1 frames"}
    gt_by_frame, gt_cat, gt_ts = _load_gt(outputs_dir, seg)
    # ViPE (step1) intrinsics are per-video unreliable on Waymo (focal off up to ~5x);
    # GT intrinsics (waymo_production/<seg>/gt) are the true Waymo calibration.
    if intrinsics_source == "gt":
        K_all = np.load(f"{outputs_dir}/waymo_production/{seg}/gt/intrinsics.npy")
    else:
        K_all = np.load(f"{step1}/intrinsics.npy")
    per_frame_K = K_all.ndim == 3

    # which tracked objects to process: canonical pairing-index list if provided
    # (skip-frame logic already baked in by dataset-scout), else walk the tree.
    if index_objs is not None:
        obj_names = sorted(index_objs.keys())
    else:
        obj_names = [o for o in sorted(os.listdir(f"{outputs_dir}/{step4_subdir}/{seg}"))
                     if gt_cat.get(o) in categories]
    objs = {}
    for obj in obj_names:
        try:
            nb = _noisy_boxes(outputs_dir, seg, obj, step4_subdir=step4_subdir)
        except Exception:
            nb = {}
        if not nb:
            continue
        # In gt-2D mode we don't need SAM3 masks — identity comes from projecting
        # the GT 3D box for the correct object. In mask mode we do (and skip the
        # track if the mask file is absent, since that means Step-2 dropped it).
        if geo_prompt_source == "mask":
            mpath = f"{outputs_dir}/step2_waymo/{seg}/{obj}/masks_rle.json"
            if not os.path.exists(mpath):
                continue
            masks = json.load(open(mpath))
        else:
            masks = None
        objs[obj] = {"noisy": nb, "masks": masks,
                     "frames": index_objs[obj]["frames"] if index_objs else None}
    if not objs:
        os.makedirs(f"{cache_dir}/.done", exist_ok=True)
        open(done_marker, "w").close()
        return {"seg": seg, "n_obj": 0}

    frame_cache = {}
    per_obj = {obj: [] for obj in objs}

    # union of trainable frames across tracked objs: need mask + GT for that obj
    all_idx = sorted(gt_by_frame.keys())
    for idx in all_idx:
        present = []
        for obj, d in objs.items():
            if d["frames"] is not None and idx not in d["frames"]:
                continue  # restrict to canonical paired frames
            if obj not in gt_by_frame[idx]:
                continue
            if geo_prompt_source == "mask":
                mk = d["masks"].get(str(idx))
                if mk is None or mk.get("mask_rle") is None:
                    continue
                present.append((obj, mk["mask_rle"]))
            else:  # gt-2D box from GT 3D corners projected through K
                present.append((obj, None))
        if not present:
            continue
        img = np.array(Image.open(f"{frames_dir}/{idx:06d}.jpg").convert("RGB")).astype(np.float32)
        H, W = img.shape[:2]
        K = (K_all[idx] if per_frame_K else K_all).astype(np.float32)
        prompts, kept = [], []
        for obj, mk in present:
            if geo_prompt_source == "mask":
                bb = _mask_bbox_xyxy(mk, H, W)
            else:
                bb = _gt_box2d_from_obj(gt_by_frame[idx][obj], K, H, W)
            if bb is not None:
                prompts.append(bb); kept.append(obj)
        if not prompts:
            continue
        if depth_source == "lidar":
            depth = _lidar_depth_map(outputs_dir, seg, idx, H, W)
        else:
            dpath = f"{depth_dir}/{idx:06d}.npy"
            depth = np.load(dpath).astype(np.float32) if os.path.exists(dpath) else None
        feat = ext.extract(img, K, prompts, depth=depth)

        frame_cache[idx] = {
            "depth_latents": feat["depth_latents"].to(torch.bfloat16),
            "ray": (feat["ray_embeddings"].to(torch.bfloat16)
                    if feat["ray_embeddings"] is not None else None),
            "K": feat["intrinsics"], "input_hw": feat["input_hw"],
        }
        for j, obj in enumerate(kept):
            o = gt_by_frame[idx][obj]
            per_obj[obj].append({
                "idx": idx, "ts": gt_ts[idx],
                "hidden": feat["hidden_states"][:, j].to(torch.bfloat16),
                "box2d": feat["pred_box_2d"][j], "sel_iou": float(feat["sel_iou"][j]),
                "gt_center": np.array(o["center_cam"], np.float32),
                "gt_dims": np.array(o["dims_lhw"], np.float32),
                "gt_quat": _quat_wxyz_from_R(np.array(o["R_cam"], np.float32)),
            })

    _atomic_save(frame_cache, f"{cache_dir}/frames/{seg}.pt")
    input_hw = next(iter(frame_cache.values()))["input_hw"] if frame_cache else (1008, 1008)

    n_written = 0
    for obj, recs in per_obj.items():
        if len(recs) < 3:
            continue
        recs.sort(key=lambda r: r["idx"])
        idxs = np.array([r["idx"] for r in recs], np.float64)
        meas_idx = np.array(sorted(objs[obj]["noisy"].keys()), np.float64)
        meas_repr = np.stack([objs[obj]["noisy"][int(i)] for i in meas_idx], 0)
        filled = _interp_box_repr(idxs, meas_idx, meas_repr)
        meas_set = set(int(i) for i in meas_idx)
        box_repr = np.stack([
            objs[obj]["noisy"][r["idx"]] if r["idx"] in meas_set else filled[k]
            for k, r in enumerate(recs)], 0).astype(np.float32)
        t0 = recs[0]["ts"]
        traj = {
            "hidden": torch.stack([r["hidden"] for r in recs], 0),
            "box2d": torch.stack([r["box2d"] for r in recs], 0),
            "sel_iou": torch.tensor([r["sel_iou"] for r in recs]),
            "frame_index": torch.tensor([r["idx"] for r in recs], dtype=torch.long),
            "ts_sec": torch.tensor([(r["ts"] - t0) / 1e6 for r in recs], dtype=torch.float32),
            "measured": torch.tensor([r["idx"] in meas_set for r in recs], dtype=torch.bool),
            "box_repr": torch.from_numpy(box_repr),
            "gt_center": torch.from_numpy(np.stack([r["gt_center"] for r in recs])),
            "gt_dims": torch.from_numpy(np.stack([r["gt_dims"] for r in recs])),
            "gt_quat": torch.from_numpy(np.stack([r["gt_quat"] for r in recs])),
            "category": gt_cat.get(obj, ""), "video_id": seg, "object_id": obj,
            "input_hw": input_hw,
        }
        _atomic_save(traj, f"{cache_dir}/traj/{seg}__{obj}.pt")
        n_written += 1

    os.makedirs(f"{cache_dir}/.done", exist_ok=True)
    open(done_marker, "w").close()
    return {"seg": seg, "n_obj": len(objs), "n_traj": n_written, "n_frames": len(frame_cache)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--categories", default="vehicle,cyclist")
    ap.add_argument("--pairing_index", default="",
                    help="pairing_index.jsonl: drive canonical track/frame set from it "
                         "(skip-frame logic baked in) instead of walking the tree")
    ap.add_argument("--depth_source", default="vipe", choices=["vipe", "lidar"],
                    help="vipe = step1_waymo ViPE depth (default); lidar = sparse GT-LiDAR "
                         "splatted to a meters map (WildDet3D's training LiDAR format, no densify)")
    ap.add_argument("--intrinsics_source", default="vipe", choices=["vipe", "gt"],
                    help="vipe = step1_waymo ViPE intrinsics (default, per-video unreliable: "
                         "focal off up to ~5x); gt = waymo_production/<seg>/gt true calibration")
    ap.add_argument("--geo_prompt_source", default="mask", choices=["mask", "gt"],
                    help="mask = SAM3 Step-2 mask -> tight xyxy (shipped pipeline, but SAM3 "
                         "identity drifts on far/small objects); gt = project 8 GT 3D corners "
                         "through GT-K -> tight xyxy (use when eval-only or benchmarking; "
                         "requires intrinsics_source='gt' for geometric consistency)")
    ap.add_argument("--step4_subdir", default="step4_waymo",
                    help="trajectory-prior source under outputs/: step4_waymo (ViPE-K pipeline, "
                         "default; median center err 5-13m on val) OR step4_waymo_gt (regenerated "
                         "GT-K + GT-LiDAR pipeline, task #41; median <1m on smoke pair 722)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cats = set(c.strip() for c in args.categories.split(",") if c.strip())

    index = _load_pairing_index(args.outputs, args.pairing_index) if args.pairing_index else None
    if index is not None:
        # only segments with wanted-category tracks in the index
        segs_all = sorted(s for s, objs in index.items()
                          if any(o["category"] in cats for o in objs.values()))
    else:
        segs_all = sorted(os.listdir(f"{args.outputs}/{args.step4_subdir}"))
    segs = [s for i, s in enumerate(segs_all) if i % args.num_shards == args.shard]
    if args.limit:
        segs = segs[:args.limit]
    print(f"[shard {args.shard}/{args.num_shards}] {len(segs)} segs, cats={cats}, "
          f"index={'yes' if index else 'no'}", flush=True)

    ext = FrozenFeatureExtractor(checkpoint=args.ckpt, device=args.device, use_depth_input=True)
    t0 = time.time()
    for k, s in enumerate(segs):
        try:
            iobjs = None
            if index is not None:
                iobjs = {o: v for o, v in index[s].items() if v["category"] in cats}
            r = precompute_segment(ext, args.outputs, s, args.cache_dir, cats,
                                   index_objs=iobjs, depth_source=args.depth_source,
                                   intrinsics_source=args.intrinsics_source,
                                   geo_prompt_source=args.geo_prompt_source,
                                   step4_subdir=args.step4_subdir)
        except Exception as e:
            r = {"seg": s, "error": repr(e)[:200]}
        dt = time.time() - t0
        print(f"[shard {args.shard}] {k+1}/{len(segs)} {r} ({dt:.0f}s)", flush=True)
    print(f"[shard {args.shard}] DONE {len(segs)} segs in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
