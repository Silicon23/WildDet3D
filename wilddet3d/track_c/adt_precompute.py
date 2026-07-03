"""Track C feature precompute for ADT (Aria Digital Twin) — same cache format
as waymo_precompute.py / precompute.py so dataset.py reads it unchanged.

ADT data contract (dataset-scout 2026-06-28):
  RGB        : adt_production/<SEG>/frames/<idx:06d>.jpg                (704x704)
  depth (GT) : adt_production/<SEG>/gt/depth/<idx:06d>.npz key 'z_dense'
               (DENSE float32 metres — no splatting needed, unlike Waymo's sparse
               LiDAR; GT depth is the only available depth on ADT)
  intrinsics : adt_production/<SEG>/gt/intrinsics.npy   ([N,3,3]; GT/canonical)
  geo-prompt : step2_adt_production/<SEG>/<OBJ>/masks_rle.json
  noisy prior: step4_adt_production/<SEG>/<OBJ>/meta.json  f['box']  (skip None)
  GT target  : adt_production/<SEG>/gt/boxes.json  frames[].objects[]
               (uses dims_lhw alias added by scout = object-axis AABB extents)
  pairing    : adt_production/pairing_index_adt.jsonl  (5,524 tracks, jasonr paths
               from start; frame-list encodes noisy+GT skip-logic already)

Run one shard:
  python -m wilddet3d.track_c.adt_precompute --outputs <outputs> --ckpt <ckpt>
    --cache_dir <out> --pairing_index <jsonl> --shard <i> --num_shards <N>
"""
from __future__ import annotations

import argparse
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
from wilddet3d.track_c.waymo_precompute import _gt_box2d_from_obj


def _dense_depth_map(outputs_dir, seg, idx, H, W):
    """ADT dense GT depth in metres. npz key 'z_dense', shape (H,W) float32,
    0=invalid (e.g. pixels outside the digital-twin coverage). Returns None if
    the npz is missing OR shape doesn't match the RGB frame (defensive)."""
    p = f"{outputs_dir}/adt_production/{seg}/gt/depth/{idx:06d}.npz"
    if not os.path.exists(p):
        return None
    z = np.load(p)
    if "z_dense" not in z.files:
        return None
    d = z["z_dense"].astype(np.float32)
    if d.shape != (H, W):
        return None
    return d


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


def _load_gt(outputs_dir, seg):
    """ADT gt/boxes.json -> ({frame_index: {obj_id: obj}}, {obj_id: category},
    {frame_index: ts_ns}). Uses dims_lhw (added by scout) as the canonical dims."""
    g = json.load(open(f"{outputs_dir}/adt_production/{seg}/gt/boxes.json"))
    by_frame, cat, ts = {}, {}, {}
    for fr in g["frames"]:
        fi = int(fr["frame_index"])
        ts[fi] = int(fr.get("timestamp_ns", fi * 33333333))  # ~30 Hz fallback
        d = {}
        for o in fr["objects"]:
            d[str(o["id"])] = o
            cat.setdefault(str(o["id"]), o["category"])
        by_frame[fi] = d
    return by_frame, cat, ts


def _noisy_boxes(outputs_dir, seg, obj, step4_subdir="step4_adt_production"):
    """ADT step4 meta -> {step1_index: box_repr[12]} for frames with FP box.
    step4_subdir: 'step4_adt_production' (ViPE pipeline, default) OR
    'step4_adt_production_gt' (regen GT-K+GT-dense pipeline, task #41)."""
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
    """pairing_index_adt.jsonl -> {seg: {obj: {category, frames:set(step1_index)}}}.
    Frame-list is restricted to entries where BOTH noisy_meta_path_present AND
    gt_present are True (the scout's skip-logic, baked in)."""
    by_seg = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        frames = set(int(f["step1_index"]) for f in d["frames"]
                     if f.get("noisy_meta_path_present") and f.get("gt_present"))
        by_seg.setdefault(d["seg"], {})[str(d["track_id"])] = {
            "category": d["category"], "frames": frames}
    return by_seg


def precompute_segment(ext, outputs_dir, seg, cache_dir, categories, index_objs,
                       geo_prompt_source="gt", step4_subdir="step4_adt_production"):
    done_marker = f"{cache_dir}/.done/{seg}"
    if os.path.exists(done_marker):
        return {"seg": seg, "skipped": True}

    frames_dir = f"{outputs_dir}/adt_production/{seg}/frames"
    if not os.path.isdir(frames_dir):
        return {"seg": seg, "error": "no adt frames"}
    gt_by_frame, gt_cat, gt_ts = _load_gt(outputs_dir, seg)
    # GT intrinsics from ADT digital-twin (always canonical; no ViPE on ADT)
    K_all = np.load(f"{outputs_dir}/adt_production/{seg}/gt/intrinsics.npy")
    per_frame_K = K_all.ndim == 3

    obj_names = sorted(index_objs.keys())
    objs = {}
    for obj in obj_names:
        try:
            nb = _noisy_boxes(outputs_dir, seg, obj, step4_subdir=step4_subdir)
        except Exception:
            nb = {}
        if not nb:
            continue
        if geo_prompt_source == "mask":
            mpath = f"{outputs_dir}/step2_adt_production/{seg}/{obj}/masks_rle.json"
            if not os.path.exists(mpath):
                continue
            masks = json.load(open(mpath))
        else:  # gt: identity comes from GT 3D box projection; SAM3 mask not needed
            masks = None
        objs[obj] = {"noisy": nb, "masks": masks,
                     "frames": index_objs[obj]["frames"]}
    if not objs:
        os.makedirs(f"{cache_dir}/.done", exist_ok=True)
        open(done_marker, "w").close()
        return {"seg": seg, "n_obj": 0}

    frame_cache = {}
    per_obj = {obj: [] for obj in objs}

    all_idx = sorted(gt_by_frame.keys())
    for idx in all_idx:
        present = []
        for obj, d in objs.items():
            if idx not in d["frames"]:
                continue  # restrict to paired frames per index
            if obj not in gt_by_frame[idx]:
                continue
            if geo_prompt_source == "mask":
                mk = d["masks"].get(str(idx))
                if mk is None or mk.get("mask_rle") is None:
                    continue
                present.append((obj, mk["mask_rle"]))
            else:
                present.append((obj, None))
        if not present:
            continue
        img_path = f"{frames_dir}/{idx:06d}.jpg"
        if not os.path.exists(img_path):
            continue
        img = np.array(Image.open(img_path).convert("RGB")).astype(np.float32)
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
        depth = _dense_depth_map(outputs_dir, seg, idx, H, W)
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
            "ts_sec": torch.tensor([(r["ts"] - t0) / 1e9 for r in recs], dtype=torch.float32),
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
    ap.add_argument("--pairing_index", required=True,
                    help="adt_production/pairing_index_adt.jsonl (required for ADT)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--categories", default="",
                    help="comma list to restrict by GT category; empty = all ADT cats")
    ap.add_argument("--geo_prompt_source", default="gt", choices=["mask", "gt"],
                    help="gt = project 8 GT 3D corners through GT-K -> tight xyxy (default, "
                         "since ADT has canonical GT for everything); mask = SAM3 Step-2 mask "
                         "(only for shipped-pipeline simulation)")
    ap.add_argument("--step4_subdir", default="step4_adt_production",
                    help="trajectory-prior source: step4_adt_production (ViPE pipeline, "
                         "default) OR step4_adt_production_gt (regen GT-K+GT-dense pipeline, "
                         "task #41; 17mm median center err on smoke pair 778 vs 231mm ViPE)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cats = set(c.strip() for c in args.categories.split(",") if c.strip()) if args.categories else None

    index = _load_pairing_index(args.outputs, args.pairing_index)
    if cats is not None:
        # filter to segments with any wanted-category track
        segs_all = sorted(s for s, objs in index.items()
                          if any(o["category"] in cats for o in objs.values()))
    else:
        segs_all = sorted(index.keys())
    segs = [s for i, s in enumerate(segs_all) if i % args.num_shards == args.shard]
    if args.limit:
        segs = segs[:args.limit]
    print(f"[shard {args.shard}/{args.num_shards}] {len(segs)} segs, "
          f"cats={cats if cats else 'ALL'}", flush=True)

    ext = FrozenFeatureExtractor(checkpoint=args.ckpt, device=args.device, use_depth_input=True)
    t0 = time.time()
    for k, s in enumerate(segs):
        try:
            iobjs = {o: v for o, v in index[s].items()
                     if (cats is None or v["category"] in cats)}
            r = precompute_segment(ext, args.outputs, s, args.cache_dir, cats or set(),
                                   iobjs, geo_prompt_source=args.geo_prompt_source,
                                   step4_subdir=args.step4_subdir)
        except Exception as e:
            r = {"seg": s, "error": repr(e)[:200]}
        dt = time.time() - t0
        print(f"[shard {args.shard}] {k+1}/{len(segs)} {r} ({dt:.0f}s)", flush=True)
    print(f"[shard {args.shard}] DONE {len(segs)} segs in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
