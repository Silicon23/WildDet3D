"""Track C feature precompute.

One frozen-stack forward per (video, frame), prompting ALL tracked objects present
in that frame at once. depth_latents + ray_embeddings are deduped per (video,
frame); the per-object selected-query hidden states go to per-trajectory files.

Cache layout (under ``--cache_dir``):
    frames/<video>.pt   dict{step1_index: {depth_latents[2401,256] bf16,
                                            ray[2401,81] bf16, K[3,3] f32,
                                            input_hw}}
    traj/<video>__<object>.pt  dict{
        hidden[T,L,256] bf16, box2d[T,4] f32, sel_iou[T] f32,
        frame_index[T] int, ts_sec[T] f32, measured[T] bool,
        box_repr[T,12] f32, gt_center[T,3], gt_dims[T,3], gt_quat[T,4],
        category, video_id, object_id, input_hw }
    .done/<video>           marker (atomic, written last)

Run one shard:
    python -m wilddet3d.track_c.precompute --outputs <outputs> --ckpt <ckpt>
        --cache_dir <out> --shard <i> --num_shards 4
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

from wilddet3d.track_c.data import (
    _interp_box_repr,
    _load_clip_range,
    _quat_wxyz_from_R,
    _step4_camera_boxes,
)
from wilddet3d.track_c.feature_extractor import FrozenFeatureExtractor


def _atomic_save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def precompute_video(ext, outputs_dir, video_id, cache_dir,
                     ca1m_videos_dir=None, step1_dir=None, max_frames=None):
    ca1m_videos_dir = ca1m_videos_dir or f"{outputs_dir}/ca1m_extracted/videos"
    step1_dir = step1_dir or f"{outputs_dir}/step1/{video_id}"
    done_marker = f"{cache_dir}/.done/{video_id}"
    if os.path.exists(done_marker):
        return {"video": video_id, "skipped": True}

    clip = _load_clip_range(outputs_dir, video_id)
    if clip is None:
        return {"video": video_id, "error": "no clip"}
    start, end = int(clip["start_frame_idx"]), int(clip["end_frame_idx"])
    qual = clip.get("qualifying_object_ids", [])

    # objects with a Step-4 prior trajectory
    priors = {}
    for obj in qual:
        if os.path.isdir(f"{outputs_dir}/step4/{video_id}/{obj}"):
            try:
                p = _step4_camera_boxes(outputs_dir, video_id, obj)
            except Exception:
                p = {}
            if p:
                priors[obj] = p
    if not priors:
        # nothing trackable; still mark done so we don't retry
        _atomic_save({}, f"{cache_dir}/frames/{video_id}.pt")
        open(done_marker_dir(done_marker), "w").close()
        return {"video": video_id, "n_obj": 0}

    wdirs = sorted(glob.glob(f"{ca1m_videos_dir}/{video_id}/*.wide"),
                   key=lambda p: int(os.path.basename(p).split(".")[0]))
    if not wdirs:
        return {"video": video_id, "error": "no wide dirs"}
    ts_ns = [int(os.path.basename(p).split(".")[0]) for p in wdirs]
    n_frames = len(wdirs)
    K_all = np.load(f"{step1_dir}/intrinsics.npy")
    end = min(end, n_frames)

    frame_cache = {}                      # step1_index -> {depth_latents, ray, K, input_hw}
    per_obj = {obj: [] for obj in priors}  # obj -> list of frame records

    for idx in range(start, end):
        insts = json.load(open(f"{wdirs[idx]}/instances.json"))
        by_id = {o["id"]: o for o in insts}
        present = [obj for obj in priors if obj in by_id]
        if not present:
            continue
        prompts = [[float(x) for x in by_id[obj]["box_2d_proj"]] for obj in present]
        img = np.array(Image.open(f"{wdirs[idx]}/image.png").convert("RGB")).astype(np.float32)
        dpath = f"{step1_dir}/depth/{idx:06d}.npy"
        depth = np.load(dpath).astype(np.float32) if os.path.exists(dpath) else None
        feat = ext.extract(img, K_all[idx].astype(np.float32), prompts, depth=depth)

        frame_cache[idx] = {
            "depth_latents": feat["depth_latents"].to(torch.bfloat16),
            "ray": (feat["ray_embeddings"].to(torch.bfloat16)
                    if feat["ray_embeddings"] is not None else None),
            "K": feat["intrinsics"],                 # [3,3] model space
            "input_hw": feat["input_hw"],
        }
        for j, obj in enumerate(present):
            o = by_id[obj]
            per_obj[obj].append({
                "idx": idx, "ts_ns": ts_ns[idx],
                "hidden": feat["hidden_states"][:, j].to(torch.bfloat16),  # [L,256]
                "box2d": feat["pred_box_2d"][j],                           # [4]
                "sel_iou": float(feat["sel_iou"][j]),
                "gt_center": np.array(o["position"], np.float32),
                "gt_dims": np.array(o["scale"], np.float32),
                "gt_quat": _quat_wxyz_from_R(np.array(o["R"], np.float32)),
            })

    _atomic_save(frame_cache, f"{cache_dir}/frames/{video_id}.pt")
    input_hw = next(iter(frame_cache.values()))["input_hw"] if frame_cache else (1008, 1008)

    n_written = 0
    for obj, recs in per_obj.items():
        if len(recs) < 3:
            continue
        recs.sort(key=lambda r: r["idx"])
        idxs = np.array([r["idx"] for r in recs], np.float64)
        meas_idx = np.array(sorted(priors[obj].keys()), np.float64)
        meas_repr = np.stack([priors[obj][int(i)] for i in meas_idx], 0)
        filled = _interp_box_repr(idxs, meas_idx, meas_repr)
        meas_set = set(int(i) for i in meas_idx)
        box_repr = np.stack([
            priors[obj][r["idx"]] if r["idx"] in meas_set else filled[k]
            for k, r in enumerate(recs)
        ], 0).astype(np.float32)

        traj = {
            "hidden": torch.stack([r["hidden"] for r in recs], 0),          # [T,L,256]
            "box2d": torch.stack([r["box2d"] for r in recs], 0),            # [T,4]
            "sel_iou": torch.tensor([r["sel_iou"] for r in recs]),
            "frame_index": torch.tensor([r["idx"] for r in recs], dtype=torch.long),
            "ts_sec": torch.tensor(
                (np.array([r["ts_ns"] for r in recs], np.float64)
                 - recs[0]["ts_ns"]) / 1e9, dtype=torch.float32),
            "measured": torch.tensor([r["idx"] in meas_set for r in recs], dtype=torch.bool),
            "box_repr": torch.from_numpy(box_repr),
            "gt_center": torch.from_numpy(np.stack([r["gt_center"] for r in recs])),
            "gt_dims": torch.from_numpy(np.stack([r["gt_dims"] for r in recs])),
            "gt_quat": torch.from_numpy(np.stack([r["gt_quat"] for r in recs])),
            "category": "", "video_id": video_id, "object_id": obj,
            "input_hw": input_hw,
        }
        _atomic_save(traj, f"{cache_dir}/traj/{video_id}__{obj}.pt")
        n_written += 1

    os.makedirs(f"{cache_dir}/.done", exist_ok=True)
    open(done_marker, "w").close()
    return {"video": video_id, "n_obj": len(priors), "n_traj": n_written,
            "n_frames": len(frame_cache)}


def done_marker_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap #videos")
    ap.add_argument("--videos", default="", help="comma-sep video ids (overrides shard)")
    args = ap.parse_args()

    # video list = those with a clip entry AND step4 outputs
    clip_path = f"{args.outputs}/step2_5_clip_select/clip_index.jsonl"
    vids = []
    for line in open(clip_path):
        j = json.loads(line)
        v = str(j["video_id"])
        if os.path.isdir(f"{args.outputs}/step4/{v}"):
            vids.append(v)
    vids = sorted(set(vids))
    if args.videos:
        vids = [v for v in args.videos.split(",")]
    else:
        vids = [v for i, v in enumerate(vids) if i % args.num_shards == args.shard]
    if args.limit:
        vids = vids[:args.limit]
    print(f"[shard {args.shard}/{args.num_shards}] {len(vids)} videos", flush=True)

    ext = FrozenFeatureExtractor(checkpoint=args.ckpt, device=args.device, use_depth_input=True)
    t0 = time.time()
    for k, v in enumerate(vids):
        try:
            r = precompute_video(ext, args.outputs, v, args.cache_dir)
        except Exception as e:
            r = {"video": v, "error": repr(e)[:200]}
        dt = time.time() - t0
        print(f"[shard {args.shard}] {k+1}/{len(vids)} {r}  ({dt:.0f}s, {dt/(k+1):.1f}s/vid)", flush=True)
    print(f"[shard {args.shard}] DONE {len(vids)} videos in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
