"""Cached Track C dataset.

Reads the precompute cache (see ``precompute.py``): per-trajectory hidden states
+ metadata, and a per-video frames file with deduped depth_latents + ray
embeddings. Each item is one trajectory ("pack") ready for ``TrackCRefiner``.

Split is **by video** (never by object) to avoid leakage. An in-process LRU keeps
a few videos' frame files hot; pair with the video-grouped sampler so each video's
~90 MB frames file is loaded once per epoch.
"""

from __future__ import annotations

import glob
import os
from collections import OrderedDict
from typing import List, Optional

import torch
from torch.utils.data import Dataset, Sampler


def list_cached_trajectories(cache_dir: str) -> List[str]:
    return sorted(glob.glob(f"{cache_dir}/traj/*.pt"))


CANONICAL_VAL_SPLIT = (
    "/weka/oe-training-default/jasonr/3d_box/3d_boundingbox_detection/"
    "video_3d_box/itw_3dbox_det/outputs/track_c/val_split.json"
)
# NOTE 2026-07-08: repointed weikaih -> jasonr/3d_box after the workspace
# migration. The old path silently didn't exist, causing RNG-fallback splits
# on any run that didn't pass --val_split_file explicitly (e.g. WX3_adt).


def split_by_video(traj_paths: List[str], val_frac: float = 0.05, seed: int = 0,
                   split_file: str = CANONICAL_VAL_SPLIT):
    """Split trajectory files into train/val by VIDEO id.

    If ``split_file`` exists, the persisted val_video_ids are used verbatim
    (matched to v1..v10's eval set so new runs remain directly comparable).
    Otherwise falls back to the torch-RNG split — useful only for forensics.
    """
    vids = sorted({os.path.basename(p).split("__")[0] for p in traj_paths})
    if split_file and os.path.exists(split_file):
        import json
        with open(split_file) as f:
            val_vids = set(json.load(f)["val_video_ids"])
    else:
        import torch
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(vids), generator=g).tolist()
        n_val = max(1, int(len(vids) * val_frac))
        val_vids = set(vids[i] for i in perm[:n_val])
    train = [p for p in traj_paths if os.path.basename(p).split("__")[0] not in val_vids]
    val = [p for p in traj_paths if os.path.basename(p).split("__")[0] in val_vids]
    return train, val, sorted(val_vids)


class _FramesLRU:
    def __init__(self, cache_dir: str, maxsize: int = 8):
        self.cache_dir = cache_dir
        self.maxsize = maxsize
        self.store: "OrderedDict[str, dict]" = OrderedDict()

    def get(self, video_id: str) -> dict:
        if video_id in self.store:
            self.store.move_to_end(video_id)
            return self.store[video_id]
        fc = torch.load(f"{self.cache_dir}/frames/{video_id}.pt", weights_only=False)
        self.store[video_id] = fc
        if len(self.store) > self.maxsize:
            self.store.popitem(last=False)
        return fc

    def preload(self, video_ids) -> None:
        """Load all given videos' frames files into RAM once and disable
        eviction, so epochs incur zero re-deserialization."""
        vids = sorted(set(video_ids))
        self.maxsize = len(vids) + 4
        import time as _t
        t0 = _t.time()
        for k, v in enumerate(vids):
            if v not in self.store:
                self.store[v] = torch.load(
                    f"{self.cache_dir}/frames/{v}.pt", weights_only=False)
            if (k + 1) % 200 == 0:
                print(f"[preload] {k+1}/{len(vids)} frames files "
                      f"({_t.time()-t0:.0f}s)", flush=True)
        print(f"[preload] {len(vids)} frames files into RAM "
              f"({_t.time()-t0:.0f}s)", flush=True)


class CachedTrackCDataset(Dataset):
    def __init__(self, cache_dir: str, traj_paths: List[str], lru: int = 8,
                 preload: bool = False):
        self.cache_dir = cache_dir
        self.traj_paths = traj_paths
        self.frames = _FramesLRU(cache_dir, maxsize=lru)
        self.traj_cache = None
        if preload:
            self.frames.preload(
                os.path.basename(p).split("__")[0] for p in traj_paths)
            self.traj_cache = [
                torch.load(p, weights_only=False) for p in traj_paths]

    def __len__(self):
        return len(self.traj_paths)

    def video_of(self, i: int) -> str:
        return os.path.basename(self.traj_paths[i]).split("__")[0]

    def traj_T(self, i: int) -> int:
        """Number of frames in trajectory i (cheap; from RAM cache if preloaded)."""
        if self.traj_cache is not None:
            return int(self.traj_cache[i]["box_repr"].shape[0])
        return int(torch.load(self.traj_paths[i], weights_only=False)["box_repr"].shape[0])

    def __getitem__(self, i: int) -> dict:
        traj = (self.traj_cache[i] if self.traj_cache is not None
                else torch.load(self.traj_paths[i], weights_only=False))
        vid = traj["video_id"]
        fc = self.frames.get(vid)
        idxs = traj["frame_index"].tolist()
        # Keep the big tensors bf16 here; convert to float on the GPU in the
        # train step (CPU bf16->fp32 is slow and would stall the GPU).
        depth = torch.stack([fc[ix]["depth_latents"] for ix in idxs], 0)   # bf16 [T,2401,256]
        ray = torch.stack([fc[ix]["ray"] for ix in idxs], 0)              # bf16 [T,2401,81]
        K = fc[idxs[0]]["K"]                                              # [3,3]
        input_hw = fc[idxs[0]]["input_hw"]
        # hidden cached as [T,L,256]; head wants [L,T,1,256]
        hidden = traj["hidden"].permute(1, 0, 2).unsqueeze(2)            # bf16 [L,T,1,256]
        return {
            "hidden": hidden,
            "ray": ray,
            "depth": depth,
            "box2d": traj["box2d"].float(),
            "K": K.float(),
            "box_repr": traj["box_repr"].float(),
            "ts": traj["ts_sec"].float(),
            "measured": traj["measured"].bool(),
            "gt_center": traj["gt_center"].float(),
            "gt_dims": traj["gt_dims"].float(),
            "gt_quat": traj["gt_quat"].float(),
            "input_hw": input_hw,
            "video_id": vid,
            "object_id": traj["object_id"],
            "sel_iou": traj["sel_iou"].float(),
        }


class VideoGroupedSampler(Sampler):
    """Yield trajectory indices grouped by video (keeps the frames LRU hot),
    with videos (and trajectories within a video) shuffled each epoch."""

    def __init__(self, dataset: CachedTrackCDataset, shuffle: bool = True, seed: int = 0):
        self.ds = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.by_video = {}
        for i in range(len(dataset)):
            self.by_video.setdefault(dataset.video_of(i), []).append(i)

    def set_epoch(self, e: int):
        self.epoch = e

    def __iter__(self):
        vids = list(self.by_video.keys())
        if self.shuffle:
            g = torch.Generator().manual_seed(self.seed + self.epoch)
            vids = [vids[i] for i in torch.randperm(len(vids), generator=g).tolist()]
        for v in vids:
            idxs = self.by_video[v]
            if self.shuffle:
                g = torch.Generator().manual_seed(self.seed + self.epoch + hash(v) % 10000)
                idxs = [idxs[i] for i in torch.randperm(len(idxs), generator=g).tolist()]
            for i in idxs:
                yield i

    def __len__(self):
        return len(self.ds)


# ----------------------------------------------------------------------
# COMBINED (multi-source) training: CA-1M + Waymo + ADT with within-batch mix
# ----------------------------------------------------------------------

class CombinedTrackCDataset(Dataset):
    """Wraps N per-dataset CachedTrackCDatasets under a single flat index.

    Each source is filtered by category (via pairing_index if provided) and
    split-by-video with its own val_split_file. Only the TRAIN split is
    exposed by this class; build separate val CachedTrackCDatasets per source
    for the multi-val eval loop.

    Layout in flat-index space (offsets):
        [0 .. len(src0)) [len(src0) .. len(src0)+len(src1)) ...
    """

    def __init__(self, sources, preload: bool = True):
        """sources: list of dicts, each with keys
           name, cache_dir, val_split_file (may be None/"" for RNG),
           pairing_index (optional), categories (comma str, optional),
           val_frac (default 0.05), seed (default 0)."""
        import json as _json
        self.sources = []          # list[CachedTrackCDataset] (train paths only)
        self.names = []            # parallel list[str]
        self.offsets = [0]         # length N+1
        self.per_source_val_paths = {}  # name -> list[str] (for building val loaders)
        self.per_source_all_paths = {}  # name -> list[str] (for reference)
        for s in sources:
            paths = list_cached_trajectories(s["cache_dir"])
            if s.get("categories") and s.get("pairing_index"):
                keep = set(c.strip() for c in s["categories"].split(",") if c.strip())
                cat_of = {}
                for line in open(s["pairing_index"]):
                    d = _json.loads(line)
                    cat_of[f"{d['seg']}__{d['track_id']}"] = d["category"]
                paths = [p for p in paths
                         if cat_of.get(os.path.basename(p)[:-3]) in keep]
            split_kw = ({} if s.get("val_split_file") is None
                        else {"split_file": s["val_split_file"]})
            train_paths, val_paths, _ = split_by_video(
                paths, s.get("val_frac", 0.05),
                seed=s.get("seed", 0), **split_kw)
            ds = CachedTrackCDataset(s["cache_dir"], train_paths, preload=preload)
            self.sources.append(ds)
            self.names.append(s["name"])
            self.offsets.append(self.offsets[-1] + len(ds))
            self.per_source_val_paths[s["name"]] = val_paths
            self.per_source_all_paths[s["name"]] = paths
            print(f"[combined] {s['name']}: {len(paths)} filtered trajs -> "
                  f"train {len(train_paths)} / val {len(val_paths)}", flush=True)
        self.total = self.offsets[-1]

    def __len__(self):
        return self.total

    def _resolve(self, i: int):
        # linear scan is fine for N<=8 sources; binary search wouldn't help
        for k in range(len(self.sources)):
            if i < self.offsets[k + 1]:
                return k, i - self.offsets[k]
        raise IndexError(i)

    def __getitem__(self, i: int) -> dict:
        k, li = self._resolve(i)
        pack = self.sources[k][li]
        pack["dataset_name"] = self.names[k]
        return pack

    def traj_T(self, i: int) -> int:
        k, li = self._resolve(i)
        return self.sources[k].traj_T(li)

    def video_of(self, i: int) -> str:
        k, li = self._resolve(i)
        return f"{self.names[k]}:{self.sources[k].video_of(li)}"


class WeightedMultiSourceSampler(Sampler):
    """Yield flat indices into a CombinedTrackCDataset with within-batch
    weighted mixing. Each yield picks a source with prob proportional to
    weights, then draws a randomly-shuffled index from that source's queue
    (refilled on exhaustion). One epoch = total combined dataset size.
    """

    def __init__(self, combined_ds: CombinedTrackCDataset,
                 weights: List[float], shuffle: bool = True, seed: int = 0):
        assert len(weights) == len(combined_ds.sources), (
            f"weights ({len(weights)}) must match sources ({len(combined_ds.sources)})")
        self.cds = combined_ds
        w = torch.tensor(weights, dtype=torch.float64)
        assert (w > 0).all(), f"weights must be positive: {weights}"
        self.weights = (w / w.sum()).tolist()
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, e: int):
        self.epoch = e

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        n_src = len(self.cds.sources)
        # per-source shuffled queues; refill when empty so no source starves
        queues = []
        for k in range(n_src):
            ds = self.cds.sources[k]
            if self.shuffle:
                perm = torch.randperm(len(ds), generator=g).tolist()
            else:
                perm = list(range(len(ds)))
            queues.append(perm)
        total = len(self.cds)
        source_seq = torch.multinomial(
            torch.tensor(self.weights), num_samples=total,
            replacement=True, generator=g).tolist()
        for k in source_seq:
            if not queues[k]:
                perm = torch.randperm(len(self.cds.sources[k]),
                                      generator=g).tolist()
                queues[k] = perm
            local_i = queues[k].pop()
            yield self.cds.offsets[k] + local_i

    def __len__(self):
        return len(self.cds)


def build_per_source_val_datasets(combined_ds: CombinedTrackCDataset,
                                   preload: bool = True):
    """Return {name: CachedTrackCDataset} for the val split of each source."""
    out = {}
    for i, name in enumerate(combined_ds.names):
        src = combined_ds.sources[i]
        val_paths = combined_ds.per_source_val_paths[name]
        out[name] = CachedTrackCDataset(src.cache_dir, val_paths, preload=preload)
    return out


def collate_trajs(packs: list) -> dict:
    """Collate K per-trajectory packs into one vectorized batch.

    Sequences (for the trajectory encoder) are padded to Tmax with a pad mask;
    per-frame head inputs + GT are concatenated in (k, then t) order so they line
    up with ``encoder_tokens[~pad_mask]``. Per-frame intrinsics are built by
    repeating each trajectory's single K across its frames.
    """
    K = len(packs)
    Ts = [p["box_repr"].shape[0] for p in packs]
    Tmax = max(Ts)
    box_repr = torch.zeros(K, Tmax, packs[0]["box_repr"].shape[1])
    ts = torch.zeros(K, Tmax)
    measured = torch.zeros(K, Tmax, dtype=torch.bool)
    pad_mask = torch.ones(K, Tmax, dtype=torch.bool)
    for k, (p, t) in enumerate(zip(packs, Ts)):
        box_repr[k, :t] = p["box_repr"]
        ts[k, :t] = p["ts"]
        measured[k, :t] = p["measured"]
        pad_mask[k, :t] = False
    hidden = torch.cat([p["hidden"] for p in packs], dim=1)          # [L, sumT, 1, 256]
    ray = torch.cat([p["ray"] for p in packs], dim=0)               # [sumT, ntok, 81]
    depth = torch.cat([p["depth"] for p in packs], dim=0)           # [sumT, ntok, 256]
    box2d = torch.cat([p["box2d"] for p in packs], dim=0)           # [sumT, 4]
    K_pf = torch.cat([p["K"].unsqueeze(0).expand(t, 3, 3)
                      for p, t in zip(packs, Ts)], dim=0)            # [sumT, 3, 3]
    gt_center = torch.cat([p["gt_center"] for p in packs], dim=0)
    gt_dims = torch.cat([p["gt_dims"] for p in packs], dim=0)
    gt_quat = torch.cat([p["gt_quat"] for p in packs], dim=0)
    return {
        "box_repr": box_repr, "ts": ts, "measured": measured, "pad_mask": pad_mask,
        "hidden": hidden, "ray": ray, "depth": depth, "box2d": box2d, "K": K_pf,
        "gt_center": gt_center, "gt_dims": gt_dims, "gt_quat": gt_quat,
        "input_hw": packs[0]["input_hw"],
        "sizes": Ts,   # per-trajectory frame counts (concat order), for derivative loss
    }
