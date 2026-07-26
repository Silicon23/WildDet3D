"""Target-complete cached dataset for v2 Track C.

The cache is intentionally a single universe containing all three GT datasets,
while every trajectory retains an explicit ``prompt_variant_id``.  Splits are
scene-level (CA-1M groups all clips from the same source video) and stratified
by dataset.  No IoU or category filtering exists in this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset, Sampler

from wilddet3d.track_c.v2_precompute import VARIANT_TO_ID


SOURCE_ORDER = ("ca1m", "adt", "waymo")


def list_v2_trajectory_paths(cache_root: Path | str) -> list[Path]:
    return sorted(Path(cache_root).glob("traj/*.pt"))


def load_v2_trajectories(cache_root: Path | str) -> list[dict[str, Any]]:
    paths = list_v2_trajectory_paths(cache_root)
    records = []
    for path in paths:
        record = torch.load(path, map_location="cpu", weights_only=False)
        record["_cache_path"] = str(path)
        variant = record.get("prompt_variant")
        variant_id = int(record.get("prompt_variant_id", -1))
        if variant not in VARIANT_TO_ID or variant_id != VARIANT_TO_ID[variant]:
            raise ValueError(
                f"variant contract failure in {path}: {variant!r}/{variant_id}"
            )
        if record.get("dataset") not in SOURCE_ORDER:
            raise ValueError(f"unsupported dataset in {path}: {record.get('dataset')}")
        records.append(record)
    return records


def scene_group(record: dict[str, Any]) -> str:
    dataset = str(record["dataset"])
    unit = str(record["unit_id"])
    # Multiple CA-1M clips from one source video must not cross the split.
    scene = unit.split("__", 1)[0] if dataset == "ca1m" else unit
    return f"{dataset}:{scene}"


def _groups_digest(groups: Iterable[str]) -> str:
    payload = "\n".join(sorted(groups)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.tmp.", delete=False
    ) as f:
        tmp = Path(f.name)
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def build_or_load_scene_split(
    records: list[dict[str, Any]],
    split_path: Path | str,
    val_fraction: float = 0.08,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return train/val records using a persisted dataset-stratified split."""
    split_path = Path(split_path)
    groups_by_dataset: dict[str, set[str]] = defaultdict(set)
    for record in records:
        groups_by_dataset[str(record["dataset"])].add(scene_group(record))
    all_groups = set().union(*groups_by_dataset.values()) if groups_by_dataset else set()
    digest = _groups_digest(all_groups)

    if split_path.exists():
        with split_path.open() as f:
            payload = json.load(f)
        if payload.get("all_groups_sha256") != digest:
            raise ValueError(
                "persisted split does not match the complete cache universe: "
                f"{payload.get('all_groups_sha256')} != {digest}"
            )
        val_groups = set(payload["val_groups"])
    else:
        val_groups: set[str] = set()
        per_dataset = {}
        for dataset in SOURCE_ORDER:
            groups = sorted(groups_by_dataset.get(dataset, set()))
            if not groups:
                raise ValueError(f"cache has no {dataset} trajectories")
            ranked = sorted(
                groups,
                key=lambda group: hashlib.sha256(
                    f"{seed}|{group}".encode("utf-8")
                ).digest(),
            )
            n_val = max(1, int(round(len(ranked) * val_fraction)))
            chosen = ranked[:n_val]
            val_groups.update(chosen)
            per_dataset[dataset] = {
                "groups_total": len(groups),
                "groups_val": len(chosen),
            }
        payload = {
            "schema_version": "v2_track_c_scene_split_v1",
            "seed": seed,
            "val_fraction": val_fraction,
            "grouping": {
                "ca1m": "source video_id (all clips together)",
                "adt": "sequence",
                "waymo": "segment",
            },
            "all_groups_sha256": digest,
            "val_groups": sorted(val_groups),
            "per_dataset": per_dataset,
        }
        _atomic_json(split_path, payload)

    train = [record for record in records if scene_group(record) not in val_groups]
    val = [record for record in records if scene_group(record) in val_groups]
    train_groups = {scene_group(record) for record in train}
    val_groups_observed = {scene_group(record) for record in val}
    if train_groups & val_groups_observed:
        raise AssertionError("scene leakage between train and validation")
    for dataset in SOURCE_ORDER:
        if not any(record["dataset"] == dataset for record in train):
            raise ValueError(f"no {dataset} train trajectories")
        if not any(record["dataset"] == dataset for record in val):
            raise ValueError(f"no {dataset} validation trajectories")
    return train, val, payload


class V2FrameStore:
    """Shared RAM store for deduplicated per-unit frame tensors."""

    def __init__(self, cache_root: Path | str) -> None:
        self.cache_root = Path(cache_root)
        self.frames: dict[str, dict[int, dict[str, Any]]] = {}

    def get(self, video_key: str) -> dict[int, dict[str, Any]]:
        if video_key not in self.frames:
            path = self.cache_root / "frames" / f"{video_key}.pt"
            self.frames[video_key] = torch.load(
                path, map_location="cpu", weights_only=False
            )
        return self.frames[video_key]

    def preload(self, video_keys: Iterable[str], progress_every: int = 200) -> None:
        keys = sorted(set(video_keys))
        for index, key in enumerate(keys, 1):
            self.get(key)
            if progress_every and index % progress_every == 0:
                print(f"[preload] frame caches {index:,}/{len(keys):,}", flush=True)
        print(f"[preload] frame caches {len(keys):,}/{len(keys):,}", flush=True)


class V2CachedTrackDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        frame_store: V2FrameStore,
    ) -> None:
        self.records = records
        self.frame_store = frame_store

    def __len__(self) -> int:
        return len(self.records)

    def traj_T(self, index: int) -> int:
        return int(self.records[index]["box_repr"].shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        trajectory = self.records[index]
        frame_cache = self.frame_store.get(trajectory["video_key"])
        frame_indices = trajectory["vggt_frame_index"].tolist()
        frames = [frame_cache[int(frame_index)] for frame_index in frame_indices]
        input_hw = tuple(int(x) for x in frames[0]["input_hw"])
        if any(tuple(frame["input_hw"]) != input_hw for frame in frames):
            raise ValueError(f"input_hw changes in {trajectory['video_key']}")
        # trajectory hidden is [T,L,256]; head consumes [L,T,1,256].
        hidden = trajectory["hidden"].permute(1, 0, 2).unsqueeze(2)
        return {
            "hidden": hidden,
            "ray": torch.stack([frame["ray"] for frame in frames]),
            "depth": torch.stack([frame["depth_latents"] for frame in frames]),
            "K": torch.stack([frame["K"] for frame in frames]).float(),
            "box2d": trajectory["box2d"].float(),
            "box_repr": trajectory["box_repr"].float(),
            "ts": trajectory["ts_sec"].float(),
            "measured": trajectory["measured"].bool(),
            "gt_center": trajectory["gt_center"].float(),
            "gt_dims": trajectory["gt_dims"].float(),
            "gt_quat": trajectory["gt_quat"].float(),
            "input_hw": input_hw,
            "prompt_variant": trajectory["prompt_variant"],
            "prompt_variant_id": int(trajectory["prompt_variant_id"]),
            "dataset": trajectory["dataset"],
            "unit_id": trajectory["unit_id"],
            "track_id": trajectory["track_id"],
            "category": trajectory["category"],
            "source_frame_index": trajectory["source_frame_index"].long(),
            "iou3d_input": trajectory["iou3d_input"].float(),
        }


class WeightedCoverageSampler(Sampler[int]):
    """Weighted source mixing with deterministic whole-corpus coverage.

    One epoch remains the natural combined train-set size, matching the proven
    v1 schedule rather than expanding an epoch until the largest source is
    exhausted. Each source has a continuous shuffled queue across epochs, so
    every trajectory is visited within ``coverage_epochs[source]`` while the
    per-epoch source proportions remain exact up to integer rounding.
    """

    def __init__(
        self,
        dataset: V2CachedTrackDataset,
        source_weights: dict[str, float],
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0
        self.by_source: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(dataset.records):
            self.by_source[str(record["dataset"])].append(index)
        if set(self.by_source) != set(source_weights):
            raise ValueError(
                f"source weight keys {set(source_weights)} do not match "
                f"dataset sources {set(self.by_source)}"
            )
        total_weight = sum(float(weight) for weight in source_weights.values())
        if total_weight <= 0 or any(weight <= 0 for weight in source_weights.values()):
            raise ValueError(f"source weights must be positive: {source_weights}")
        self.weights = {
            source: float(weight) / total_weight
            for source, weight in source_weights.items()
        }
        self.epoch_size = len(dataset)
        raw_counts = {
            source: self.epoch_size * self.weights[source]
            for source in self.by_source
        }
        self.counts = {
            source: max(1, int(math.floor(value)))
            for source, value in raw_counts.items()
        }
        difference = self.epoch_size - sum(self.counts.values())
        ranked = sorted(
            self.by_source,
            key=lambda source: (
                raw_counts[source] - math.floor(raw_counts[source]),
                source,
            ),
            reverse=True,
        )
        for index in range(abs(difference)):
            source = ranked[index % len(ranked)]
            self.counts[source] += 1 if difference > 0 else -1
        if any(count < 1 for count in self.counts.values()):
            raise ValueError(f"invalid source allocation {self.counts}")
        self.coverage_epochs = {
            source: int(math.ceil(len(indices) / self.counts[source]))
            for source, indices in self.by_source.items()
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        sequence = []
        for source, count in self.counts.items():
            sequence.extend([source] * count)
        order = torch.randperm(len(sequence), generator=generator).tolist()
        sequence = [sequence[index] for index in order]

        queues: dict[str, list[int]] = {}
        for source_index, source in enumerate(sorted(self.by_source)):
            indices = self.by_source[source]
            count = self.counts[source]
            start = self.epoch * count
            selected = []
            while len(selected) < count:
                absolute = start + len(selected)
                cycle = absolute // len(indices)
                offset = absolute % len(indices)
                cycle_generator = torch.Generator().manual_seed(
                    self.seed + 1_000_003 * (source_index + 1) + cycle
                )
                permutation = torch.randperm(
                    len(indices), generator=cycle_generator
                ).tolist()
                take = min(count - len(selected), len(indices) - offset)
                selected.extend(
                    indices[permutation[position]]
                    for position in range(offset, offset + take)
                )
            queues[source] = selected
        cursors = defaultdict(int)
        for source in sequence:
            yield queues[source][cursors[source]]
            cursors[source] += 1

    def __len__(self) -> int:
        return self.epoch_size


def collate_v2_trajectories(packs: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(packs)
    sizes = [int(pack["box_repr"].shape[0]) for pack in packs]
    max_frames = max(sizes)
    box_repr = torch.zeros(count, max_frames, 12)
    timestamps = torch.zeros(count, max_frames)
    measured = torch.zeros(count, max_frames, dtype=torch.bool)
    pad_mask = torch.ones(count, max_frames, dtype=torch.bool)
    variant_ids = torch.zeros(count, dtype=torch.long)
    for index, (pack, frames) in enumerate(zip(packs, sizes)):
        box_repr[index, :frames] = pack["box_repr"]
        timestamps[index, :frames] = pack["ts"]
        measured[index, :frames] = pack["measured"]
        pad_mask[index, :frames] = False
        variant_ids[index] = int(pack["prompt_variant_id"])

    input_hw = packs[0]["input_hw"]
    if any(pack["input_hw"] != input_hw for pack in packs):
        raise ValueError("mixed model input sizes in one batch")
    variant_ids_per_frame = torch.repeat_interleave(
        variant_ids, torch.tensor(sizes, dtype=torch.long)
    )
    return {
        "box_repr": box_repr,
        "ts": timestamps,
        "measured": measured,
        "pad_mask": pad_mask,
        "hidden": torch.cat([pack["hidden"] for pack in packs], dim=1),
        "ray": torch.cat([pack["ray"] for pack in packs], dim=0),
        "depth": torch.cat([pack["depth"] for pack in packs], dim=0),
        "box2d": torch.cat([pack["box2d"] for pack in packs], dim=0),
        "K": torch.cat([pack["K"] for pack in packs], dim=0),
        "gt_center": torch.cat([pack["gt_center"] for pack in packs], dim=0),
        "gt_dims": torch.cat([pack["gt_dims"] for pack in packs], dim=0),
        "gt_quat": torch.cat([pack["gt_quat"] for pack in packs], dim=0),
        "iou3d_input": torch.cat([pack["iou3d_input"] for pack in packs], dim=0),
        "prompt_variant_id": variant_ids,
        "prompt_variant_id_per_frame": variant_ids_per_frame,
        "datasets": [pack["dataset"] for pack in packs],
        "variants": [pack["prompt_variant"] for pack in packs],
        "unit_ids": [pack["unit_id"] for pack in packs],
        "track_ids": [pack["track_id"] for pack in packs],
        "categories": [pack["category"] for pack in packs],
        "input_hw": input_hw,
        "sizes": sizes,
    }
