"""Train the v2 image-grounded Track C refiner on all three GT datasets.

The two point-prompt lineages share the refiner only through an explicit
learned variant token.  Data are split at scene level, no pair is filtered by
IoU or category, and every source trajectory is visited within a deterministic
finite epoch horizon. The proven v1 combined recipe is retained: direct output,
four-layer
bidirectional trajectory encoder, xattn-only multi-token temporal context,
5% masked frames, and no explicit smoothness regularizer.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import random
import signal
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor
from vis4d.op.geometry.rotation import matrix_to_quaternion, quaternion_to_matrix

# cuDNN SDPA cannot build an execution plan for some Transformer attention
# shapes on Blackwell (sm_103). Keep flash, memory-efficient, and math SDPA
# available while excluding only the failing cuDNN backend.
if torch.cuda.is_available():
    torch.backends.cuda.enable_cudnn_sdp(False)

from wilddet3d.ops.iou_3d_safe import batch_box3d_iou
from wilddet3d.ops.rotation import rotation_6d_to_matrix
from wilddet3d.track_c.losses import encode_targets_batched
from wilddet3d.track_c.v2_dataset import (
    WeightedCoverageSampler,
    V2CachedTrackDataset,
    V2FrameStore,
    build_or_load_scene_split,
    collate_v2_trajectories,
    load_v2_trajectories,
)
from wilddet3d.track_c.v2_losses import (
    physical_dims_rotation_errors,
    proper_axis_relabelings,
    v2_track_c_loss,
)
from wilddet3d.track_c.v2_refiner import V2TrackCRefiner


SOURCE_WEIGHTS = {"ca1m": 0.35, "adt": 0.40, "waymo": 0.25}
EXPECTED_VARIANTS = {
    "ca1m": "point_v3",
    "adt": "point_v3",
    "waymo": "point_vlm_v1",
}
AUTHORITATIVE_COMBINED = {
    "point_v3": "pairs.jsonl",
    "point_vlm_v1": "pairs_vlm_v1.jsonl",
}
RESUME_SCHEMA_VERSION = "v2_track_c_resume_v2"
RESUME_COMPATIBILITY_KEYS = (
    "epochs",
    "batch_trajs",
    "max_frames_per_batch",
    "lr",
    "temporal_lr_mult",
    "variant_lr_mult",
    "mask_frame_p",
    "warmup_steps",
    "val_fraction",
    "seed",
)


class PreemptionHandler:
    """Defer termination until the current optimizer step is checkpointed."""

    def __init__(self) -> None:
        self.signal_number: int | None = None
        self._previous_handlers: dict[int, Any] = {}

    @property
    def requested(self) -> bool:
        return self.signal_number is not None

    def _handle(self, signal_number: int, _frame: Any) -> None:
        if self.signal_number is None:
            self.signal_number = signal_number

    def install(self) -> None:
        for signal_number in (signal.SIGTERM, signal.SIGINT):
            self._previous_handlers[signal_number] = signal.getsignal(signal_number)
            signal.signal(signal_number, self._handle)

    def terminate(self) -> None:
        """Restore default handling and preserve the scheduler's signal exit."""
        if self.signal_number is None:
            return
        signal_number = self.signal_number
        signal.signal(signal_number, signal.SIG_DFL)
        os.kill(os.getpid(), signal_number)
        raise SystemExit(128 + signal_number)


def enable_parent_death_signal() -> None:
    """Ask Linux to SIGTERM Python if its training wrapper disappears."""
    if not sys.platform.startswith("linux"):
        return
    parent_pid = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    pr_set_pdeathsig = 1
    if libc.prctl(pr_set_pdeathsig, signal.SIGTERM, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    # Close the race where the parent died immediately before prctl().
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGTERM)


def capture_rng_state(device: str) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any], device: str) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if (
        device.startswith("cuda")
        and torch.cuda.is_available()
        and "torch_cuda" in state
    ):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def groups_sha256(groups: Iterable[Iterable[int]]) -> str:
    digest = hashlib.sha256()
    for group in groups:
        digest.update(b"[")
        for index in group:
            digest.update(str(int(index)).encode("ascii"))
            digest.update(b",")
        digest.update(b"]")
    return digest.hexdigest()


def validate_resume_compatibility(
    saved_args: dict[str, Any], current_args: argparse.Namespace
) -> None:
    mismatches = {
        key: (saved_args.get(key), getattr(current_args, key))
        for key in RESUME_COMPATIBILITY_KEYS
        if saved_args.get(key) != getattr(current_args, key)
    }
    if mismatches:
        raise ValueError(
            "resume checkpoint training arguments changed: "
            + json.dumps(mismatches, sort_keys=True)
        )


def validate_cache_completion(
    cache_root: Path, pairs_root: Path, verification_root: Path
) -> dict[str, Any]:
    """Fail closed unless the cache exactly covers both finalized manifests."""
    expected: dict[str, tuple[Path, str]] = {}
    report_rows = 0
    report_tracks = 0
    for variant in ("point_v3", "point_vlm_v1"):
        variant_dir = pairs_root / variant
        verification_path = (
            verification_root / variant / "track_c_verification.json"
        )
        if not verification_path.is_file():
            raise FileNotFoundError(
                f"target manifest is not finalized: {verification_path}"
            )
        with verification_path.open() as handle:
            verification = json.load(handle)
        if (
            verification.get("schema_version")
            != "v2_track_c_target_verification_v1"
            or verification.get("status") != "verified"
            or verification.get("prompt_variant") != variant
        ):
            raise ValueError(
                f"invalid Track C target marker: {verification_path}"
            )
        output_path = Path(str(verification["output_path"]))
        if output_path.name != AUTHORITATIVE_COMBINED[variant]:
            raise ValueError(
                f"non-authoritative target verification for {variant}: "
                f"{output_path.name}"
            )
        report_rows += int(verification["rows"])
        report_tracks += int(verification["tracks"])
        unit_paths = sorted(
            path
            for path in variant_dir.glob("*.pairs.jsonl")
            if path.stat().st_size > 0
        )
        if len(unit_paths) != int(verification["units"]):
            raise ValueError(
                f"target unit count mismatch for {variant}: "
                f"{len(unit_paths)} != {verification['units']}"
            )
        for path in unit_paths:
            with path.open() as handle:
                row = json.loads(next(line for line in handle if line.strip()))
            key = f"{variant}__{row['dataset']}__{row['sequence']}"
            if key in expected:
                raise ValueError(f"duplicate target cache key {key}")
            expected[key] = (path, str(row["dataset"]))

    observed_paths = sorted((cache_root / ".done").glob("*.json"))
    observed = {path.stem: path for path in observed_paths}
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    if missing or extra:
        raise ValueError(
            f"cache completion mismatch: missing={len(missing)} extra={len(extra)} "
            f"examples_missing={missing[:3]} examples_extra={extra[:3]}"
        )
    cached_rows = 0
    cached_tracks = 0
    query_iou_min = 1.0
    query_iou_weighted_sum = 0.0
    query_iou_below_threshold = 0
    trajectory_paths: set[str] = set()
    for key, marker_path in observed.items():
        with marker_path.open() as handle:
            marker = json.load(handle)
        if (
            marker.get("status") != "done"
            or marker.get("schema_version") != "v2_track_c_cache_v4"
        ):
            raise ValueError(f"non-done marker {marker_path}")
        if (
            f"{marker['prompt_variant']}__{marker['dataset']}__{marker['unit_id']}"
            != key
        ):
            raise ValueError(f"marker identity mismatch {marker_path}")
        expected_pairs_path = expected[key][0].resolve()
        if Path(marker["pairs_path"]).resolve() != expected_pairs_path:
            raise ValueError(f"marker pair path mismatch {marker_path}")
        digest = hashlib.sha256()
        with expected_pairs_path.open("rb") as handle:
            while chunk := handle.read(8 << 20):
                digest.update(chunk)
        if digest.hexdigest() != marker["pairs_sha256"]:
            raise ValueError(f"marker pair hash is stale {marker_path}")
        cached_rows += int(marker["pairs"])
        cached_tracks += int(marker["tracks"])
        for trajectory_path in marker["trajectory_paths"]:
            resolved = str(Path(trajectory_path).resolve())
            if resolved in trajectory_paths:
                raise ValueError(f"duplicate trajectory path {resolved}")
            if not Path(resolved).is_file():
                raise FileNotFoundError(resolved)
            trajectory_paths.add(resolved)
        if len(marker["trajectory_paths"]) != int(marker["tracks"]):
            raise ValueError(f"track count mismatch in {marker_path}")
        if not Path(marker["frame_cache_path"]).is_file():
            raise FileNotFoundError(marker["frame_cache_path"])
        query_iou = marker.get("query_reference_iou", {})
        if (
            query_iou.get("selection") != "argmax_no_gate"
            or float(query_iou.get("diagnostic_low_threshold", -1)) != 0.5
        ):
            raise ValueError(
                f"query correspondence contract mismatch in {marker_path}"
            )
        query_iou_min = min(query_iou_min, float(query_iou["min"]))
        query_iou_weighted_sum += float(query_iou["mean"]) * int(marker["pairs"])
        query_iou_below_threshold += int(
            query_iou["below_diagnostic_threshold"]
        )
    actual_trajectories = {
        str(path.resolve()) for path in (cache_root / "traj").glob("*.pt")
    }
    if actual_trajectories != trajectory_paths:
        raise ValueError(
            f"trajectory file universe mismatch: marker={len(trajectory_paths)} "
            f"filesystem={len(actual_trajectories)}"
        )
    if cached_rows != report_rows or cached_tracks != report_tracks:
        raise ValueError(
            f"cache totals mismatch: rows={cached_rows}/{report_rows}, "
            f"tracks={cached_tracks}/{report_tracks}"
        )
    return {
        "units": len(expected),
        "rows": cached_rows,
        "tracks": cached_tracks,
        "variants": ["point_v3", "point_vlm_v1"],
        "track_c_post_pair_filtering": "none",
        "ca1m_upstream_suitability_filter": "already_applied",
        "query_correspondence": {
            "selection": "argmax_no_gate",
            "diagnostic_low_threshold": 0.5,
            "min": query_iou_min,
            "mean": query_iou_weighted_sum / cached_rows,
            "below_diagnostic_threshold": query_iou_below_threshold,
            "below_diagnostic_fraction": (
                query_iou_below_threshold / cached_rows
            ),
        },
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=f".{path.name}.tmp.", delete=False
    ) as handle:
        tmp = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.tmp.", delete=False
    ) as handle:
        tmp = Path(handle.name)
    try:
        torch.save(value, tmp)
        with tmp.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def to_device(value: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        key: item.to(device, non_blocking=True) if torch.is_tensor(item) else item
        for key, item in value.items()
    }


def validate_universe(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_source = defaultdict(lambda: {"tracks": 0, "frames": 0, "variants": set()})
    for record in records:
        dataset = str(record["dataset"])
        variant = str(record["prompt_variant"])
        if dataset not in EXPECTED_VARIANTS:
            raise ValueError(f"unexpected source {dataset!r}")
        if variant != EXPECTED_VARIANTS[dataset]:
            raise ValueError(
                f"D12 violation: {dataset} must be {EXPECTED_VARIANTS[dataset]}, "
                f"found {variant}"
            )
        frames = int(record["box_repr"].shape[0])
        if frames < 1:
            raise ValueError(f"empty trajectory {record.get('_cache_path')}")
        by_source[dataset]["tracks"] += 1
        by_source[dataset]["frames"] += frames
        by_source[dataset]["variants"].add(variant)
    if set(by_source) != set(EXPECTED_VARIANTS):
        raise ValueError(
            f"training requires all three sources: found {sorted(by_source)}"
        )
    return {
        dataset: {
            "tracks": value["tracks"],
            "frames": value["frames"],
            "variants": sorted(value["variants"]),
        }
        for dataset, value in sorted(by_source.items())
    }


def build_groups(
    indices: Iterable[int],
    dataset: V2CachedTrackDataset,
    batch_trajs: int,
    max_frames: int,
) -> list[list[int]]:
    groups: list[list[int]] = []
    current: list[int] = []
    current_frames = 0
    for index in indices:
        frames = dataset.traj_T(index)
        if current and (
            len(current) >= batch_trajs or current_frames + frames > max_frames
        ):
            groups.append(current)
            current = []
            current_frames = 0
        current.append(index)
        current_frames += frames
    if current:
        groups.append(current)
    return groups


def _rotation_steps_deg(R: Tensor, dt: Tensor) -> Tensor:
    # A cuboid's local axes may be relabeled by any of 24 proper signed
    # permutations without changing its physical geometry. Minimize each
    # temporal step over that group so annotation/coder 90-degree axis swaps
    # are not reported as motion.
    relabelings, _ = proper_axis_relabelings(R.device, R.dtype)
    next_candidates = torch.einsum("nij,sjk->nsik", R[1:], relabelings)
    relative = torch.matmul(
        R[:-1].unsqueeze(1).transpose(-1, -2), next_candidates
    )
    trace = relative[..., 0, 0] + relative[..., 1, 1] + relative[..., 2, 2]
    angle = torch.acos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0)).min(dim=1).values
    return torch.rad2deg(angle) / dt


def time_metrics(
    pred_center: Tensor,
    pred_R: Tensor,
    gt_center: Tensor,
    gt_R: Tensor,
    ts: Tensor,
) -> dict[str, float]:
    """Ground-truth-referenced, cadence-aware temporal metrics for one track."""
    dt = ts[1:] - ts[:-1]
    if len(dt) == 0 or bool((dt <= 1e-6).any()):
        return {}
    pred_velocity = (pred_center[1:] - pred_center[:-1]) / dt[:, None]
    gt_velocity = (gt_center[1:] - gt_center[:-1]) / dt[:, None]
    pred_rot_speed = _rotation_steps_deg(pred_R, dt)
    gt_rot_speed = _rotation_steps_deg(gt_R, dt)
    result = {
        "velocity_residual_mps": float(
            (pred_velocity - gt_velocity).norm(dim=-1).mean()
        ),
        "angular_speed_residual_degps": float(
            (pred_rot_speed - gt_rot_speed).abs().mean()
        ),
        "center_speed_mps": float(pred_velocity.norm(dim=-1).mean()),
        "angular_speed_degps": float(pred_rot_speed.mean()),
    }
    if len(dt) >= 2:
        acceleration_dt = (dt[1:] + dt[:-1]) * 0.5
        pred_acceleration = (
            pred_velocity[1:] - pred_velocity[:-1]
        ) / acceleration_dt[:, None]
        gt_acceleration = (
            gt_velocity[1:] - gt_velocity[:-1]
        ) / acceleration_dt[:, None]
        result.update(
            acceleration_residual_mps2=float(
                (pred_acceleration - gt_acceleration).norm(dim=-1).mean()
            ),
            center_acceleration_mps2=float(
                pred_acceleration.norm(dim=-1).mean()
            ),
            angular_acceleration_degps2=float(
                (
                    (pred_rot_speed[1:] - pred_rot_speed[:-1])
                    / acceleration_dt
                )
                .abs()
                .mean()
            ),
        )
    return result


class MetricAccumulator:
    def __init__(self) -> None:
        self.frame_sums = defaultdict(float)
        self.frames = 0
        self.track_values: dict[str, list[float]] = defaultdict(list)

    def add_frames(self, metrics: dict[str, Tensor]) -> None:
        if not metrics:
            return
        count = int(next(iter(metrics.values())).numel())
        self.frames += count
        for name, values in metrics.items():
            self.frame_sums[name] += float(values.sum())

    def add_track(self, values: dict[str, float]) -> None:
        for name, value in values.items():
            if math.isfinite(value):
                self.track_values[name].append(value)

    def result(self) -> dict[str, float | int]:
        result: dict[str, float | int] = {
            name: value / max(self.frames, 1)
            for name, value in self.frame_sums.items()
        }
        result.update(
            {
                name: float(np.mean(values))
                for name, values in self.track_values.items()
                if values
            }
        )
        result["frames"] = self.frames
        result["tracks_with_time_metrics"] = max(
            (len(values) for values in self.track_values.values()), default=0
        )
        return result


def box_frame_metrics(
    center: Tensor,
    dims: Tensor,
    R: Tensor,
    gt_center: Tensor,
    gt_dims: Tensor,
    gt_R: Tensor,
    gt_quat: Tensor,
) -> dict[str, Tensor]:
    dims_error, rot_error = physical_dims_rotation_errors(
        dims, R, gt_dims, gt_R
    )
    box = torch.cat([center, dims, matrix_to_quaternion(R)], dim=-1)
    gt_box = torch.cat([gt_center, gt_dims, gt_quat], dim=-1)
    return {
        "center_m": (center - gt_center).norm(dim=-1),
        "dims_m": dims_error,
        "rot_deg": torch.rad2deg(rot_error),
        "iou3d": batch_box3d_iou(box, gt_box).to(center.device),
    }


@torch.no_grad()
def evaluate(
    refiner: V2TrackCRefiner,
    dataset: V2CachedTrackDataset,
    device: str,
    batch_trajs: int,
    max_frames: int,
    variant_intervention: str = "correct",
) -> dict[str, Any]:
    refiner.eval()
    accumulators: dict[str, dict[str, MetricAccumulator]] = defaultdict(
        lambda: {"input": MetricAccumulator(), "track_c": MetricAccumulator()}
    )
    groups = build_groups(range(len(dataset)), dataset, batch_trajs, max_frames)
    for group in groups:
        packs = [dataset[index] for index in group]
        batch = to_device(collate_v2_trajectories(packs), device)
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")
        ):
            output = refiner.forward_vectorized(
                batch, variant_intervention=variant_intervention
            )
        decoded = refiner.decode_layer(
            output["reg"][-1, :, 0, :].float(),
            batch["box2d"].float(),
            batch["K"].float(),
            batch["input_hw"],
        )
        predicted = (
            decoded[:, :3],
            decoded[:, 3:6],
            quaternion_to_matrix(decoded[:, 6:10]),
        )
        prior_flat = batch["box_repr"][~batch["pad_mask"]].float()
        input_box = (
            prior_flat[:, :3],
            prior_flat[:, 3:6].exp(),
            rotation_6d_to_matrix(prior_flat[:, 6:12]),
        )
        gt_center = batch["gt_center"].float()
        gt_dims = batch["gt_dims"].float()
        gt_quat = batch["gt_quat"].float()
        gt_R = quaternion_to_matrix(gt_quat)
        offset = 0
        for pack_index, frames in enumerate(batch["sizes"]):
            sl = slice(offset, offset + int(frames))
            offset += int(frames)
            key = f"{batch['datasets'][pack_index]}/{batch['variants'][pack_index]}"
            for source, values in (("input", input_box), ("track_c", predicted)):
                center, dims, R = (value[sl] for value in values)
                accumulators[key][source].add_frames(
                    box_frame_metrics(
                        center,
                        dims,
                        R,
                        gt_center[sl],
                        gt_dims[sl],
                        gt_R[sl],
                        gt_quat[sl],
                    )
                )
                accumulators[key][source].add_track(
                    time_metrics(
                        center,
                        R,
                        gt_center[sl],
                        gt_R[sl],
                        packs[pack_index]["ts"].to(device),
                    )
                )

    per_group = {
        key: {source: accumulator.result() for source, accumulator in values.items()}
        for key, values in sorted(accumulators.items())
    }
    unified: dict[str, float] = {}
    for metric in ("iou3d", "center_m", "dims_m", "rot_deg"):
        unified[metric] = sum(
            SOURCE_WEIGHTS[key.split("/", 1)[0]]
            * float(values["track_c"][metric])
            for key, values in per_group.items()
        )
    return {
        "variant_intervention": variant_intervention,
        "per_dataset_variant": per_group,
        "unified_track_c": unified,
    }


def load_v1_warm_start(refiner: V2TrackCRefiner, path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("refiner", checkpoint)
    missing, unexpected = refiner.load_state_dict(state, strict=False)
    allowed_missing = {"prompt_variant_embed.weight"}
    if set(missing) != allowed_missing or unexpected:
        raise ValueError(
            f"warm-start architecture mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    return {
        "path": str(path),
        "epoch": checkpoint.get("epoch"),
        "missing_initialized": sorted(missing),
    }


def learning_rate_scale(
    step: int, warmup_steps: int, total_steps: int
) -> float:
    if warmup_steps and step < warmup_steps:
        return max(1e-3, (step + 1) / warmup_steps)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_root", type=Path, required=True)
    parser.add_argument("--pairs_root", type=Path, required=True)
    parser.add_argument("--verification_root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--split_file", type=Path)
    parser.add_argument(
        "--init_from",
        type=Path,
        required=True,
        help="v1 combined Track C checkpoint used as the architecture warm start",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_trajs", type=int, default=4)
    parser.add_argument("--max_frames_per_batch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temporal_lr_mult", type=float, default=20.0)
    parser.add_argument("--variant_lr_mult", type=float, default=5.0)
    parser.add_argument("--mask_frame_p", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--val_fraction", type=float, default=0.08)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preload_frames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--auto_resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--checkpoint_every_steps",
        type=int,
        default=250,
        help="atomic mid-epoch checkpoint interval; 0 disables the step trigger",
    )
    parser.add_argument(
        "--checkpoint_every_seconds",
        type=float,
        default=1800.0,
        help="atomic mid-epoch checkpoint interval; 0 disables the time trigger",
    )
    parser.add_argument("--max_epochs_this_run", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_file = args.split_file or args.out_dir / "scene_split.json"
    done_path = args.out_dir / ".done"
    if done_path.exists():
        print(f"[train] already complete: {done_path}", flush=True)
        return
    if not (0 <= args.mask_frame_p < 1):
        raise ValueError("--mask_frame_p must be in [0,1)")
    if args.checkpoint_every_steps < 0 or args.checkpoint_every_seconds < 0:
        raise ValueError("checkpoint intervals must be non-negative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    print("[data] verifying exact cache completion", flush=True)
    cache_verification = validate_cache_completion(
        args.cache_root, args.pairs_root, args.verification_root
    )
    print(
        "[data] exact cache "
        + json.dumps(cache_verification, sort_keys=True),
        flush=True,
    )
    print("[data] loading target-complete trajectories", flush=True)
    records = load_v2_trajectories(args.cache_root)
    universe = validate_universe(records)
    train_records, val_records, split = build_or_load_scene_split(
        records, split_file, val_fraction=args.val_fraction, seed=args.seed
    )
    frame_store = V2FrameStore(args.cache_root)
    if args.preload_frames:
        frame_store.preload(record["video_key"] for record in records)
    train_dataset = V2CachedTrackDataset(train_records, frame_store)
    val_dataset = V2CachedTrackDataset(val_records, frame_store)
    sampler = WeightedCoverageSampler(
        train_dataset, SOURCE_WEIGHTS, seed=args.seed
    )
    data_report = {
        "universe": universe,
        "train_tracks": len(train_records),
        "val_tracks": len(val_records),
        "sampler_epoch_size": len(sampler),
        "sampler_source_counts": sampler.counts,
        "sampler_coverage_epochs": sampler.coverage_epochs,
        "source_weights": SOURCE_WEIGHTS,
        "split": split,
        "track_c_post_pair_filtering": "none",
        "ca1m_upstream_suitability_filter": "already_applied",
        "cache_verification": cache_verification,
    }
    atomic_json(args.out_dir / "data_report.json", data_report)
    print(
        f"[data] train={len(train_records):,} val={len(val_records):,} "
        f"epoch_samples={len(sampler):,} source_counts={sampler.counts}",
        flush=True,
    )

    refiner = V2TrackCRefiner(
        reg_residual_from_prior=False,
        use_temporal_modules=True,
        use_layer_bias=False,
        use_temporal_kv_norm=True,
        temporal_multi_token=True,
        temporal_block="xattn_only",
    ).to(args.device)
    warm_start = load_v1_warm_start(refiner, args.init_from)

    temporal_parameters, variant_parameters, base_parameters = [], [], []
    temporal_names = (
        "temporal_gate",
        "prompt_temporal",
        "project_temporal",
        "temporal_kv_norm",
        "traj_encoder",
    )
    for name, parameter in refiner.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("prompt_variant_embed."):
            variant_parameters.append(parameter)
        elif any(token in name for token in temporal_names):
            temporal_parameters.append(parameter)
        else:
            base_parameters.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": base_parameters, "lr": args.lr, "name": "base"},
            {
                "params": temporal_parameters,
                "lr": args.lr * args.temporal_lr_mult,
                "name": "temporal",
            },
            {
                "params": variant_parameters,
                "lr": args.lr * args.variant_lr_mult,
                "name": "variant",
            },
        ],
        weight_decay=1e-4,
    )
    estimated_steps_per_epoch = max(
        math.ceil(len(sampler) / args.batch_trajs),
        math.ceil(
            sum(train_dataset.traj_T(i) for i in range(len(train_dataset)))
            / args.max_frames_per_batch
        ),
    )
    total_steps = args.epochs * estimated_steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: learning_rate_scale(
            step, args.warmup_steps, total_steps
        ),
    )

    start_epoch = 0
    start_group_index = 0
    resumed_epoch_state: dict[str, Any] | None = None
    resumed_groups_sha256: str | None = None
    global_step = 0
    best_iou = -float("inf")
    history: list[dict[str, Any]] = []
    resume_path = args.out_dir / "resume.pt"
    if args.auto_resume and resume_path.exists():
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        refiner.load_state_dict(resume["refiner"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        global_step = int(resume["global_step"])
        best_iou = float(resume["best_iou"])
        history = list(resume["history"])
        if resume.get("schema_version") == RESUME_SCHEMA_VERSION:
            validate_resume_compatibility(resume["args"], args)
            start_epoch = int(resume["epoch"])
            start_group_index = int(resume["next_group_index"])
            resumed_epoch_state = dict(resume["epoch_state"])
            resumed_groups_sha256 = resume.get("groups_sha256")
            restore_rng_state(resume["rng_state"], args.device)
        else:
            # Backward-compatible with the original epoch-boundary checkpoint.
            start_epoch = int(resume["epoch"]) + 1
        print(
            f"[resume] epoch={start_epoch}/{args.epochs} "
            f"next_group={start_group_index} step={global_step} "
            f"best_iou={best_iou:.4f} schema="
            f"{resume.get('schema_version', 'legacy_epoch_boundary')}",
            flush=True,
        )
    else:
        print(f"[build] v1 warm start: {warm_start}", flush=True)
        baseline = evaluate(
            refiner,
            val_dataset,
            args.device,
            args.batch_trajs,
            args.max_frames_per_batch,
        )
        atomic_json(args.out_dir / "baseline_eval.json", baseline)
        print(
            "[baseline] "
            + json.dumps(baseline["unified_track_c"], sort_keys=True),
            flush=True,
        )

    stop_epoch = args.epochs
    if args.max_epochs_this_run is not None:
        stop_epoch = min(stop_epoch, start_epoch + args.max_epochs_this_run)
    preemption = PreemptionHandler()
    preemption.install()
    enable_parent_death_signal()
    last_checkpoint_monotonic = time.monotonic()

    def save_resume(
        *,
        epoch: int,
        next_group_index: int,
        epoch_state: dict[str, Any],
        group_digest: str | None,
        reason: str,
    ) -> None:
        nonlocal last_checkpoint_monotonic
        atomic_torch_save(
            resume_path,
            {
                "schema_version": RESUME_SCHEMA_VERSION,
                "refiner": refiner.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "next_group_index": next_group_index,
                "epoch_state": epoch_state,
                "groups_sha256": group_digest,
                "global_step": global_step,
                "best_iou": best_iou,
                "history": history,
                "rng_state": capture_rng_state(args.device),
                "args": vars(args),
                "checkpoint_reason": reason,
                "saved_unix_time": time.time(),
            },
        )
        last_checkpoint_monotonic = time.monotonic()
        print(
            f"[checkpoint] reason={reason} epoch={epoch} "
            f"next_group={next_group_index} global_step={global_step}",
            flush=True,
        )

    for epoch in range(start_epoch, stop_epoch):
        refiner.train()
        sampler.set_epoch(epoch)
        groups = build_groups(
            iter(sampler),
            train_dataset,
            args.batch_trajs,
            args.max_frames_per_batch,
        )
        group_digest = groups_sha256(groups)
        if epoch == start_epoch and resumed_groups_sha256 is not None:
            if resumed_groups_sha256 != group_digest:
                raise ValueError(
                    "resume checkpoint group ordering changed: "
                    f"{resumed_groups_sha256} != {group_digest}"
                )
        next_group_index = start_group_index if epoch == start_epoch else 0
        if not (0 <= next_group_index <= len(groups)):
            raise ValueError(
                f"invalid resume group {next_group_index}/{len(groups)}"
            )
        epoch_started = time.time()
        if epoch == start_epoch and resumed_epoch_state is not None:
            loss_sum = float(resumed_epoch_state["loss_sum"])
            succeeded_steps = int(resumed_epoch_state["succeeded_steps"])
            skipped_steps = int(resumed_epoch_state["skipped_steps"])
            elapsed_before_resume = float(
                resumed_epoch_state["elapsed_seconds"]
            )
            last_loss_values = resumed_epoch_state.get("last_loss")
        else:
            loss_sum = 0.0
            succeeded_steps = 0
            skipped_steps = 0
            elapsed_before_resume = 0.0
            last_loss_values = None

        def epoch_state() -> dict[str, Any]:
            return {
                "loss_sum": loss_sum,
                "succeeded_steps": succeeded_steps,
                "skipped_steps": skipped_steps,
                "elapsed_seconds": (
                    elapsed_before_resume + time.time() - epoch_started
                ),
                "last_loss": last_loss_values,
                "group_count": len(groups),
            }

        for step_index in range(next_group_index, len(groups)):
            group = groups[step_index]
            packs = [train_dataset[index] for index in group]
            batch = to_device(collate_v2_trajectories(packs), args.device)
            frame_count = int(batch["hidden"].shape[1])
            frame_mask = (
                torch.rand(frame_count, device=args.device) < args.mask_frame_p
                if args.mask_frame_p
                else None
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=args.device.startswith("cuda"),
            ):
                output = refiner.forward_vectorized(batch, frame_mask=frame_mask)
            prediction = output["reg"][:, :, 0, :].float()
            target, weights = encode_targets_batched(
                refiner.coder,
                batch["gt_center"].float(),
                batch["gt_dims"].float(),
                batch["gt_quat"].float(),
                batch["box2d"].float(),
                batch["K"].float(),
                batch["input_hw"],
            )
            valid = batch["gt_center"][:, 2] > 1e-3
            loss = v2_track_c_loss(
                prediction,
                target,
                weights,
                batch["gt_dims"].float(),
                batch["gt_quat"].float(),
                valid,
                dim_scale=float(refiner.coder.dim_scale),
            )
            total = loss["loss"]
            if not bool(torch.isfinite(total)):
                skipped_steps += 1
                scheduler.step()
                global_step += 1
                print(
                    f"[skip] epoch={epoch} step={step_index} nonfinite_loss",
                    flush=True,
                )
            else:
                total.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    refiner.parameters(), max_norm=5.0
                )
                if not bool(torch.isfinite(grad_norm)):
                    optimizer.zero_grad(set_to_none=True)
                    skipped_steps += 1
                    scheduler.step()
                    global_step += 1
                    print(
                        f"[skip] epoch={epoch} step={step_index} nonfinite_grad",
                        flush=True,
                    )
                else:
                    optimizer.step()
                    scheduler.step()
                    global_step += 1
                    succeeded_steps += 1
                    loss_sum += float(total.detach())
                    last_loss_values = {
                        name: float(loss[name].detach())
                        for name in (
                            "loss_center",
                            "loss_depth",
                            "loss_dims_encoded",
                            "loss_rot_deg",
                        )
                    }
            completed_groups = step_index + 1
            checkpoint_by_step = (
                args.checkpoint_every_steps > 0
                and global_step % args.checkpoint_every_steps == 0
            )
            checkpoint_by_time = (
                args.checkpoint_every_seconds > 0
                and time.monotonic() - last_checkpoint_monotonic
                >= args.checkpoint_every_seconds
            )
            if preemption.requested or checkpoint_by_step or checkpoint_by_time:
                reason = (
                    f"signal_{preemption.signal_number}"
                    if preemption.requested
                    else "periodic"
                )
                save_resume(
                    epoch=epoch,
                    next_group_index=completed_groups,
                    epoch_state=epoch_state(),
                    group_digest=group_digest,
                    reason=reason,
                )
                if preemption.requested:
                    preemption.terminate()
            if (step_index + 1) % 100 == 0 or step_index + 1 == len(groups):
                elapsed = max(
                    elapsed_before_resume + time.time() - epoch_started, 1e-6
                )
                completed = succeeded_steps + skipped_steps
                rate = completed / elapsed
                eta = (len(groups) - step_index - 1) / max(rate, 1e-9)
                print(
                    f"[epoch {epoch}] steps={step_index + 1}/{len(groups)} "
                    f"loss={loss_sum / max(succeeded_steps, 1):.4f} "
                    f"skip={skipped_steps} rate={rate:.2f}/s "
                    f"ETA={eta / 60:.1f}m",
                    flush=True,
                )
        # Protect the complete epoch before potentially long validation. A
        # preemption during validation will rerun validation, never training.
        save_resume(
            epoch=epoch,
            next_group_index=len(groups),
            epoch_state=epoch_state(),
            group_digest=group_digest,
            reason="pre_eval",
        )
        if preemption.requested:
            preemption.terminate()
        if last_loss_values is None:
            raise RuntimeError(f"epoch {epoch} had no finite optimizer steps")
        record: dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": loss_sum / max(succeeded_steps, 1),
            "succeeded_steps": succeeded_steps,
            "skipped_steps": skipped_steps,
            "epoch_seconds": (
                elapsed_before_resume + time.time() - epoch_started
            ),
            "lr": [group["lr"] for group in optimizer.param_groups],
            **last_loss_values,
        }
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            evaluation = evaluate(
                refiner,
                val_dataset,
                args.device,
                args.batch_trajs,
                args.max_frames_per_batch,
            )
            record["eval"] = evaluation
            current_iou = float(evaluation["unified_track_c"]["iou3d"])
            print(
                f"[epoch {epoch}] eval="
                + json.dumps(evaluation["unified_track_c"], sort_keys=True),
                flush=True,
            )
            if current_iou > best_iou:
                best_iou = current_iou
                atomic_torch_save(
                    args.out_dir / "best.pt",
                    {
                        "refiner": refiner.state_dict(),
                        "args": vars(args),
                        "epoch": epoch,
                        "global_step": global_step,
                        "eval": evaluation,
                        "warm_start": warm_start,
                    },
                )
                print(
                    f"[epoch {epoch}] saved best unified_iou={best_iou:.4f}",
                    flush=True,
                )
        history.append(record)
        atomic_json(args.out_dir / "history.json", history)
        save_resume(
            epoch=epoch + 1,
            next_group_index=0,
            epoch_state={
                "loss_sum": 0.0,
                "succeeded_steps": 0,
                "skipped_steps": 0,
                "elapsed_seconds": 0.0,
                "last_loss": None,
                "group_count": None,
            },
            group_digest=None,
            reason="epoch_complete",
        )
        if preemption.requested:
            preemption.terminate()
        start_group_index = 0
        resumed_epoch_state = None
        resumed_groups_sha256 = None

    if stop_epoch == args.epochs:
        atomic_torch_save(
            args.out_dir / "last.pt",
            {
                "refiner": refiner.state_dict(),
                "args": vars(args),
                "epoch": args.epochs - 1,
                "global_step": global_step,
            },
        )
        best_checkpoint = torch.load(
            args.out_dir / "best.pt", map_location="cpu", weights_only=False
        )
        refiner.load_state_dict(best_checkpoint["refiner"], strict=True)
        interventions = {
            name: evaluate(
                refiner,
                val_dataset,
                args.device,
                args.batch_trajs,
                args.max_frames_per_batch,
                variant_intervention=name,
            )
            for name in ("correct", "swapped", "zero")
        }
        atomic_json(args.out_dir / "variant_interventions.json", interventions)
        atomic_json(
            done_path,
            {
                "status": "done",
                "epochs": args.epochs,
                "global_step": global_step,
                "best_iou": best_iou,
                "best_checkpoint": str(args.out_dir / "best.pt"),
                "variant_contract": EXPECTED_VARIANTS,
                "track_c_post_pair_filtering": "none",
                "ca1m_upstream_suitability_filter": "already_applied",
            },
        )
        print(f"[train] done best_iou={best_iou:.4f}", flush=True)
    else:
        print(
            f"[train] bounded run stopped after epoch {stop_epoch - 1}; resumable",
            flush=True,
        )


if __name__ == "__main__":
    main()
