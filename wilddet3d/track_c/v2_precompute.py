"""Build canonical point-prompt Track C caches for one v2 production unit.

One invocation processes a single sequence/unit.  It consumes the authoritative
target-complete v2 pair shards and the frozen Stage-4 aggregate
``predictions.jsonl``.  Cache publication follows the production contract:

* every tensor file is written with temporary-file + rename;
* ``.done/<variant>__<dataset>__<unit>.json`` is the sole completion marker;
* a preempted unit is recomputed, never inferred complete from partial files;
* every Stage-4 prompt in a frame participates in replay, including
  prompts excluded upstream by CA-1M's suitability policy;
* prompt-independent repeated-image features are deduplicated only within one
  actual forward chunk, because BF16 kernels can change them with batch size;
* replay-vs-raw deltas are retained as diagnostics, not used to filter rows.

The last point is deliberate. Original Stage-4 jobs mixed H100 and B300 kernels
and did not record OOM-bisected effective batch sizes, so exact cross-hardware
replay is not recoverable for every row. The raw v2 box remains the temporal
prior; the newly replayed hidden/depth/2D anchor form a self-consistent frozen
visual observation for the direct-output refiner.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from wilddet3d.track_c.v2_feature_extractor import V2PointFeatureExtractor


VARIANT_TO_ID = {"point_v3": 0, "point_vlm_v1": 1}


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.tmp.", delete=False
    ) as f:
        tmp = Path(f.name)
    try:
        torch.save(value, tmp)
        with tmp.open("rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_json(path: Path, value: Any) -> None:
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


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _quat_to_R(q: list[float]) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        raise ValueError("zero quaternion")
    w, x, y, z = (w / norm, x / norm, y / norm, z / norm)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _box_repr(row: dict[str, Any]) -> torch.Tensor:
    center = np.asarray(row["pred_center_cam"], dtype=np.float32)
    dims = np.asarray(row["pred_dims_canonical_lhw"], dtype=np.float32)
    if not (dims > 0).all():
        raise ValueError(f"non-positive prior dims: {dims.tolist()}")
    R = _quat_to_R(row["pred_quat_wxyz"])
    rot6d = R[:2].reshape(-1)
    return torch.from_numpy(
        np.concatenate([center, np.log(dims), rot6d]).astype(np.float32)
    )


def _quat_abs_delta(a: np.ndarray, b: np.ndarray) -> float:
    return float(min(np.max(np.abs(a - b)), np.max(np.abs(a + b))))


def _qa_prediction(
    raw: dict[str, Any],
    box2d: torch.Tensor,
    box3d: torch.Tensor,
    score: torch.Tensor,
    score_2d: torch.Tensor,
    score_3d: torch.Tensor,
) -> dict[str, float]:
    box2d_np = box2d.numpy().astype(np.float64)
    box3d_np = box3d.numpy().astype(np.float64)
    center_delta = float(
        np.max(np.abs(box3d_np[:3] - np.asarray(raw["pred_center_cam"])))
    )
    dims_delta = float(
        np.max(np.abs(box3d_np[3:6] - np.asarray(raw["pred_dims_wlh"])))
    )
    quat_delta = _quat_abs_delta(
        box3d_np[6:10], np.asarray(raw["pred_quat_wxyz"], dtype=np.float64)
    )
    box2d_delta = float(
        np.max(np.abs(box2d_np - np.asarray(raw["box_2d_xyxy"])))
    )
    score_delta = abs(float(score) - float(raw["score"]))
    score_2d_delta = abs(float(score_2d) - float(raw["score_2d"]))
    score_3d_delta = abs(float(score_3d) - float(raw["score_3d"]))
    deltas = {
        "center_m": center_delta,
        "dims_m": dims_delta,
        "quat_abs": quat_delta,
        "box2d_px": box2d_delta,
        "score": score_delta,
        "score_2d": score_2d_delta,
        "score_3d": score_3d_delta,
    }
    return deltas


RAW_REPLAY_TOLERANCES = {
    "center_m": 2e-3,
    "dims_m": 2e-3,
    "quat_abs": 2e-3,
    "box2d_px": 0.12,
    "score": 2e-3,
    "score_2d": 2e-3,
    "score_3d": 2e-3,
}


def _trajectory_filename(prefix: str, track_id: str) -> str:
    readable = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in track_id)
    readable = readable[:48]
    digest = hashlib.sha1(track_id.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}__{readable}__{digest}.pt"


def _validate_pair_header(rows: list[dict[str, Any]]) -> tuple[str, str, str]:
    if not rows:
        raise ValueError("target pairs unit is empty")
    variants = {str(row["prompt_variant"]) for row in rows}
    datasets = {str(row["dataset"]) for row in rows}
    units = {str(row["sequence"]) for row in rows}
    if len(variants) != 1 or len(datasets) != 1 or len(units) != 1:
        raise ValueError(
            f"unit file mixes variants/datasets/sequences: "
            f"{variants}, {datasets}, {units}"
        )
    variant, dataset, unit = next(iter(variants)), next(iter(datasets)), next(iter(units))
    if variant not in VARIANT_TO_ID:
        raise ValueError(f"unsupported prompt variant {variant!r}")
    if variant == "point_vlm_v1":
        variant_sources = {
            str(row.get("prompt_variant_source")) for row in rows
        }
        if variant_sources != {"operator_declared"}:
            raise ValueError(
                "point_vlm_v1 requires corpus-repaired operator provenance; "
                f"found {sorted(variant_sources)}"
            )
    for index, row in enumerate(rows, 1):
        suitable = row.get("vlm_suitable")
        suitable_source = row.get("vlm_suitable_source")
        if dataset == "ca1m":
            if suitable != "yes":
                raise ValueError(
                    f"row {index}: CA-1M suitability contract failure: "
                    f"{suitable!r}"
                )
        elif (
            suitable is not None
            or suitable_source != "no_v1_labels_for_dataset"
        ):
            raise ValueError(
                f"row {index}: {dataset} suitability bypass contract failure: "
                f"{suitable!r}/{suitable_source!r}"
            )
    required = {
        "gt_center_cam",
        "gt_dims_local_xyz",
        "gt_quat_wxyz",
        "gt_corners_cam",
    }
    for index, row in enumerate(rows, 1):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"pair row {index} is missing GT targets: {sorted(missing)}")
    return variant, dataset, unit


def process_unit(
    pairs_unit: Path,
    prediction_unit_dir: Path,
    output_root: Path,
    extractor: V2PointFeatureExtractor,
    amp_dtype: str,
    max_prompts_per_forward: int,
) -> dict[str, Any]:
    started = time.time()
    if extractor.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(extractor.device)
    pairs = _load_jsonl(pairs_unit)
    variant, dataset, unit = _validate_pair_header(pairs)
    prefix = f"{variant}__{dataset}__{unit}"
    done_path = output_root / ".done" / f"{prefix}.json"
    if done_path.exists():
        with done_path.open() as f:
            return json.load(f)

    pred_path = prediction_unit_dir / "predictions.jsonl"
    raw_rows = _load_jsonl(pred_path)
    raw_by_key = {
        (str(row["track_id"]), int(row["frame_index"])): row
        for row in raw_rows
        if row.get("status") == "ok"
    }
    pair_by_key = {
        (str(row["track_id"]), int(row["frame_index"])): row for row in pairs
    }
    if len(pair_by_key) != len(pairs):
        raise ValueError(f"duplicate pair keys in {pairs_unit}")
    missing_raw = sorted(set(pair_by_key) - set(raw_by_key))
    if missing_raw:
        raise KeyError(f"{len(missing_raw)} pair rows absent from Stage-4 records")

    meta_path = prediction_unit_dir / "meta.json"
    with meta_path.open() as f:
        meta = json.load(f)
    lineage_exception = (
        variant == "point_vlm_v1"
        and dataset == "waymo"
        and str(meta.get("prompt_variant")) == "point_v3"
        and prediction_unit_dir.parent.name == "point_vlm_v1"
    )
    if (
        str(meta.get("dataset")) != dataset
        or (
            str(meta.get("prompt_variant")) != variant
            and not lineage_exception
        )
    ):
        raise ValueError(
            f"metadata mismatch: pairs={variant}/{dataset}, "
            f"stage4={meta.get('prompt_variant')}/{meta.get('dataset')}"
        )
    vggt_dir = Path(meta["vggt_dir"])
    scale = float(meta["scale_value"])

    grouped: dict[int, list[tuple[dict | None, dict]]] = collections.defaultdict(
        list
    )
    # Preserve predictions.jsonl order. CA-1M's target corpus deliberately
    # excludes unsuitable tracks, but those prompts were present in the
    # original same-text Stage-4 batches and therefore remain replay context.
    for raw in raw_rows:
        if raw.get("status") != "ok":
            continue
        key = (str(raw["track_id"]), int(raw["frame_index"]))
        pair = pair_by_key.get(key)
        if pair is not None and str(raw["category"]) != str(pair["category"]):
            raise ValueError(f"category mismatch for {key}")
        # Stage 4 groups and prompts with the manifest text_prompt, which is
        # intentionally not always the dataset category (Waymo pedestrian ->
        # "person"). Replaying with category silently changes both the text
        # embedding and same-label batch context.
        text_prompt = str(raw["text_prompt"])
        if str(raw["prompt_text"]) != f"geometric: {text_prompt}":
            raise ValueError(f"prompt text contract mismatch for {key}")
        grouped[int(raw["vggt_frame_index"])].append((pair, raw))

    frame_cache: dict[int, dict[str, Any]] = {}
    prompt_features: dict[tuple[str, int], dict[str, Any]] = {}
    max_qa = collections.defaultdict(float)
    raw_replay_within_tolerance = 0
    n_forwards = 0
    replay_context_prompts = 0
    n_frame_feature_rechecks = 0
    max_within_forward_feature_delta = collections.defaultdict(float)
    max_across_group_feature_delta = collections.defaultdict(float)
    input_hw: tuple[int, int] | None = None

    for vggt_index, entries in sorted(grouped.items()):
        image_path = vggt_dir / "frames" / f"{vggt_index:06d}.jpg"
        depth_path = vggt_dir / "depth" / f"{vggt_index:06d}.npy"
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
        depth_m = np.load(depth_path).astype(np.float32) * scale
        K = np.asarray(entries[0][1]["intrinsics"], dtype=np.float32)
        for offset in range(0, len(entries), max_prompts_per_forward):
            chunk = entries[offset : offset + max_prompts_per_forward]
            target_count = sum(pair is not None for pair, _ in chunk)
            if target_count == 0:
                continue
            replay_context_prompts += len(chunk) - target_count
            points = [
                [
                    (float(x), float(y), 1)
                    for x, y in raw["positive_points_saved_frame_xy"]
                ]
                for _, raw in chunk
            ]
            labels = [str(raw["text_prompt"]) for _, raw in chunk]
            features = extractor.extract(
                image=image,
                intrinsics=K,
                points_xy_label=points,
                label=labels,
                depth_m=depth_m,
                amp_dtype=amp_dtype,
            )
            n_forwards += 1
            if input_hw is None:
                input_hw = tuple(int(x) for x in features["input_hw"])
            elif input_hw != tuple(features["input_hw"]):
                raise ValueError(
                    f"model input size changed within unit: "
                    f"{input_hw} vs {features['input_hw']}"
                )

            depth_batch = features["depth_latents"]
            ray_batch = features["ray_embeddings"]
            if depth_batch.shape[0] != len(chunk) or ray_batch.shape[0] != len(chunk):
                raise RuntimeError(
                    "forward feature batch does not align with prompt chunk: "
                    f"depth={tuple(depth_batch.shape)} ray={tuple(ray_batch.shape)} "
                    f"prompts={len(chunk)}"
                )
            for name, value in (
                ("depth_latents", depth_batch),
                ("ray", ray_batch),
            ):
                delta = float(
                    (value.float() - value[:1].float()).abs().max()
                )
                max_within_forward_feature_delta[name] = max(
                    max_within_forward_feature_delta[name], delta
                )
                if delta > 2e-3:
                    raise RuntimeError(
                        f"repeated-image {name} differs within one prompt batch: "
                        f"unit={unit} frame={vggt_index} labels={labels!r} "
                        f"chunk={offset // max_prompts_per_forward} delta={delta}"
                    )

            frame_value = {
                # Clone the selected batch view. Without this, torch.save keeps
                # the full prompt-batch backing storage for every frame and
                # silently inflates caches by up to max_prompts_per_forward.
                "depth_latents": depth_batch[0].clone(),
                "ray": ray_batch[0].clone(),
                "K": features["intrinsics"],
                "input_hw": features["input_hw"],
                "vggt_frame_index": vggt_index,
            }
            if vggt_index not in frame_cache:
                frame_cache[vggt_index] = frame_value
            else:
                n_frame_feature_rechecks += 1
                old = frame_cache[vggt_index]
                for name in ("depth_latents", "ray", "K"):
                    delta = float(
                        (old[name].float() - frame_value[name].float()).abs().max()
                    )
                    max_across_group_feature_delta[name] = max(
                        max_across_group_feature_delta[name], delta
                    )
                    if delta > 2e-3:
                        raise RuntimeError(
                            f"shared-image {name} depends on prompt group: "
                            f"unit={unit} frame={vggt_index} labels={labels!r} "
                            f"delta={delta}"
                        )
            feature_slot = vggt_index

            for i, (pair, raw) in enumerate(chunk):
                if pair is None:
                    continue
                qa = _qa_prediction(
                    raw,
                    features["final_box_2d_xyxy"][i],
                    features["final_box_3d"][i],
                    features["final_score"][i],
                    features["final_score_2d"][i],
                    features["final_score_3d"][i],
                )
                for name, value in qa.items():
                    max_qa[name] = max(max_qa[name], value)
                raw_replay_within_tolerance += int(
                    all(
                        qa[name] <= limit
                        for name, limit in RAW_REPLAY_TOLERANCES.items()
                    )
                )
                key = (str(pair["track_id"]), int(pair["frame_index"]))
                prompt_features[key] = {
                    "hidden": features["hidden_states"][:, i, :],
                    "box2d": features["pred_box_2d"][i],
                    "sel_idx": features["sel_idx"][i],
                    "sel_n_positive_inside": features[
                        "sel_n_positive_inside"
                    ][i],
                    "vggt_frame_index": vggt_index,
                    "feature_slot": feature_slot,
                }

    if len(prompt_features) != len(pairs):
        raise RuntimeError(
            f"feature count mismatch: {len(prompt_features)} vs {len(pairs)} pairs"
        )

    by_track: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for pair in pairs:
        by_track[str(pair["track_id"])].append(pair)

    frame_path = output_root / "frames" / f"{prefix}.pt"
    atomic_torch_save(frame_path, frame_cache)
    trajectory_paths = []
    for track_id, track_rows in sorted(by_track.items()):
        track_rows.sort(
            key=lambda row: (
                int(row["source_frame_index"]),
                int(row["frame_index"]),
            )
        )
        keys = [(track_id, int(row["frame_index"])) for row in track_rows]
        timestamps_ns = [row.get("source_timestamp_ns") for row in track_rows]
        if any(value is None for value in timestamps_ns):
            # Only used when a source genuinely has no timestamps.  Preserve
            # irregular source-frame gaps instead of collapsing to row index.
            base_frame = int(track_rows[0]["source_frame_index"])
            timestamps_sec = [
                (int(row["source_frame_index"]) - base_frame) / 30.0
                for row in track_rows
            ]
            timestamp_source = "source_frame_index_assumed_30fps"
        else:
            base_ns = min(int(value) for value in timestamps_ns)
            timestamps_sec = [
                (int(value) - base_ns) / 1e9 for value in timestamps_ns
            ]
            timestamp_source = "source_timestamp_ns"

        trajectory = {
            "schema_version": "v2_track_c_cache_v2",
            "prompt_variant": variant,
            "prompt_variant_id": VARIANT_TO_ID[variant],
            "dataset": dataset,
            "unit_id": unit,
            "video_key": prefix,
            "track_id": track_id,
            "category": str(track_rows[0]["category"]),
            "frame_index": torch.tensor(
                [int(row["frame_index"]) for row in track_rows], dtype=torch.long
            ),
            "source_frame_index": torch.tensor(
                [int(row["source_frame_index"]) for row in track_rows],
                dtype=torch.long,
            ),
            "vggt_frame_index": torch.tensor(
                [prompt_features[key]["vggt_frame_index"] for key in keys],
                dtype=torch.long,
            ),
            "feature_slot": torch.tensor(
                [prompt_features[key]["feature_slot"] for key in keys],
                dtype=torch.long,
            ),
            "ts_sec": torch.tensor(timestamps_sec, dtype=torch.float32),
            "timestamp_source": timestamp_source,
            "measured": torch.ones(len(track_rows), dtype=torch.bool),
            "hidden": torch.stack(
                [prompt_features[key]["hidden"] for key in keys], dim=0
            ),
            "box2d": torch.stack(
                [prompt_features[key]["box2d"] for key in keys], dim=0
            ),
            "sel_idx": torch.stack(
                [prompt_features[key]["sel_idx"] for key in keys]
            ).long(),
            "sel_n_positive_inside": torch.stack(
                [prompt_features[key]["sel_n_positive_inside"] for key in keys]
            ).long(),
            "box_repr": torch.stack([_box_repr(row) for row in track_rows]),
            "gt_center": torch.tensor(
                [row["gt_center_cam"] for row in track_rows], dtype=torch.float32
            ),
            "gt_dims": torch.tensor(
                [row["gt_dims_local_xyz"] for row in track_rows],
                dtype=torch.float32,
            ),
            "gt_quat": torch.tensor(
                [row["gt_quat_wxyz"] for row in track_rows], dtype=torch.float32
            ),
            "iou3d_input": torch.tensor(
                [float(row["iou3d"]) for row in track_rows], dtype=torch.float32
            ),
        }
        traj_path = output_root / "traj" / _trajectory_filename(prefix, track_id)
        atomic_torch_save(traj_path, trajectory)
        trajectory_paths.append(str(traj_path))

    result = {
        "schema_version": "v2_track_c_cache_v2",
        "status": "done",
        "prompt_variant": variant,
        "prompt_variant_id": VARIANT_TO_ID[variant],
        "dataset": dataset,
        "unit_id": unit,
        "pairs": len(pairs),
        "tracks": len(by_track),
        "unique_vggt_frames": len(frame_cache),
        "feature_cache_slots": len(frame_cache),
        "prompt_groups": len(grouped),
        "model_forwards": n_forwards,
        "replay_context_prompts": replay_context_prompts,
        "shared_frame_feature_rechecks": n_frame_feature_rechecks,
        "raw_replay_within_tolerance": raw_replay_within_tolerance,
        "raw_replay_total": len(pairs),
        "raw_replay_tolerances": RAW_REPLAY_TOLERANCES,
        "max_within_forward_feature_delta": dict(
            sorted(max_within_forward_feature_delta.items())
        ),
        "max_across_group_feature_delta": dict(
            sorted(max_across_group_feature_delta.items())
        ),
        "input_hw": list(input_hw) if input_hw else None,
        "max_raw_record_delta_diagnostic": dict(sorted(max_qa.items())),
        "replay_contract": (
            "all raw frame prompts in canonical shared-image multi-label chunks; "
            "raw Stage-4 box is the temporal prior; replay hidden/depth/2D box "
            "is the self-consistent visual observation"
        ),
        "pairs_path": str(pairs_unit),
        "pairs_sha256": sha256_file(pairs_unit),
        "predictions_path": str(pred_path),
        "predictions_sha256": sha256_file(pred_path),
        "stage4_meta_path": str(meta_path),
        "stage4_variant_metadata_exception": (
            "point_vlm_v1 operator-declared pairing repair over immutable "
            "point_v3 constant"
            if lineage_exception
            else None
        ),
        "frame_cache_path": str(frame_path),
        "trajectory_paths": trajectory_paths,
        "cuda_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(extractor.device))
            if extractor.device.type == "cuda"
            else None
        ),
        "cuda_peak_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(extractor.device))
            if extractor.device.type == "cuda"
            else None
        ),
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(done_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_unit", type=Path, required=True)
    parser.add_argument("--prediction_unit_dir", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sam3_checkpoint", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp_dtype", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--max_prompts_per_forward", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_prompts_per_forward < 1:
        raise ValueError("--max_prompts_per_forward must be >= 1")
    extractor = V2PointFeatureExtractor(
        checkpoint=args.checkpoint,
        sam3_checkpoint=args.sam3_checkpoint,
        device=args.device,
    )
    result = process_unit(
        pairs_unit=args.pairs_unit,
        prediction_unit_dir=args.prediction_unit_dir,
        output_root=args.output_root,
        extractor=extractor,
        amp_dtype=args.amp_dtype,
        max_prompts_per_forward=args.max_prompts_per_forward,
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
