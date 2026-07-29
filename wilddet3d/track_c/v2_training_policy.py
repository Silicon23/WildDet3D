"""Explicit train/eval disposition policies for Track C trajectories.

Policies operate on cached trajectory records in memory. They never rewrite or
delete the authoritative pairs corpus or feature cache. A category holdout is
removed from both the optimization set and the primary scene-level validation
set, then evaluated as its own named slice so it cannot affect checkpoint
selection.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from wilddet3d.track_c.v2_dataset import scene_group


WAYMO_GATE_REFERENCE = "gt_box_eroded"
WAYMO_GATE_MAX_TRACKING_LOSS = 0.15
WAYMO_GATE_MIN_LONGEST_RUN_SEC = 1.0


def _canonical_token(value: Any) -> str:
    return str(value).strip().lower()


def _record_key(record: dict[str, Any]) -> str:
    cache_path = record.get("_cache_path")
    if cache_path:
        return str(cache_path)
    return (
        f"{record.get('dataset')}::{record.get('unit_id')}::"
        f"{record.get('track_id')}"
    )


def _frame_count(record: dict[str, Any]) -> int:
    box_repr = record.get("box_repr")
    if box_repr is None or not hasattr(box_repr, "shape"):
        raise ValueError(f"trajectory has no box_repr shape: {_record_key(record)}")
    frames = int(box_repr.shape[0])
    if frames < 1:
        raise ValueError(f"trajectory is empty: {_record_key(record)}")
    return frames


@dataclass(frozen=True)
class DatasetCategoryHoldout:
    """One exact dataset/category slice reserved exclusively for evaluation."""

    dataset: str
    category: str
    name: str

    def matches(self, record: dict[str, Any]) -> bool:
        return (
            _canonical_token(record.get("dataset")) == self.dataset
            and _canonical_token(record.get("category")) == self.category
        )


@dataclass(frozen=True)
class TrainingPolicyPartition:
    train_records: list[dict[str, Any]]
    primary_val_records: list[dict[str, Any]]
    heldout_eval_slices: dict[str, list[dict[str, Any]]]
    quality_gate_rejects: list[dict[str, Any]]
    annotations: list[dict[str, Any]]
    report: dict[str, Any]


@dataclass(frozen=True)
class WaymoGateIndex:
    rows_by_canonical_id: dict[str, dict[str, Any]]
    summary: dict[str, Any]
    artifact_report: dict[str, Any]

    def row_for_record(self, record: dict[str, Any]) -> dict[str, Any]:
        if _canonical_token(record.get("dataset")) != "waymo":
            raise ValueError("Waymo gate requested for a non-Waymo trajectory")
        canonical_id = (
            f"waymo/{record.get('unit_id')}/{record.get('track_id')}"
        )
        if canonical_id not in self.rows_by_canonical_id:
            raise ValueError(
                f"Waymo gate does not cover trajectory {canonical_id}"
            )
        row = self.rows_by_canonical_id[canonical_id]
        if _canonical_token(row["category"]) != _canonical_token(
            record.get("category")
        ):
            raise ValueError(
                f"Waymo gate category mismatch for {canonical_id}: "
                f"{row['category']} != {record.get('category')}"
            )
        return row


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _derived_waymo_gate_pass(
    row: dict[str, Any], max_tracking_loss: float, min_longest_run_sec: float
) -> bool:
    tracking_loss = row.get("tracking_loss")
    return (
        tracking_loss is not None
        and float(tracking_loss) <= max_tracking_loss
        and float(row["longest_run_sec"]) >= min_longest_run_sec
    )


def load_waymo_gate_index(
    rows_path: Path | str, summary_path: Path | str
) -> WaymoGateIndex:
    """Load and fail-closed validate the final Waymo tracking-quality gate."""
    rows_path = Path(rows_path)
    summary_path = Path(summary_path)
    with summary_path.open() as handle:
        summary = json.load(handle)
    if summary.get("schema_version") != "waymo-gate/1.1":
        raise ValueError(
            f"unexpected Waymo gate schema {summary.get('schema_version')!r}"
        )
    gate = summary.get("gate", {})
    expected_gate = {
        "reference": WAYMO_GATE_REFERENCE,
        "max_tracking_loss": WAYMO_GATE_MAX_TRACKING_LOSS,
        "min_longest_run_sec": WAYMO_GATE_MIN_LONGEST_RUN_SEC,
        "bridge_gaps": False,
    }
    if gate != expected_gate:
        raise ValueError(
            f"Waymo gate contract changed: {gate} != {expected_gate}"
        )

    required = {
        "canonical_id",
        "unit",
        "native_id",
        "category",
        "tracking_loss",
        "longest_run_frames",
        "longest_run_sec",
        "n_segments",
        "n_present",
        "n_covered",
        "gate_pass",
        "reason",
        "threshold_fit_member",
        "human_label",
    }
    rows_by_id: dict[str, dict[str, Any]] = {}
    pass_count = 0
    category_counts: dict[str, dict[str, int]] = {}
    with rows_path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = required - set(row)
            if missing:
                raise ValueError(
                    f"Waymo gate row {line_number} missing {sorted(missing)}"
                )
            canonical_id = str(row["canonical_id"])
            expected_id = (
                f"waymo/{row['unit']}/{row['native_id']}"
            )
            if canonical_id != expected_id:
                raise ValueError(
                    f"Waymo gate identity mismatch: {canonical_id} != "
                    f"{expected_id}"
                )
            if canonical_id in rows_by_id:
                raise ValueError(f"duplicate Waymo gate row {canonical_id}")
            derived_pass = _derived_waymo_gate_pass(
                row,
                WAYMO_GATE_MAX_TRACKING_LOSS,
                WAYMO_GATE_MIN_LONGEST_RUN_SEC,
            )
            if bool(row["gate_pass"]) != derived_pass:
                raise ValueError(
                    f"Waymo gate decision disagrees with raw metrics: "
                    f"{canonical_id}"
                )
            if derived_pass != (row["reason"] == "pass"):
                raise ValueError(
                    f"Waymo gate reason disagrees with decision: {canonical_id}"
                )
            rows_by_id[canonical_id] = row
            category = _canonical_token(row["category"])
            counts = category_counts.setdefault(
                category, {"pass": 0, "total": 0}
            )
            counts["total"] += 1
            counts["pass"] += int(derived_pass)
            pass_count += int(derived_pass)

    if len(rows_by_id) != int(summary["n_tracks"]):
        raise ValueError(
            f"Waymo gate row count mismatch: {len(rows_by_id)} != "
            f"{summary['n_tracks']}"
        )
    if pass_count != int(summary["n_pass"]):
        raise ValueError(
            f"Waymo gate pass count mismatch: {pass_count} != "
            f"{summary['n_pass']}"
        )
    for category, counts in category_counts.items():
        reported = summary["by_category"].get(category)
        if reported is None or any(
            int(reported[key]) != value for key, value in counts.items()
        ):
            raise ValueError(
                f"Waymo gate category count mismatch for {category}: "
                f"{counts} != {reported}"
            )
    provenance = summary.get("provenance", {})
    rows_sha256 = _sha256(rows_path)
    if (
        provenance.get("rows_sha256") != rows_sha256
        or int(provenance.get("n_rows", -1)) != len(rows_by_id)
    ):
        raise ValueError("Waymo gate provenance does not match rows artifact")
    leakage = summary.get("threshold_fit_leakage", {})
    fit_track_ids = leakage.get("fit_track_ids", {})
    observed_fit_track_ids = {
        canonical_id: row["human_label"]
        for canonical_id, row in rows_by_id.items()
        if bool(row["threshold_fit_member"])
    }
    if (
        int(leakage.get("n_fit_tracks", -1)) != len(observed_fit_track_ids)
        or fit_track_ids != observed_fit_track_ids
    ):
        raise ValueError(
            "Waymo gate threshold-fit leakage metadata does not match rows"
        )
    return WaymoGateIndex(
        rows_by_canonical_id=rows_by_id,
        summary=summary,
        artifact_report={
            "rows_path": str(rows_path.resolve()),
            "rows_sha256": rows_sha256,
            "summary_path": str(summary_path.resolve()),
            "summary_sha256": _sha256(summary_path),
            "schema_version": summary["schema_version"],
            "rows": len(rows_by_id),
            "pass": pass_count,
            "gate": expected_gate,
        },
    )


def parse_dataset_category_holdouts(
    values: Iterable[str],
) -> list[DatasetCategoryHoldout]:
    """Parse ``dataset:category`` rules from an explicit training config."""
    rules = []
    for value in values:
        parts = [_canonical_token(part) for part in str(value).split(":")]
        if len(parts) != 2 or not all(parts):
            raise ValueError(
                f"holdout must be dataset:category, received {value!r}"
            )
        dataset, category = parts
        rules.append(
            DatasetCategoryHoldout(
                dataset=dataset,
                category=category,
                name=f"{dataset}_{category}",
            )
        )
    names = [rule.name for rule in rules]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate holdout rules: {names}")
    return rules


def _summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "tracks": len(records),
        "frames": sum(_frame_count(record) for record in records),
        "scenes": len({scene_group(record) for record in records}),
    }


def partition_training_policy(
    all_records: Sequence[dict[str, Any]],
    base_train_records: Sequence[dict[str, Any]],
    base_val_records: Sequence[dict[str, Any]],
    holdout_rules: Sequence[DatasetCategoryHoldout],
    waymo_gate: WaymoGateIndex | None = None,
) -> TrainingPolicyPartition:
    """Apply eval-only category holdouts over a persisted scene-level split.

    A matching record is excluded from optimization regardless of its original
    scene split. It is also removed from primary validation so an intentionally
    unlearnable slice cannot determine the best checkpoint. Every matching
    record remains available under a dedicated held-out evaluation slice.
    """
    all_by_key = {_record_key(record): record for record in all_records}
    if len(all_by_key) != len(all_records):
        raise ValueError("trajectory keys are not unique")
    train_keys = {_record_key(record) for record in base_train_records}
    val_keys = {_record_key(record) for record in base_val_records}
    if train_keys & val_keys:
        raise ValueError("base train/validation split overlaps")
    if train_keys | val_keys != set(all_by_key):
        raise ValueError("base train/validation split does not cover all records")

    heldout_eval_slices = {rule.name: [] for rule in holdout_rules}
    quality_gate_rejects: list[dict[str, Any]] = []
    dispositions: dict[str, tuple[str, str]] = {}
    gate_rows_by_key: dict[str, dict[str, Any]] = {}
    for key, record in all_by_key.items():
        matching = [rule for rule in holdout_rules if rule.matches(record)]
        if len(matching) > 1:
            raise ValueError(
                f"trajectory matches multiple holdouts: {key}: "
                f"{[rule.name for rule in matching]}"
            )
        base_split = "train" if key in train_keys else "val"
        gate_row = None
        if (
            waymo_gate is not None
            and _canonical_token(record.get("dataset")) == "waymo"
        ):
            gate_row = waymo_gate.row_for_record(record)
            gate_rows_by_key[key] = gate_row
        if gate_row is not None and not bool(gate_row["gate_pass"]):
            quality_gate_rejects.append(record)
            disposition = "excluded_quality_gate"
        elif matching:
            name = matching[0].name
            heldout_eval_slices[name].append(record)
            disposition = f"heldout_eval:{name}"
        else:
            disposition = "train" if base_split == "train" else "primary_val"
        dispositions[key] = (base_split, disposition)

    train_records = [
        record
        for record in base_train_records
        if dispositions[_record_key(record)][1] == "train"
    ]
    primary_val_records = [
        record
        for record in base_val_records
        if dispositions[_record_key(record)][1] == "primary_val"
    ]
    for rule in holdout_rules:
        if not heldout_eval_slices[rule.name]:
            raise ValueError(f"holdout slice is empty: {rule.name}")
        if any(rule.matches(record) for record in train_records):
            raise AssertionError(f"holdout leaked into training: {rule.name}")
        if any(rule.matches(record) for record in primary_val_records):
            raise AssertionError(
                f"holdout leaked into primary validation: {rule.name}"
            )

    annotations = []
    for key in sorted(all_by_key):
        record = all_by_key[key]
        base_split, disposition = dispositions[key]
        annotations.append(
            {
                "trajectory_key": key,
                "dataset": _canonical_token(record.get("dataset")),
                "category": _canonical_token(record.get("category")),
                "unit_id": str(record.get("unit_id")),
                "track_id": str(record.get("track_id")),
                "scene_group": scene_group(record),
                "frames": _frame_count(record),
                "base_scene_split": base_split,
                "training_disposition": disposition,
                "quality_gate": (
                    {
                        "name": "waymo_tracking_quality_v1",
                        "canonical_id": gate_rows_by_key[key]["canonical_id"],
                        "gate_pass": bool(gate_rows_by_key[key]["gate_pass"]),
                        "reason": str(gate_rows_by_key[key]["reason"]),
                        "tracking_loss": gate_rows_by_key[key]["tracking_loss"],
                        "longest_run_sec": float(
                            gate_rows_by_key[key]["longest_run_sec"]
                        ),
                        "threshold_fit_member": gate_rows_by_key[key].get(
                            "threshold_fit_member"
                        ),
                    }
                    if key in gate_rows_by_key
                    else None
                ),
            }
        )

    gate_reject_reason_counts: dict[str, int] = {}
    for record in quality_gate_rejects:
        reason = str(gate_rows_by_key[_record_key(record)]["reason"])
        gate_reject_reason_counts[reason] = (
            gate_reject_reason_counts.get(reason, 0) + 1
        )
    matched_gate_ids = {
        row["canonical_id"] for row in gate_rows_by_key.values()
    }
    heldout_reports = {}
    for name, records in sorted(heldout_eval_slices.items()):
        from_base_train = [
            record
            for record in records
            if _record_key(record) in train_keys
        ]
        from_base_val = [
            record
            for record in records
            if _record_key(record) in val_keys
        ]
        heldout_report = {
            "all": _summarize(records),
            "from_base_train_scenes": _summarize(from_base_train),
            "from_base_val_scenes": _summarize(from_base_val),
        }
        if waymo_gate is not None:
            heldout_report["threshold_fit_members"] = {
                "all": sum(
                    bool(
                        gate_rows_by_key.get(
                            _record_key(record), {}
                        ).get("threshold_fit_member")
                    )
                    for record in records
                ),
                "from_base_train_scenes": sum(
                    bool(
                        gate_rows_by_key.get(
                            _record_key(record), {}
                        ).get("threshold_fit_member")
                    )
                    for record in from_base_train
                ),
                "from_base_val_scenes": sum(
                    bool(
                        gate_rows_by_key.get(
                            _record_key(record), {}
                        ).get("threshold_fit_member")
                    )
                    for record in from_base_val
                ),
            }
        heldout_reports[name] = heldout_report
    report = {
        "schema_version": "v2_track_c_training_policy_v2",
        "authoritative_data_mutation": "none",
        "checkpoint_selection_slice": "primary_val",
        "category_holdouts": [
            {
                "name": rule.name,
                "dataset": rule.dataset,
                "category": rule.category,
                "disposition": "heldout_eval_only",
            }
            for rule in holdout_rules
        ],
        "base_train": _summarize(base_train_records),
        "base_val": _summarize(base_val_records),
        "train_after_policy": _summarize(train_records),
        "primary_val_after_policy": _summarize(primary_val_records),
        "quality_gate": (
            {
                "name": "waymo_tracking_quality_v1",
                "scope": "waymo_all_categories_only",
                "artifact": waymo_gate.artifact_report,
                "paired_trajectory_coverage": {
                    "matched": len(matched_gate_ids),
                    "missing": 0,
                    "artifact_rows_not_in_paired_cache": (
                        len(waymo_gate.rows_by_canonical_id)
                        - len(matched_gate_ids)
                    ),
                },
                "passing": _summarize(
                    [
                        record
                        for key, record in all_by_key.items()
                        if key in gate_rows_by_key
                        and bool(gate_rows_by_key[key]["gate_pass"])
                    ]
                ),
                "rejected": _summarize(quality_gate_rejects),
                "rejected_by_reason": dict(
                    sorted(gate_reject_reason_counts.items())
                ),
                "continuity_threshold_note": waymo_gate.summary.get(
                    "gate_provenance"
                ),
                "threshold_fit_leakage": waymo_gate.summary.get(
                    "threshold_fit_leakage"
                ),
                "known_blind_spot": waymo_gate.summary.get("reference_note"),
            }
            if waymo_gate is not None
            else None
        ),
        "heldout_eval_slices": heldout_reports,
        "scene_overlap_note": (
            "category holdouts may share a scene with non-heldout training "
            "tracks; heldout metrics are reported separately and never used "
            "for checkpoint selection"
        ),
    }
    return TrainingPolicyPartition(
        train_records=train_records,
        primary_val_records=primary_val_records,
        heldout_eval_slices=heldout_eval_slices,
        quality_gate_rejects=quality_gate_rejects,
        annotations=annotations,
        report=report,
    )
