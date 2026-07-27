"""Explicit train/eval disposition policies for Track C trajectories.

Policies operate on cached trajectory records in memory. They never rewrite or
delete the authoritative pairs corpus or feature cache. A category holdout is
removed from both the optimization set and the primary scene-level validation
set, then evaluated as its own named slice so it cannot affect checkpoint
selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from wilddet3d.track_c.v2_dataset import scene_group


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
    annotations: list[dict[str, Any]]
    report: dict[str, Any]


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
    dispositions: dict[str, tuple[str, str]] = {}
    for key, record in all_by_key.items():
        matching = [rule for rule in holdout_rules if rule.matches(record)]
        if len(matching) > 1:
            raise ValueError(
                f"trajectory matches multiple holdouts: {key}: "
                f"{[rule.name for rule in matching]}"
            )
        base_split = "train" if key in train_keys else "val"
        if matching:
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
            }
        )

    report = {
        "schema_version": "v2_track_c_training_policy_v1",
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
        "heldout_eval_slices": {
            name: _summarize(records)
            for name, records in sorted(heldout_eval_slices.items())
        },
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
        annotations=annotations,
        report=report,
    )
