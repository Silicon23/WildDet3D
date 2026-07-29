import hashlib
import json

import torch

from wilddet3d.track_c.v2_training_policy import (
    load_waymo_gate_index,
    parse_dataset_category_holdouts,
    partition_training_policy,
)


def _record(
    dataset: str,
    category: str,
    unit: str,
    track: str,
    frames: int = 3,
) -> dict:
    return {
        "_cache_path": f"/cache/{dataset}/{unit}/{track}.pt",
        "dataset": dataset,
        "category": category,
        "unit_id": unit,
        "track_id": track,
        "box_repr": torch.zeros(frames, 12),
    }


def test_waymo_pedestrians_are_eval_only_without_mutating_records():
    waymo_ped_train = _record("waymo", "pedestrian", "segment_a", "ped_a")
    waymo_vehicle_train = _record("waymo", "vehicle", "segment_a", "car_a")
    ca_ped_train = _record("ca1m", "pedestrian", "video_1__clip_1", "ped_ca")
    waymo_ped_val = _record("waymo", "pedestrian", "segment_b", "ped_b")
    waymo_vehicle_val = _record("waymo", "vehicle", "segment_b", "car_b")
    records = [
        waymo_ped_train,
        waymo_vehicle_train,
        ca_ped_train,
        waymo_ped_val,
        waymo_vehicle_val,
    ]
    rules = parse_dataset_category_holdouts(["waymo:pedestrian"])
    partition = partition_training_policy(
        records,
        base_train_records=[
            waymo_ped_train,
            waymo_vehicle_train,
            ca_ped_train,
        ],
        base_val_records=[waymo_ped_val, waymo_vehicle_val],
        holdout_rules=rules,
    )

    assert partition.train_records == [waymo_vehicle_train, ca_ped_train]
    assert partition.primary_val_records == [waymo_vehicle_val]
    assert partition.heldout_eval_slices["waymo_pedestrian"] == [
        waymo_ped_train,
        waymo_ped_val,
    ]
    assert partition.report["authoritative_data_mutation"] == "none"
    assert partition.report["checkpoint_selection_slice"] == "primary_val"
    assert partition.report["heldout_eval_slices"]["waymo_pedestrian"] == {
        "all": {"tracks": 2, "frames": 6, "scenes": 2},
        "from_base_train_scenes": {
            "tracks": 1,
            "frames": 3,
            "scenes": 1,
        },
        "from_base_val_scenes": {
            "tracks": 1,
            "frames": 3,
            "scenes": 1,
        },
    }
    assert all("training_disposition" not in record for record in records)
    annotations = {
        row["track_id"]: row["training_disposition"]
        for row in partition.annotations
    }
    assert annotations == {
        "ped_a": "heldout_eval:waymo_pedestrian",
        "car_a": "train",
        "ped_ca": "train",
        "ped_b": "heldout_eval:waymo_pedestrian",
        "car_b": "primary_val",
    }


def test_holdout_config_is_explicit_and_rejects_duplicates():
    rule = parse_dataset_category_holdouts([" Waymo : Pedestrian "])[0]
    assert (rule.dataset, rule.category, rule.name) == (
        "waymo",
        "pedestrian",
        "waymo_pedestrian",
    )
    for values in (["waymo"], ["waymo:"], ["waymo:pedestrian"] * 2):
        try:
            parse_dataset_category_holdouts(values)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid holdout config accepted: {values}")


def test_base_split_must_cover_the_authoritative_trajectory_universe():
    record = _record("waymo", "vehicle", "segment_a", "car_a")
    try:
        partition_training_policy(
            [record],
            base_train_records=[],
            base_val_records=[],
            holdout_rules=[],
        )
    except ValueError as error:
        assert "cover all records" in str(error)
    else:
        raise AssertionError("incomplete base split was accepted")


def _write_gate(tmp_path, rows):
    rows_path = tmp_path / "gate_waymo.jsonl"
    summary_path = tmp_path / "gate_waymo_summary.json"
    rows_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    rows_sha256 = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    categories = {}
    for row in rows:
        counts = categories.setdefault(
            row["category"], {"pass": 0, "total": 0, "rate": 0.0}
        )
        counts["total"] += 1
        counts["pass"] += int(row["gate_pass"])
    for counts in categories.values():
        counts["rate"] = counts["pass"] / counts["total"]
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "waymo-gate/1.1",
                "gate": {
                    "reference": "gt_box_eroded",
                    "max_tracking_loss": 0.15,
                    "min_longest_run_sec": 1.0,
                    "bridge_gaps": False,
                },
                "gate_provenance": "1.0s is the human-specified floor",
                "threshold_fit_leakage": {
                    "human_reviewed_tracks": 21,
                    "waymo_pedestrians": 13,
                    "n_fit_tracks": 0,
                    "fit_track_ids": {},
                },
                "reference_note": "2D references cannot detect occluders",
                "n_tracks": len(rows),
                "n_pass": sum(row["gate_pass"] for row in rows),
                "n_zero_prompt_tracks": 0,
                "by_category": categories,
                "provenance": {
                    "rows_sha256": rows_sha256,
                    "n_rows": len(rows),
                },
            }
        )
    )
    return rows_path, summary_path


def _gate_row(
    unit,
    track,
    category,
    gate_pass,
    *,
    tracking_loss=None,
    longest_run_sec=0.0,
    reason=None,
):
    if tracking_loss is None and gate_pass:
        tracking_loss = 0.05
    if gate_pass and longest_run_sec == 0:
        longest_run_sec = 2.0
    return {
        "canonical_id": f"waymo/{unit}/{track}",
        "unit": unit,
        "native_id": track,
        "category": category,
        "tracking_loss": tracking_loss,
        "longest_run_frames": int(longest_run_sec * 10),
        "longest_run_sec": longest_run_sec,
        "n_segments": int(longest_run_sec > 0),
        "n_present": 20,
        "n_covered": 10,
        "n_drifted": 0,
        "n_occluded": 0,
        "gate_pass": gate_pass,
        "reason": reason or (
            "pass" if gate_pass else "no prompt points selected"
        ),
        "threshold_fit_member": False,
        "human_label": None,
    }


def test_waymo_gate_and_pedestrian_holdout_apply_in_one_partition(tmp_path):
    vehicle_pass = _record("waymo", "vehicle", "segment_a", "car_pass")
    vehicle_fail = _record("waymo", "vehicle", "segment_a", "car_fail")
    ped_pass_train_scene = _record(
        "waymo", "pedestrian", "segment_a", "ped_pass_a"
    )
    ped_pass_val_scene = _record(
        "waymo", "pedestrian", "segment_b", "ped_pass_b"
    )
    ped_fail = _record("waymo", "pedestrian", "segment_b", "ped_fail")
    ca_record = _record("ca1m", "pedestrian", "video_1", "ca_ped")
    records = [
        vehicle_pass,
        vehicle_fail,
        ped_pass_train_scene,
        ped_pass_val_scene,
        ped_fail,
        ca_record,
    ]
    rows = [
        _gate_row("segment_a", "car_pass", "vehicle", True),
        _gate_row(
            "segment_a",
            "car_fail",
            "vehicle",
            False,
            tracking_loss=0.2,
            longest_run_sec=3.0,
            reason="tracking_loss above threshold",
        ),
        _gate_row("segment_a", "ped_pass_a", "pedestrian", True),
        _gate_row("segment_b", "ped_pass_b", "pedestrian", True),
        _gate_row(
            "segment_b",
            "ped_fail",
            "pedestrian",
            False,
        ),
    ]
    rows_path, summary_path = _write_gate(tmp_path, rows)
    gate = load_waymo_gate_index(rows_path, summary_path)
    partition = partition_training_policy(
        records,
        base_train_records=[
            vehicle_pass,
            vehicle_fail,
            ped_pass_train_scene,
            ca_record,
        ],
        base_val_records=[ped_pass_val_scene, ped_fail],
        holdout_rules=parse_dataset_category_holdouts(
            ["waymo:pedestrian"]
        ),
        waymo_gate=gate,
    )

    assert partition.train_records == [vehicle_pass, ca_record]
    assert partition.primary_val_records == []
    assert partition.heldout_eval_slices["waymo_pedestrian"] == [
        ped_pass_train_scene,
        ped_pass_val_scene,
    ]
    assert partition.quality_gate_rejects == [vehicle_fail, ped_fail]
    report = partition.report
    assert report["quality_gate"]["paired_trajectory_coverage"] == {
        "matched": 5,
        "missing": 0,
        "artifact_rows_not_in_paired_cache": 0,
    }
    assert report["quality_gate"]["passing"]["tracks"] == 3
    assert report["quality_gate"]["rejected"]["tracks"] == 2
    assert report["quality_gate"]["rejected_by_reason"] == {
        "no prompt points selected": 1,
        "tracking_loss above threshold": 1,
    }
    assert report["heldout_eval_slices"]["waymo_pedestrian"][
        "threshold_fit_members"
    ] == {
        "all": 0,
        "from_base_train_scenes": 0,
        "from_base_val_scenes": 0,
    }
    assert report["checkpoint_selection_slice"] == "primary_val"
    annotations = {
        row["track_id"]: row for row in partition.annotations
    }
    assert (
        annotations["ped_fail"]["training_disposition"]
        == "excluded_quality_gate"
    )
    assert (
        annotations["ped_pass_a"]["training_disposition"]
        == "heldout_eval:waymo_pedestrian"
    )
    assert annotations["ca_ped"]["quality_gate"] is None


def test_waymo_gate_fails_closed_on_decision_or_coverage_mismatch(tmp_path):
    row = _gate_row("segment_a", "car_a", "vehicle", True)
    row["gate_pass"] = False
    rows_path, summary_path = _write_gate(tmp_path, [row])
    try:
        load_waymo_gate_index(rows_path, summary_path)
    except ValueError as error:
        assert "raw metrics" in str(error)
    else:
        raise AssertionError("inconsistent gate decision was accepted")

    valid_row = _gate_row("segment_a", "car_a", "vehicle", True)
    rows_path, summary_path = _write_gate(tmp_path, [valid_row])
    gate = load_waymo_gate_index(rows_path, summary_path)
    missing_record = _record("waymo", "vehicle", "segment_b", "car_b")
    try:
        partition_training_policy(
            [missing_record],
            base_train_records=[missing_record],
            base_val_records=[],
            holdout_rules=[],
            waymo_gate=gate,
        )
    except ValueError as error:
        assert "does not cover" in str(error)
    else:
        raise AssertionError("missing Waymo gate coverage was accepted")
