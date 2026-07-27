import torch

from wilddet3d.track_c.v2_training_policy import (
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
        "tracks": 2,
        "frames": 6,
        "scenes": 2,
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
