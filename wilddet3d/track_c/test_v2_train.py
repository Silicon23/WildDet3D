import math

import torch

from wilddet3d.track_c.v2_feature_extractor import _select_reference_queries
from wilddet3d.track_c.v2_train import (
    EXPECTED_VARIANTS,
    learning_rate_scale,
    time_metrics,
    validate_universe,
)
from wilddet3d.track_c.v2_refiner import V2TrackCRefiner
from wilddet3d.track_c.v2_dataset import WeightedCoverageSampler


def _record(dataset: str, variant: str, frames: int = 4) -> dict:
    return {
        "dataset": dataset,
        "prompt_variant": variant,
        "box_repr": torch.zeros(frames, 12),
    }


def test_validate_universe_enforces_d12_and_all_sources():
    records = [
        _record(dataset, variant)
        for dataset, variant in EXPECTED_VARIANTS.items()
    ]
    report = validate_universe(records)
    assert set(report) == {"ca1m", "adt", "waymo"}
    assert report["waymo"]["variants"] == ["point_vlm_v1"]

    records[-1]["prompt_variant"] = "point_v3"
    try:
        validate_universe(records)
    except ValueError as error:
        assert "D12" in str(error)
    else:
        raise AssertionError("D12 mismatch was accepted")


def test_time_metrics_are_cadence_aware_and_zero_on_matching_motion():
    ts = torch.tensor([0.0, 0.1, 0.3, 0.6])
    center = torch.stack([2.0 * ts, -ts, torch.ones_like(ts)], dim=-1)
    R = torch.eye(3).repeat(len(ts), 1, 1)
    metrics = time_metrics(center, R, center, R, ts)
    assert metrics["velocity_residual_mps"] < 1e-6
    assert metrics["acceleration_residual_mps2"] < 1e-6
    assert metrics["angular_speed_residual_degps"] < 1e-6
    assert math.isclose(
        metrics["center_speed_mps"], math.sqrt(5.0), rel_tol=1e-5
    )


def test_warmup_cosine_schedule_bounds():
    assert math.isclose(learning_rate_scale(499, 500, 10_000), 1.0)
    assert math.isclose(learning_rate_scale(10_000, 500, 10_000), 0.0)
    values = [
        learning_rate_scale(step, 500, 10_000)
        for step in (0, 100, 499, 2_000, 10_000, 20_000)
    ]
    assert all(0.0 <= value <= 1.0 for value in values)


def test_v2_decode_uses_each_frames_intrinsics():
    model = V2TrackCRefiner(
        reg_residual_from_prior=False,
        use_temporal_modules=False,
    )
    reg = torch.zeros(2, 12)
    reg[:, 6] = 1
    reg[:, 10] = 1
    box2d = torch.tensor(
        [[0.25, 0.25, 0.75, 0.75], [0.25, 0.25, 0.75, 0.75]]
    )
    K = torch.tensor(
        [
            [[400.0, 0, 320.0], [0, 400.0, 240.0], [0, 0, 1]],
            [[800.0, 0, 300.0], [0, 700.0, 200.0], [0, 0, 1]],
        ]
    )
    vectorized = model.decode_layer(reg, box2d, K, (480, 640))
    loop = torch.cat(
        [
            super(V2TrackCRefiner, model).decode_layer(
                reg[index : index + 1],
                box2d[index : index + 1],
                K[index],
                (480, 640),
            )
            for index in range(2)
        ]
    )
    torch.testing.assert_close(vectorized, loop)
    assert not torch.allclose(vectorized[0, :3], vectorized[1, :3])


def test_reference_query_selection_uses_raw_public_box_correspondence():
    captured = {
        "pred_boxes_2d": torch.tensor(
            [
                [
                    [0.05, 0.10, 0.25, 0.40],
                    [0.40, 0.20, 0.80, 0.70],
                    [0.60, 0.50, 0.95, 0.90],
                ],
                [
                    [0.10, 0.10, 0.30, 0.30],
                    [0.45, 0.45, 0.70, 0.75],
                    [0.72, 0.10, 0.92, 0.35],
                ],
            ]
        )
    }
    selected, iou = _select_reference_queries(
        captured,
        reference_boxes_model_xyxy=[
            [82.0, 21.0, 158.0, 69.0],
            [144.0, 10.0, 184.0, 35.0],
        ],
        input_hw=(100, 200),
    )
    torch.testing.assert_close(selected, torch.tensor([1, 2]))
    assert torch.all(iou > 0.90)


def test_weighted_sampler_covers_large_source_across_epoch_horizon():
    class Dataset:
        records = (
            [{"dataset": "ca1m"} for _ in range(20)]
            + [{"dataset": "adt"} for _ in range(4)]
            + [{"dataset": "waymo"} for _ in range(2)]
        )

        def __len__(self):
            return len(self.records)

    dataset = Dataset()
    sampler = WeightedCoverageSampler(
        dataset, {"ca1m": 0.35, "adt": 0.40, "waymo": 0.25}, seed=7
    )
    assert len(sampler) == len(dataset)
    seen = {source: set() for source in ("ca1m", "adt", "waymo")}
    for epoch in range(max(sampler.coverage_epochs.values())):
        sampler.set_epoch(epoch)
        indices = list(sampler)
        assert len(indices) == len(dataset)
        for index in indices:
            seen[dataset.records[index]["dataset"]].add(index)
    assert all(
        len(seen[source])
        == sum(record["dataset"] == source for record in dataset.records)
        for source in seen
    )
