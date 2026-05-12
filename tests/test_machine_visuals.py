"""Tests for machine-native trajectory visualization helpers."""

from __future__ import annotations

import json

import numpy as np
import pytest

from development.render_machine_visuals import (
    comparison_rows,
    collect_event_windows,
    correlation_matrix,
    filter_records,
    group_runs,
    joined_records,
    metric_value,
    recurrence_matrix,
    render_all,
    row_scaled_matrix,
)


def sample_payload() -> dict:
    points = []
    metrics = []
    events = []
    for step in range(1, 9):
        point_id = f"candidate-a|seed=94|perturbed|scale=1.0|step={step}"
        points.append(
            {
                "point_id": point_id,
                "step": step,
                "seed": 94,
                "substrate": "candidate-a",
                "candidate_id": "candidate-a",
                "run_kind": "perturbed",
                "perturb_scale": 1.0,
            }
        )
        metrics.append(
            {
                "point_id": point_id,
                "residual_delta": float(step) / 10.0,
                "fast_state_norm": float(step),
                "slow_state_norm": float(9 - step),
                "message_state_norm": float(step % 3),
                "fast_state_delta": float(step) / 20.0,
                "slow_state_delta": float(8 - step) / 20.0,
                "message_state_delta": float(step % 2) / 10.0,
                "route_metrics": {
                    "release_pressure_mean": float(step),
                    "release_open_mean": 1.0 if step == 4 else 0.0,
                    "release_strength_mean": 0.5 if step == 4 else 0.0,
                    "carrier_residual_norm": float(step) / 8.0,
                    "fast_update_mean": 0.5,
                    "slow_write_mean": 0.4 + float(step) / 100.0,
                    "message_write_mean": 0.3,
                },
            }
        )
        if step == 4:
            events.append(
                {
                    "event_kind": "perturbation",
                    "point_id": point_id,
                    "step": step,
                    "seed": 94,
                    "substrate": "candidate-a",
                    "run_kind": "perturbed",
                    "perturb_scale": 1.0,
                }
            )
    return {
        "metadata": {
            "candidate_metrics": {
                "candidate-a": {
                    "release_duty_cycle": 0.125,
                    "release_effectiveness": 0.75,
                    "phase_transition_score": 0.5,
                    "internal_richness": 0.9,
                    "geometric_coherence": 0.8,
                }
            }
        },
        "points": points,
        "metrics": metrics,
        "events": events,
        "summaries": {"runs": []},
    }


def test_joined_records_and_nested_metric_lookup():
    records = joined_records(sample_payload())

    assert len(records) == 8
    assert metric_value(records[0], "route_metrics.release_pressure_mean") == 1.0
    assert metric_value(records[0], "missing.metric") is None


def test_filter_records_accepts_substrate_or_candidate_and_seed():
    records = joined_records(sample_payload())

    assert len(filter_records(records, "candidate-a", "94")) == 8
    assert filter_records(records, "candidate-a", "95") == []
    assert filter_records(records, "missing", "all") == []


def test_recurrence_matrix_is_symmetric_with_zero_diagonal():
    records = joined_records(sample_payload())
    matrix = recurrence_matrix(records, ["residual_delta", "route_metrics.release_pressure_mean"])

    assert matrix.shape == (8, 8)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 0.0)


def test_correlation_matrix_handles_constant_columns_without_nan():
    matrix = correlation_matrix(np.asarray([[1.0, 2.0], [1.0, 4.0], [1.0, 6.0]]))

    assert matrix.shape == (2, 2)
    assert not np.isnan(matrix).any()
    assert matrix[0, 0] == 0.0
    assert matrix[1, 1] == pytest.approx(1.0)


def test_row_scaled_matrix_preserves_shape_and_scales_each_row():
    records = joined_records(sample_payload())
    matrix = row_scaled_matrix(records, ["residual_delta", "route_metrics.release_pressure_mean"])

    assert matrix.shape == (2, 8)
    assert matrix[0, -1] == pytest.approx(1.0)
    assert matrix[1, -1] == pytest.approx(1.0)


def test_collect_event_windows_uses_perturbation_event(tmp_path):
    payload = sample_payload()
    records = joined_records(payload)
    windows = collect_event_windows(
        records,
        payload["events"],
        ["route_metrics.release_pressure_mean"],
        window=2,
    )

    assert windows["route_metrics.release_pressure_mean"].shape == (1, 5)
    assert windows["route_metrics.release_pressure_mean"][0].tolist() == [2.0, 3.0, 4.0, 5.0, 6.0]


def test_comparison_rows_summarize_release_and_post_event_delta():
    payload = sample_payload()
    records = joined_records(payload)
    rows = comparison_rows(group_runs(records), payload["events"])

    assert len(rows) == 1
    row = rows[0]
    assert row["release_duty_fraction"] == pytest.approx(1.0 / 8.0)
    assert row["release_strength_max"] == pytest.approx(0.5)
    assert row["release_pressure_max"] == pytest.approx(8.0)
    assert row["post_event_residual_delta_max"] == pytest.approx(0.8)


def test_render_all_emits_pngs_and_skip_metadata(tmp_path):
    pytest.importorskip("matplotlib")
    payload_path = tmp_path / "trajectory_3d.json"
    payload_path.write_text(json.dumps(sample_payload()) + "\n", encoding="utf-8")
    args = type(
        "Args",
        (),
        {
            "input": str(payload_path),
            "out_dir": str(tmp_path / "assets"),
            "run_filter": "all",
            "seed": "all",
            "format": "png",
            "dpi": 80,
        },
    )()

    outputs = render_all(args)

    rendered = [path for path in outputs if path.exists()]
    assert rendered
    assert any(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in rendered)
    assert (tmp_path / "assets" / "release_variant_comparison.csv").exists()
    assert all(path.with_suffix(path.suffix + ".json").exists() for path in outputs)
