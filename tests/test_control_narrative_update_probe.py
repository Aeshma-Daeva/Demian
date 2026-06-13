"""Tests for control narrative update probes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from development.control_narrative_update_probe import (
    ControlNarrativeConfig,
    build_history_events,
    history_structure_metrics,
    order_sensitivity_metrics,
    plastic_state_vector,
    run_control_narrative_update_probe,
    state_snapshot,
)
from development.evolve_v9_5ch_release import default_genome
from development.gate_state_propagation_characterization import MotifSpec, build_model


def write_candidate(path: Path) -> None:
    payload = {
        "id": "gen000_candidate000",
        "genome": default_genome(8, rank=2),
        "metrics": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_history_construction_produces_expected_event_schedules():
    assert [event.model_dump() for event in build_history_events("A", (64, 96))] == [
        {"step": 64, "motif_key": "A"}
    ]
    assert [event.model_dump() for event in build_history_events("B", (64, 96))] == [
        {"step": 64, "motif_key": "B"}
    ]
    assert [event.model_dump() for event in build_history_events("A_then_B", (64, 96))] == [
        {"step": 64, "motif_key": "A"},
        {"step": 96, "motif_key": "B"},
    ]
    assert [event.model_dump() for event in build_history_events("B_then_A", (64, 96))] == [
        {"step": 64, "motif_key": "B"},
        {"step": 96, "motif_key": "A"},
    ]


def test_snapshot_helpers_return_stable_finite_vectors():
    genome = default_genome(8, rank=2)
    config = ControlNarrativeConfig(hidden_size=8, rank=2, steps=8, event_steps=(4, 6)).as_characterization_config(
        Path("archive/candidates/c.json")
    )
    model = build_model(
        {"scalars": genome["scalars"], "matrices": genome["matrices"]},
        config=config,
        condition="original",
    )
    state = model.initial_state(1, torch.device("cpu"))

    snapshot = state_snapshot(model, state)
    plastic = plastic_state_vector(model)

    assert snapshot["control"].numel() == state[2].numel()
    assert snapshot["capsule"].numel() == state[1].numel() + state[3].numel() + state[4].numel()
    assert snapshot["surface"].numel() == model.state_vector(state).numel()
    assert plastic.numel() == 10
    assert all(torch.isfinite(value).all() for value in snapshot.values())
    assert torch.isfinite(plastic).all()


def test_synthetic_repeated_histories_have_positive_separation_margin():
    vectors = {
        "A": [torch.tensor([1.0, 0.0]), torch.tensor([0.95, 0.05])],
        "B": [torch.tensor([0.0, 1.0]), torch.tensor([0.05, 0.95])],
    }

    metrics = history_structure_metrics(vectors)

    assert metrics["within_history_update_cosine"] > 0.99
    assert metrics["cross_history_update_cosine"] < 0.1
    assert metrics["history_separation_margin"] > 0.5
    assert metrics["history_classification_accuracy"] == pytest.approx(1.0)


def test_synthetic_random_histories_fail_structured_update_gate():
    vectors = {
        "A": [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])],
        "B": [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])],
    }

    metrics = history_structure_metrics(vectors)

    assert metrics["history_separation_margin"] <= 0.0
    assert metrics["history_classification_accuracy"] <= 0.5


def test_order_sensitivity_is_low_for_identical_and_high_for_swapped_sequences():
    low = order_sensitivity_metrics(
        {
            "A_then_B": [torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])],
            "B_then_A": [torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])],
            "A_then_A": [torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0])],
        }
    )
    high = order_sensitivity_metrics(
        {
            "A_then_B": [torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])],
            "B_then_A": [torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0])],
            "A_then_A": [torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0])],
        }
    )

    assert low["order_sensitivity"] == pytest.approx(0.0)
    assert high["order_sensitivity"] is not None
    assert high["order_sensitivity"] > 0.5
    assert low["repeat_consistency"] == pytest.approx(0.0)


def test_control_narrative_update_probe_writes_artifacts(tmp_path):
    candidate_path = tmp_path / "archive" / "candidates" / "gen000_candidate000.json"
    write_candidate(candidate_path)
    config = ControlNarrativeConfig(
        candidate_paths=[candidate_path],
        output_dir=tmp_path / "control_narrative",
        seeds=[96],
        perturb_scales=[0.2],
        motif_a=MotifSpec(family="basis", index=0),
        motif_b=MotifSpec(family="gaussian", index=0),
        histories=["none", "A", "B", "A_then_B", "B_then_A", "A_then_A", "B_then_B"],
        conditions=["control_preserved", "control_reset_at_event", "control_clamped_zero"],
        hidden_size=8,
        rank=2,
        steps=12,
        event_steps=(4, 8),
        tail_window=3,
    )

    payload = run_control_narrative_update_probe(config)

    assert (config.output_dir / "control_narrative_rows.jsonl").exists()
    assert (config.output_dir / "control_narrative_summary.csv").exists()
    assert (config.output_dir / "control_narrative_summary.json").exists()
    assert (config.output_dir / "README.md").exists()
    assert payload["row_count"] == 21
    assert json.loads((config.output_dir / "control_narrative_summary.json").read_text(encoding="utf-8"))[
        "experiment"
    ] == "control_narrative_update_probe"
    assert {
        row["condition"]
        for row in payload["by_candidate_condition"]
    } == {"control_preserved", "control_reset_at_event", "control_clamped_zero"}
