"""Tests for control-channel characterization helpers."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.control_characterization import (
    ControlTraceSummary,
    VectorControlTraceRow,
    cosine_similarity,
    summarize_control_trace_rows,
    vector_row_from_step,
)
from development.gate_state_propagation_characterization import CharacterizationConfig, MotifSpec
from development.evolve_v9_5ch_release import default_genome


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_summarize_norm_only_trace_reports_recovery_windows(tmp_path):
    rows = []
    for step in range(1, 9):
        rows.append(
            {
                "candidate_id": "fixture",
                "condition": "original",
                "seed": 94,
                "perturb_scale": 0.2,
                "motif_family": "basis",
                "motif_index": 0,
                "step": step,
                "control_state_norm": 1.0 if step < 4 else 2.0 if step <= 5 else 1.1,
            }
        )
    path = tmp_path / "ablation_rows.jsonl"
    write_jsonl(path, rows)

    summary = summarize_control_trace_rows(path, perturb_steps=[4], steps=8, shock_window=1, tail_window=3)

    assert len(summary) == 1
    row = summary[0]
    assert isinstance(row, ControlTraceSummary)
    assert row.condition == "original"
    assert row.pre_control_norm_mean == pytest.approx(1.0)
    assert row.shock_control_norm_mean == pytest.approx(2.0)
    assert row.tail_control_norm_mean == pytest.approx(1.1)
    assert row.tail_minus_pre_control_norm == pytest.approx(0.1)
    assert row.vector_supported is False


def test_summarize_vector_trace_reports_target_return(tmp_path):
    rows = []
    vectors = {
        1: [1.0, 0.0],
        2: [1.0, 0.0],
        3: [1.0, 0.0],
        4: [0.0, 1.0],
        5: [0.0, 1.0],
        6: [0.95, 0.05],
        7: [0.98, 0.02],
        8: [1.0, 0.0],
    }
    for step, vector in vectors.items():
        rows.append(
            {
                "candidate_id": "fixture",
                "condition": "original",
                "seed": 94,
                "perturb_scale": 0.2,
                "motif_family": "basis",
                "motif_index": 0,
                "step": step,
                "control_state_norm": float(torch.tensor(vector).norm()),
                "control_state_vector": vector,
            }
        )
    path = tmp_path / "vector_rows.jsonl"
    write_jsonl(path, rows)

    summary = summarize_control_trace_rows(path, perturb_steps=[4], steps=8, shock_window=1, tail_window=3)

    assert summary[0].vector_supported is True
    assert summary[0].tail_to_pre_control_distance < 0.05
    assert summary[0].tail_to_pre_control_cosine > 0.99


def test_vector_row_from_step_serializes_all_channel_vectors():
    components = {
        "fast": torch.tensor([1.0, 0.0]),
        "slow": torch.tensor([0.0, 1.0]),
        "control": torch.tensor([0.5, 0.5]),
        "message": torch.tensor([0.25, 0.75]),
        "carrier": torch.tensor([0.75, 0.25]),
    }

    row = vector_row_from_step(
        candidate_id="fixture",
        condition="original",
        seed=94,
        perturb_scale=0.2,
        motif_family="basis",
        motif_index=0,
        step=3,
        components=components,
    )

    assert isinstance(row, VectorControlTraceRow)
    assert row.control_state_vector == [0.5, 0.5]
    assert row.channel_state_vectors["carrier"] == [0.75, 0.25]


def test_cosine_similarity_handles_zero_vectors():
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_control_persistence_smoke_runs_without_error(tmp_path):
    from development.control_characterization import run_control_persistence_probe

    genome = default_genome(8, rank=2)
    config = CharacterizationConfig(
        seeds=[94],
        perturb_scales=[0.2],
        motifs=[MotifSpec(family="basis", index=0), MotifSpec(family="gaussian", index=0)],
        hidden_size=8,
        steps=16,
        perturb_step=4,
        capsule_pause_step=4,
    )

    result = run_control_persistence_probe(
        genome,
        candidate_id="fixture",
        config=config,
        seed=94,
        perturb_scale=0.2,
        perturb_steps=(4, 8),
        motif_specs=(MotifSpec(family="basis", index=0), MotifSpec(family="gaussian", index=0)),
    )

    assert {row["condition"] for row in result["summary"]} == {"control_disabled", "control_preserved", "control_reset"}
    assert result["run_count"] == 3
