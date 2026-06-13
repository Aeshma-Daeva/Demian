"""Tests for control-mediated capsule maintenance probes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from development.control_capsule_maintenance import (
    ControlCapsuleConfig,
    candidate_paths_from_replication_summary,
    only_channels,
    resume_state_for_kind,
    run_control_capsule_characterization,
    zero_channels,
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


def test_zero_and_only_channel_helpers_preserve_intended_state():
    state = tuple(torch.full((1, 4), float(idx + 1)) for idx in range(5))

    control_zeroed = zero_channels(state, ("control",))
    control_only = only_channels(state, ("control",))

    assert torch.count_nonzero(control_zeroed[2]) == 0
    assert torch.equal(control_zeroed[1], state[1])
    assert torch.equal(control_zeroed[4], state[4])
    assert torch.equal(control_only[2], state[2])
    for idx in (0, 1, 3, 4):
        assert torch.count_nonzero(control_only[idx]) == 0


def test_resume_state_for_kind_applies_control_capsule_arms():
    model = build_model(
        {"scalars": default_genome(8, rank=2)["scalars"], "matrices": default_genome(8, rank=2)["matrices"]},
        config=ControlCapsuleConfig(hidden_size=8).as_characterization_config(Path("archive/candidates/c.json")),
        condition="original",
    )
    state = tuple(torch.full((1, 8), float(idx + 1)) for idx in range(5))

    control_zero = resume_state_for_kind(model, state, "control_zero_at_resume")
    slow_carrier_zero = resume_state_for_kind(model, state, "slow_carrier_zero_control_preserved")
    control_only = resume_state_for_kind(model, state, "control_only")

    assert torch.count_nonzero(control_zero[2]) == 0
    assert torch.equal(control_zero[1], state[1])
    assert torch.count_nonzero(slow_carrier_zero[1]) == 0
    assert torch.count_nonzero(slow_carrier_zero[4]) == 0
    assert torch.equal(slow_carrier_zero[2], state[2])
    assert torch.equal(control_only[2], state[2])
    assert torch.count_nonzero(control_only[1]) == 0


def test_candidate_paths_from_replication_summary_extracts_top_candidates(tmp_path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "rows": [
                    {"top_candidate_path": "archive/candidates/gen001_candidate000.json"},
                    {"top_candidate_path": "archive/candidates/gen002_candidate000.json"},
                ]
            }
        ),
        encoding="utf-8",
    )

    paths = candidate_paths_from_replication_summary(summary_path)

    assert paths == [
        Path("archive/candidates/gen001_candidate000.json"),
        Path("archive/candidates/gen002_candidate000.json"),
    ]


def test_control_capsule_characterization_writes_artifacts(tmp_path):
    candidate_path = tmp_path / "archive" / "candidates" / "gen000_candidate000.json"
    write_candidate(candidate_path)
    config = ControlCapsuleConfig(
        candidate_paths=[candidate_path],
        output_dir=tmp_path / "control_capsule",
        seeds=[94],
        perturb_scales=[0.2],
        motifs=[MotifSpec(family="basis", index=0)],
        hidden_size=8,
        rank=2,
        steps=12,
        perturb_step=4,
        capsule_pause_step=4,
        tail_window=4,
    )

    payload = run_control_capsule_characterization(config)

    rows_path = config.output_dir / "control_capsule_rows.jsonl"
    summary_path = config.output_dir / "control_capsule_summary.json"
    assert rows_path.exists()
    assert summary_path.exists()
    assert payload["row_count"] == 7
    assert {
        row["resume_kind"]
        for row in payload["by_candidate_resume"]
    } == {
        "full_internal_state",
        "surface_only",
        "zero_state",
        "control_zero_at_resume",
        "control_clamped_zero",
        "slow_carrier_zero_control_preserved",
        "control_only",
    }
    assert payload["aggregate"]["control_zero_index_mean"] is not None
    assert json.loads(summary_path.read_text(encoding="utf-8"))["row_count"] == 7


def test_candidate_path_summary_requires_candidates(tmp_path):
    summary_path = tmp_path / "empty.json"
    summary_path.write_text(json.dumps({"rows": []}), encoding="utf-8")

    with pytest.raises(ValueError):
        candidate_paths_from_replication_summary(summary_path)
