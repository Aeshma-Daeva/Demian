"""Tests for matched gate-state control/null comparisons."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.evolve_v9_5ch_release import default_genome
from development.gate_state_control_nulls import (
    CandidateSpec,
    ControlNullConfig,
    apply_strict_profile_gate,
    calibrated_strict_thresholds,
    fisher_exact_greater,
    generate_control_candidates,
    pass_rate_comparisons,
    positive_candidates_from_replication_summary,
    row_for_summary,
    run_control_null_comparison,
)
from development.gate_state_propagation_characterization import MotifSpec


class SummaryFixture:
    evidence_gate = {
        "gain_zero_clean": True,
        "routes_disabled_divergence_positive": True,
        "gain_zero_divergence_positive": True,
        "full_resume_exact": True,
        "surface_resume_gap_positive": True,
    }
    ablation_summary = [
        {"condition": "routes_disabled", "causal_divergence_mean": 0.2},
        {"condition": "gain_zero", "causal_divergence_mean": 0.2},
        {"condition": "message_disabled", "causal_divergence_mean": 1.2},
        {"condition": "carrier_disabled", "causal_divergence_mean": 1.0},
        {"condition": "slow_disabled", "causal_divergence_mean": 0.8},
        {"condition": "control_disabled", "causal_divergence_mean": 0.6},
    ]
    capsule_summary = [{"resume_kind": "surface_only", "final_l2_gap_mean": 0.5}]
    channel_necessity_order = ["carrier", "message", "slow", "control"]


def write_candidate(path: Path, candidate_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "id": candidate_id,
                "generation": 0,
                "index": 0,
                "rank_mode": "fixture",
                "genome": default_genome(8, rank=2),
                "metrics": {},
                "parent_ids": [],
                "ancestor_ids": [],
            }
        ),
        encoding="utf-8",
    )


def test_positive_candidates_from_replication_summary_extracts_paths(tmp_path):
    candidate_path = tmp_path / "archive" / "candidates" / "gen000_candidate001.json"
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps({"rows": [{"top_candidate_id": "gen000_candidate001", "top_candidate_path": str(candidate_path)}]}),
        encoding="utf-8",
    )

    candidates = positive_candidates_from_replication_summary(summary_path)

    assert candidates == [
        CandidateSpec(
            candidate_id="gen000_candidate001",
            candidate_path=candidate_path,
            family="track_b_positive",
        )
    ]


def test_generate_control_candidates_writes_default_and_random_controls(tmp_path):
    config = ControlNullConfig(
        output_dir=tmp_path / "out",
        hidden_size=8,
        rank=2,
        random_control_count=2,
        random_seed=123,
    )

    candidates = generate_control_candidates(config)

    assert [candidate.family for candidate in candidates] == [
        "default_unselected",
        "random_unselected",
        "random_unselected",
    ]
    for candidate in candidates:
        payload = json.loads(candidate.candidate_path.read_text())
        assert payload["id"] == candidate.candidate_id
        assert "genome" in payload


def test_row_for_summary_requires_message_carrier_top2():
    candidate = CandidateSpec(candidate_id="fixture", candidate_path=Path("fixture.json"), family="track_b_positive")

    row = row_for_summary(candidate, SummaryFixture())

    assert row.mechanism_gate_pass is True
    assert row.message_carrier_top2 is True
    assert math.isclose(row.message_carrier_dominance, 0.4)
    assert row.channel_clamping_order == "carrier,message,slow,control"


def test_calibrated_strict_gate_uses_positive_profile_minima():
    positive = row_for_summary(
        CandidateSpec(candidate_id="positive", candidate_path=Path("positive.json"), family="track_b_positive"),
        SummaryFixture(),
    )
    weak_control = positive.model_copy(
        update={
            "candidate_id": "weak_control",
            "family": "random_unselected",
            "route_divergence_mean": positive.route_divergence_mean - 0.01,
            "gain_zero_divergence_mean": positive.gain_zero_divergence_mean - 0.01,
        }
    )

    thresholds = calibrated_strict_thresholds([positive, weak_control])
    rows = apply_strict_profile_gate([positive, weak_control], thresholds)

    assert thresholds["min_route_divergence"] == positive.route_divergence_mean
    assert rows[0].strict_profile_gate_pass is True
    assert rows[1].strict_profile_gate_pass is False


def test_pass_rate_comparisons_report_enrichment_statistics():
    positive = row_for_summary(
        CandidateSpec(candidate_id="positive", candidate_path=Path("positive.json"), family="track_b_positive"),
        SummaryFixture(),
    )
    control = positive.model_copy(
        update={
            "candidate_id": "control",
            "family": "random_unselected",
            "mechanism_gate_pass": False,
            "strict_profile_gate_pass": False,
        }
    )

    comparisons = pass_rate_comparisons([positive.model_copy(update={"strict_profile_gate_pass": True}), control])

    strict_random = next(
        row
        for row in comparisons
        if row["gate"] == "strict_profile_gate_pass" and row["control_family"] == "random_unselected"
    )
    assert strict_random["track_b_pass"] == 1
    assert strict_random["control_pass"] == 0
    assert strict_random["fisher_exact_greater_p"] == fisher_exact_greater(1, 0, 0, 1)


def test_control_null_smoke_comparison_writes_summary(tmp_path):
    archive = tmp_path / "archive"
    candidate_path = archive / "candidates" / "gen000_candidate000.json"
    write_candidate(candidate_path, "gen000_candidate000")
    replication_summary = tmp_path / "replication_summary.json"
    replication_summary.write_text(
        json.dumps({"rows": [{"top_candidate_id": "gen000_candidate000", "top_candidate_path": str(candidate_path)}]}),
        encoding="utf-8",
    )

    payload = run_control_null_comparison(
        ControlNullConfig(
            replication_summary_path=replication_summary,
            output_dir=tmp_path / "diagnostics",
            seeds=[94],
            perturb_scales=[0.2],
            motifs=[MotifSpec(family="basis", index=0)],
            steps=10,
            perturb_step=4,
            capsule_pause_step=4,
            hidden_size=8,
            rank=2,
            random_control_count=1,
        )
    )

    assert payload["candidate_count"] == 3
    assert "strict_profile_gate" in payload
    assert "pass_rate_comparisons" in payload
    assert (tmp_path / "diagnostics" / "control_null_summary.json").exists()
    assert {row["family"] for row in payload["rows"]} == {
        "track_b_positive",
        "default_unselected",
        "random_unselected",
    }
