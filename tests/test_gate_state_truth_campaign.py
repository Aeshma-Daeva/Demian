"""Tests for the gate-state truth-finding campaign entrypoint."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.evolve_v9_5ch_release import default_genome
from development.gate_state_propagation_characterization import MotifSpec
from development.gate_state_truth_campaign import (
    CampaignConfig,
    build_campaign_candidates,
    locked_strict_thresholds_from_calibration,
    pseudo_channel_slices,
    run_truth_campaign,
    surgery_pause_schedule,
    swap_state_components,
    windowed_gap_rows,
    zero_state_components,
)


def write_candidate(path: Path, candidate_id: str, hidden_size: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "id": candidate_id,
                "generation": 0,
                "index": 0,
                "rank_mode": "fixture",
                "genome": default_genome(hidden_size, rank=2),
                "metrics": {},
                "parent_ids": [],
                "ancestor_ids": [],
            }
        ),
        encoding="utf-8",
    )


def write_replication_summary(path: Path, candidate_paths: list[Path]) -> None:
    path.write_text(
        json.dumps(
            {
                "rows": [
                    {"top_candidate_id": candidate_path.stem, "top_candidate_path": str(candidate_path)}
                    for candidate_path in candidate_paths
                ]
            }
        ),
        encoding="utf-8",
    )


def test_pseudo_channel_slices_cover_hidden_size_without_overlap():
    slices = pseudo_channel_slices(32)

    assert set(slices) == {"fast", "slow", "control", "message", "carrier"}
    assert slices["fast"][0] == 0
    assert slices["carrier"][1] == 32
    assert sum(end - start for start, end in slices.values()) == 32
    assert [slices[name][1] for name in slices][:-1] == [slices[name][0] for name in list(slices)[1:]]


def test_state_surgery_swaps_and_zeros_only_requested_components():
    left = tuple(torch.full((1, 3), float(idx + 1)) for idx in range(5))
    right = tuple(torch.full((1, 3), float(10 + idx)) for idx in range(5))

    swapped = swap_state_components(left, right, ["message", "carrier"])
    zeroed = zero_state_components(left, ["slow"])

    assert torch.equal(swapped[3], right[3])
    assert torch.equal(swapped[4], right[4])
    for idx in (0, 1, 2):
        assert torch.equal(swapped[idx], left[idx])
    assert torch.count_nonzero(zeroed[1]) == 0
    for idx in (0, 2, 3, 4):
        assert torch.equal(zeroed[idx], left[idx])


def test_surgery_schedule_and_windows_expose_early_mid_late_gaps(tmp_path):
    config = CampaignConfig(
        replication_summary_path=tmp_path / "unused.json",
        output_dir=tmp_path / "diagnostics",
        steps=12,
        perturb_step=4,
        capsule_pause_step=6,
        surgery_pause_steps=[3, 8],
    )

    assert surgery_pause_schedule(config) == [3, 8]

    rows = windowed_gap_rows([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    by_window = {row["window"]: row for row in rows}
    assert by_window["full"]["mean_step_gap"] == 3.5
    assert by_window["early"]["mean_step_gap"] == 1.5
    assert by_window["mid"]["mean_step_gap"] == 3.5
    assert by_window["late"]["mean_step_gap"] == 5.5


def test_campaign_split_discipline_assigns_calibration_before_test(tmp_path):
    archive = tmp_path / "archive"
    candidate_paths = [archive / "candidates" / f"gen000_candidate00{idx}.json" for idx in range(2)]
    for idx, candidate_path in enumerate(candidate_paths):
        write_candidate(candidate_path, f"gen000_candidate00{idx}")
    replication_summary = tmp_path / "replication_summary.json"
    write_replication_summary(replication_summary, candidate_paths)

    candidates = build_campaign_candidates(
        CampaignConfig(
            replication_summary_path=replication_summary,
            output_dir=tmp_path / "diagnostics",
            hidden_size=8,
            rank=2,
            track_b_count=2,
            random_control_count=3,
            default_jitter_control_count=2,
            calibration_track_b_count=1,
            calibration_random_count=1,
            calibration_default_jitter_count=1,
        )
    )

    assert [row.split for row in candidates if row.family == "track_b_positive"] == ["calibration", "test"]
    assert [row.split for row in candidates if row.family == "random_unselected"] == ["calibration", "test", "test"]
    assert [row.split for row in candidates if row.family in {"default_unselected", "default_jittered"}] == [
        "calibration",
        "test",
    ]


def test_locked_thresholds_require_passing_calibration_track_b():
    calibration_rows = [
        SimpleNamespace(
            candidate_id="gen016_candidate012",
            family="track_b_positive",
            mechanism_gate_pass=False,
            gain_zero_clean=True,
            routes_disabled_divergence_positive=True,
            gain_zero_divergence_positive=False,
            full_resume_exact=True,
            surface_resume_gap_positive=True,
            message_carrier_top2=True,
        )
    ]

    with pytest.raises(RuntimeError, match="no calibration Track B candidate passed"):
        locked_strict_thresholds_from_calibration(calibration_rows)  # type: ignore[arg-type]


def test_truth_campaign_smoke_writes_locked_artifacts(tmp_path, monkeypatch):
    archive = tmp_path / "archive"
    candidate_path = archive / "candidates" / "gen000_candidate000.json"
    write_candidate(candidate_path, "gen000_candidate000")
    replication_summary = tmp_path / "replication_summary.json"
    write_replication_summary(replication_summary, [candidate_path])
    monkeypatch.setattr(
        "development.gate_state_truth_campaign.locked_strict_thresholds_from_calibration",
        lambda rows: {
            "min_route_divergence": 0.0,
            "min_gain_zero_divergence": 0.0,
            "min_message_carrier_dominance": -1.0,
        },
    )

    payload = run_truth_campaign(
        CampaignConfig(
            replication_summary_path=replication_summary,
            output_dir=tmp_path / "diagnostics",
            seeds=[96, 97],
            perturb_scales=[0.2],
            motifs=[MotifSpec(family="basis", index=0)],
            steps=10,
            perturb_step=4,
            capsule_pause_step=4,
            hidden_size=8,
            rank=2,
            track_b_count=1,
            random_control_count=2,
            default_jitter_control_count=1,
            calibration_track_b_count=1,
            calibration_random_count=1,
            calibration_default_jitter_count=1,
            baseline_substrates=["rnn"],
            baseline_seeds=[96],
        )
    )

    assert payload["locked_thresholds"] == json.loads((tmp_path / "diagnostics" / "locked_thresholds.json").read_text())
    assert (tmp_path / "diagnostics" / "truth_campaign_summary.json").exists()
    assert (tmp_path / "diagnostics" / "candidate_splits.csv").exists()
    assert (tmp_path / "diagnostics" / "baseline_comparison_summary.json").exists()
    assert "held_out_test_summary" in payload
    assert payload["candidate_count"] == 4
