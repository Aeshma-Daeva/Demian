"""Tests for paired causal-release diagnostic CSV generation."""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.causal_release_diagnostics import write_archive_audit_csv
from development.evolve_v9_5ch_release import default_genome


def test_archive_audit_csv_tiny_fixture(tmp_path):
    genome = default_genome(8, rank=2)
    genome["scalars"]["release_gain"] = 0.40
    genome["scalars"]["release_threshold"] = -0.40
    candidate = {
        "label": "fixture",
        "candidate_id": "candidate_fixture",
        "genome": genome,
        "config": {
            "hidden_size": 8,
            "steps": 10,
            "perturb_step": 4,
            "rank": 2,
        },
    }
    out = tmp_path / "audit.csv"
    write_archive_audit_csv(
        out,
        [candidate],
        seeds=[94],
        motif_limit=1,
        perturb_scales=[0.25],
        perturb_channels=["fast"],
        perturb_modes=["external"],
        device="cpu",
    )

    rows = list(csv.DictReader(out.open()))
    assert {row["ablation"] for row in rows} == {
        "original",
        "release_routes_disabled",
        "release_gain_zero",
    }
    original = next(row for row in rows if row["ablation"] == "original")
    for field in (
        "release_causal_divergence",
        "release_duty_cycle",
        "phase_transition_score",
        "rank_score",
    ):
        assert field in original
        assert original[field] != ""
