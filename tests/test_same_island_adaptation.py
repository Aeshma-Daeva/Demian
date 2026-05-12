"""Tests for same-island Track B adaptation analysis."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from development.analyze_same_island_adaptation import (
    build_rows,
    classify_summary,
    write_table,
)
from development.evolution.scoring import (
    CAUSAL_MODE_GATE_STATE,
    NATIVE_EMERGENCE_RANK_MODE,
    NATIVE_OBJECTIVE_COMBINED_DISCOVERY,
)


def candidate_payload(
    candidate_id: str,
    *,
    ancestors: list[str],
    parents: list[str],
    scalar: float,
    regimes: dict[str, int] | None = None,
) -> dict:
    return {
        "id": candidate_id,
        "rank_mode": NATIVE_EMERGENCE_RANK_MODE,
        "native_objective": NATIVE_OBJECTIVE_COMBINED_DISCOVERY,
        "causal_mode": CAUSAL_MODE_GATE_STATE,
        "metrics": {
            "internal_richness": 0.7 + scalar / 100.0,
            "channel_separation": 0.5,
            "mathematical_curiosity": 0.3,
            "geometric_coherence": 0.6,
            "release_causal_divergence": 0.1,
            "release_gain_zero_release_causal_divergence": 0.1,
            "gain_zero_clean_fraction": 1.0,
            "release_geometric_event": 0.2,
            "phase_transition_score": 0.1,
            "release_duty_cycle": 0.15,
            "regimes": regimes or {"bounded_strange": 24, "surface_fixed_accumulating": 24},
        },
        "genome": {"scalars": {"release_gain": scalar, "state_gain": 1.0}},
        "parent_ids": parents,
        "ancestor_ids": ancestors,
    }


def write_candidate(root: Path, payload: dict) -> Path:
    path = root / "candidates" / f"{payload['id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def summary_payload(candidate_path: Path, *, best_resume: str = "message", dominant: str = "surface_fixed_accumulating") -> dict:
    return {
        "config": {"candidate_path": candidate_path.as_posix()},
        "capsule_result_flags": {
            "best_single_channel_resume": best_resume,
            "single_channel_mean_step_gaps": {
                "message": 0.2,
                "carrier": 0.7,
            },
        },
        "channel_necessity_order": ["message", "carrier", "control"],
        "probe_summary": [
            {"channel": "carrier", "target": "family", "metric": "accuracy", "score": 0.91},
            {"channel": "carrier", "target": "step", "metric": "r2", "score": 1.0},
        ],
        "ablation_summary": [
            {
                "condition": "message_disabled",
                "dominant_regime": dominant,
                "boundedness_mean": 1.0,
            }
        ],
    }


def write_summary(root: Path, archive_id: str, candidate_id: str, payload: dict, *, step: int = 64) -> Path:
    path = root / archive_id / candidate_id / f"perturb_step_{step:03d}" / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_classify_summary_detects_message_capsule_carrier_support_and_surface_accumulator(tmp_path):
    candidate_path = tmp_path / "candidate.json"
    summary = summary_payload(candidate_path)

    classification = classify_summary(summary)

    assert classification["best_resume_channel"] == "message"
    assert classification["channel_necessity_order"] == ["message", "carrier", "control"]
    assert classification["heldout_dominant_regime"] == "surface_fixed_accumulating"
    assert classification["phenotypes"] == [
        "message_capsule",
        "carrier_support",
        "surface_accumulator",
    ]


def test_build_rows_filters_same_island_and_reads_archive_aware_summaries(tmp_path):
    archive = tmp_path / "track_b_rank_landscape_sweep_combined_discovery_2"
    diagnostics = tmp_path / "diagnostics"
    reference = write_candidate(
        archive,
        candidate_payload(
            "gen008_candidate015",
            ancestors=["gen000_candidate005", "gen000_candidate000"],
            parents=["gen007_candidate010"],
            scalar=0.5,
        ),
    )
    same_island = write_candidate(
        archive,
        candidate_payload(
            "gen010_candidate013",
            ancestors=["gen000_candidate005"],
            parents=["gen008_candidate015"],
            scalar=0.6,
        ),
    )
    write_candidate(
        archive,
        candidate_payload(
            "gen003_candidate001",
            ancestors=["other"],
            parents=["other_parent"],
            scalar=4.0,
        ),
    )
    write_summary(
        diagnostics,
        archive.name,
        "gen008_candidate015",
        summary_payload(reference),
    )
    write_summary(
        diagnostics,
        archive.name,
        "gen010_candidate013",
        summary_payload(same_island, dominant="bounded_strange"),
    )

    rows = build_rows(
        archive,
        ancestors={"gen000_candidate005", "gen000_candidate000"},
        reference_candidate_id="gen008_candidate015",
        diagnostics_roots=[diagnostics],
        extra_candidate_paths=[],
    )

    ids = {row["candidate_id"] for row in rows}
    assert ids == {"gen008_candidate015", "gen010_candidate013"}
    reference_row = next(row for row in rows if row["candidate_id"] == "gen008_candidate015")
    relative_row = next(row for row in rows if row["candidate_id"] == "gen010_candidate013")
    assert reference_row["scalar_distance"] == 0.0
    assert relative_row["scalar_distance"] > 0.0
    assert reference_row["confirmation_phenotype"] == "message_capsule|carrier_support|surface_accumulator"
    assert "unstable_strange" in relative_row["confirmation_phenotype"]


def test_write_table_emits_required_columns(tmp_path):
    out = tmp_path / "table.csv"
    row = {
        "archive_id": "archive",
        "candidate_id": "gen000_candidate000",
        "island": "archive",
        "generation": 0,
        "parent_ids": [],
        "ancestor_ids": ["gen000_candidate000"],
        "scalar_distance": 0.0,
        "rank_score": 1.0,
        "release_geometric_event": 0.2,
        "phase_transition_score": 0.1,
        "release_duty_cycle": 0.15,
        "discovery_dominant_regime": "bounded_strange",
        "discovery_regimes": {"bounded_strange": 1},
        "summary_path": "summary.json",
        "confirmation_phenotype": "message_capsule",
        "best_resume_channel": "message",
        "channel_necessity_order": "message,carrier",
        "heldout_dominant_regime": "surface_fixed_accumulating",
        "carrier_family_accuracy": 0.9,
        "carrier_step_score": 1.0,
        "phenotypes_by_step": "{}",
        "adaptation_label": "stable_confirmed_phenotype",
        "candidate_path": "candidate.json",
    }

    write_table(out, [row])

    rows = list(csv.DictReader(out.open()))
    assert rows[0]["candidate_id"] == "gen000_candidate000"
    assert rows[0]["confirmation_phenotype"] == "message_capsule"
    assert rows[0]["release_duty_cycle"] == "0.15"
