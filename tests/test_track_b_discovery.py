"""Tests for Track B discovery campaign scaffolding."""

from __future__ import annotations

import json
from pathlib import Path

from development.evolution.artifacts import (
    validate_track_b_characterization_summary,
    validate_track_b_discovery_summary,
)
from development.evolution.scoring import (
    CAUSAL_MODE_GATE_STATE,
    NATIVE_OBJECTIVE_COMBINED_DISCOVERY,
    NATIVE_OBJECTIVE_GATE_STATE,
    NATIVE_OBJECTIVE_MORPHOLOGY_LOW_DUTY,
    NATIVE_OBJECTIVE_MORPHOLOGY_ONLY,
)
from development.run_track_b_discovery import (
    DISCOVERY_LABELS,
    CONFIRMATION_OBJECTIVE_QUOTAS,
    discovery_matrix,
    matrix_summary,
    rank_landscape_runs,
    selection_summary,
    short_diversity_runs,
    smoke_runs,
)


BASE_METRICS = {
    "internal_richness": 0.5,
    "channel_separation": 0.2,
    "mathematical_curiosity": 0.3,
    "geometric_coherence": 0.4,
    "release_causal_divergence": 0.01,
    "release_gain_zero_release_causal_divergence": 0.01,
    "gain_zero_clean_fraction": 1.0,
    "release_geometric_event": 0.01,
    "phase_transition_score": 0.01,
    "release_duty_cycle": 0.1,
    "regimes": {"surface_fixed_accumulating": 1},
}


def test_track_b_discovery_matrix_has_required_labels_and_short_sweep_shape():
    runs = short_diversity_runs()

    assert "track_b_seed_motif_timing_sweep" in DISCOVERY_LABELS
    assert len(runs) == 12
    assert [run.seed for run in runs] == list(range(2026051301, 2026051313))
    assert {run.perturb_step for run in runs} == {32, 48, 64, 80}
    assert {run.perturb_channels[0] for run in runs} == {"fast", "message", "carrier", "all"}
    assert all(run.population == 16 and run.generations == 12 for run in runs)
    assert all(run.paired_causal is False for run in runs)
    assert {run.perturb_scales for run in runs} == {(0.35,)}


def test_rank_landscape_covers_four_native_objectives_fairly():
    runs = rank_landscape_runs()
    objectives = [run.native_objective for run in runs]

    assert len(runs) == 8
    assert set(objectives) == {
        NATIVE_OBJECTIVE_GATE_STATE,
        NATIVE_OBJECTIVE_MORPHOLOGY_ONLY,
        NATIVE_OBJECTIVE_MORPHOLOGY_LOW_DUTY,
        NATIVE_OBJECTIVE_COMBINED_DISCOVERY,
    }
    assert all(objectives.count(objective) == 2 for objective in set(objectives))
    assert {run.eval_seeds for run in runs} == {(94, 95)}
    assert {run.causal_mode for run in runs} == {CAUSAL_MODE_GATE_STATE}
    assert all(run.paired_causal is False for run in runs)
    assert all(run.generations == 16 for run in runs)


def test_smoke_and_full_original_presets_preserve_expected_compute_shape():
    smoke = smoke_runs()
    full = discovery_matrix("full-original")
    efficient = discovery_matrix("efficient-discovery")

    assert len(smoke) == 4
    assert all(run.population == 8 and run.generations == 1 for run in smoke)
    assert all(run.paired_causal is False for run in smoke)
    assert len(full) == 24
    assert all(run.paired_causal is True for run in full)
    assert all("--paired-causal" in run.command(Path("out")) for run in full)
    assert len(efficient) == 20


def test_discovery_summary_rejects_missing_labels_and_reused_output_ids():
    summary = matrix_summary(discovery_matrix(), Path("data/evolution/track_b"))
    assert validate_track_b_discovery_summary(
        {
            "experiment_label": "track_b_seed_motif_timing_sweep",
            "output_id": "unique",
            "native_objective": NATIVE_OBJECTIVE_GATE_STATE,
            "causal_mode": CAUSAL_MODE_GATE_STATE,
            "generation_count": 20,
            "output_ids": summary["output_ids"],
        }
    ) == []

    errors = validate_track_b_discovery_summary({"output_ids": ["same", "same"]})

    assert any("missing keys" in error for error in errors)
    assert any("reused output_ids" in error for error in errors)


def test_efficient_commands_record_cpu_worker_defaults():
    run = discovery_matrix("efficient-discovery")[0]
    command = run.command(Path("data/evolution/track_b"))

    assert "--workers" in command
    assert command[command.index("--workers") + 1] == "6"
    assert "--torch-threads" in command
    assert command[command.index("--torch-threads") + 1] == "1"
    assert "--device" in command
    assert command[command.index("--device") + 1] == "cpu"
    assert "--paired-causal" not in command


def write_candidate(
    root: Path,
    candidate_id: str,
    *,
    score: float,
    lineage: str,
    scalar: float,
    native_objective: str = NATIVE_OBJECTIVE_MORPHOLOGY_ONLY,
    regimes: dict[str, int] | None = None,
) -> None:
    candidates = root / "candidates"
    candidates.mkdir(parents=True, exist_ok=True)
    metrics = {
        **BASE_METRICS,
        "internal_richness": score,
        "regimes": regimes or {"surface_fixed_accumulating": 1},
    }
    payload = {
        "id": candidate_id,
        "rank_mode": "native_emergence",
        "native_objective": native_objective,
        "causal_mode": CAUSAL_MODE_GATE_STATE,
        "metrics": metrics,
        "genome": {"scalars": {"release_gain": scalar, "state_gain": 1.2}},
        "ancestor_ids": [lineage],
        "parent_ids": [lineage],
    }
    (candidates / f"{candidate_id}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_confirmation_selector_emits_candidate_metadata_and_commands(tmp_path):
    root = tmp_path / "archive"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"perturb_step": 48}), encoding="utf-8")
    write_candidate(root, "gen000_candidate000", score=0.5, lineage="a", scalar=0.1)
    write_candidate(root, "gen000_candidate001", score=0.4, lineage="b", scalar=0.2)

    summary = selection_summary([root])

    assert summary["preset"] == "confirmation"
    assert summary["candidate_count"] == 2
    assert summary["selected_count"] >= 1
    selected = summary["selected"][0]
    assert selected["candidate_path"].endswith(".json")
    assert selected["lineage_key"]
    assert selected["dominant_regime"] == "surface_fixed_accumulating"
    assert selected["discovery_perturb_step"] == 48
    command = selected["confirmation_commands"][0]
    assert "development/gate_state_propagation_characterization.py" in command
    assert command[command.index("--seeds") + 1] == "96,97,98,99,100,101,102"
    assert command[command.index("--perturb-scales") + 1] == "0.2,0.35,0.7"
    assert command[command.index("--perturb-step") + 1] == "48"


def test_confirmation_selector_enforces_objective_and_regime_diversity(tmp_path):
    roots = {}
    objectives = (
        NATIVE_OBJECTIVE_GATE_STATE,
        NATIVE_OBJECTIVE_MORPHOLOGY_LOW_DUTY,
        NATIVE_OBJECTIVE_MORPHOLOGY_ONLY,
        NATIVE_OBJECTIVE_COMBINED_DISCOVERY,
    )
    for objective in objectives:
        root = tmp_path / objective
        root.mkdir()
        (root / "config.json").write_text(json.dumps({"perturb_step": 64}), encoding="utf-8")
        roots[objective] = root

    for idx in range(6):
        write_candidate(
            roots[NATIVE_OBJECTIVE_GATE_STATE],
            f"gate_{idx}",
            score=1.0 - 0.01 * idx,
            lineage=f"gate_{idx}",
            scalar=0.1 + idx,
            native_objective=NATIVE_OBJECTIVE_GATE_STATE,
            regimes={"surface_fixed_accumulating": 1},
        )
    for idx, objective in enumerate((NATIVE_OBJECTIVE_MORPHOLOGY_LOW_DUTY, NATIVE_OBJECTIVE_MORPHOLOGY_ONLY)):
        for offset in range(2):
            write_candidate(
                roots[objective],
                f"{objective}_{offset}",
                score=0.7 - 0.01 * offset,
                lineage=f"{objective}_{offset}",
                scalar=10.0 + idx + offset,
                native_objective=objective,
                regimes={"bounded_strange": 1} if offset == 0 else {"surface_fixed_accumulating": 1},
            )
    for idx in range(3):
        write_candidate(
            roots[NATIVE_OBJECTIVE_COMBINED_DISCOVERY],
            f"combined_{idx}",
            score=0.45 - 0.01 * idx,
            lineage=f"combined_{idx}",
            scalar=20.0 + idx,
            native_objective=NATIVE_OBJECTIVE_COMBINED_DISCOVERY,
            regimes={"bounded_strange": 24, "surface_fixed_accumulating": 24},
        )

    summary = selection_summary(list(roots.values()))
    selected = summary["selected"]
    objectives_selected = [row["native_objective"] for row in selected]

    for objective, quota in CONFIRMATION_OBJECTIVE_QUOTAS.items():
        assert objectives_selected.count(objective) >= quota
    assert sum(1 for row in selected if row["regimes"].get("bounded_strange", 0) > 0) >= 2
    assert sum(1 for row in selected if row["regimes"].get("surface_fixed_accumulating", 0) > 0) >= 2
    combined_ids = {
        row["candidate_id"]
        for row in selected
        if row["native_objective"] == NATIVE_OBJECTIVE_COMBINED_DISCOVERY
    }
    assert combined_ids == {
        "combined_0",
        "combined_1",
        "combined_2",
    }
    assert "objective_and_regime_quotas_plus_ranked_fill" == summary["selection_policy"]


def test_characterization_summary_contract_requires_mechanism_fields():
    valid = {
        "held_out_seeds": [96, 97, 98, 99, 100, 101, 102],
        "perturb_steps": [48, 64],
        "channel_necessity_order": ["message", "carrier", "slow"],
        "gain_zero_cleanliness": {"gain_zero_clean_mean": 1.0},
        "capsule_result_flags": {"full_state_resume_outperforms_surface": True},
    }

    assert validate_track_b_characterization_summary(valid) == []
    assert validate_track_b_characterization_summary({})[0].startswith("missing keys")
