"""Tests for the v9 five-channel evolutionary archive utilities."""

import json
import os
import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.evolve_v9_5ch_release import (
    AdaptiveV9FiveChannel,
    NATIVE_EMERGENCE_RANK_MODE,
    NATIVE_REPRODUCTION_WEIGHTS,
    causal_release_metrics,
    candidate_jobs,
    cuda_lock_enabled,
    default_genome,
    evaluate_genome,
    generation_diagnostics,
    lineage_deltas,
    lineage_stability_score,
    motif_suite,
    mutate_causal_release_template_genome,
    mutate_delayed_eligibility_template_genome,
    mutate_genome,
    normalize_workers_for_devices,
    OBSERVABLE_KEYS,
    parse_args,
    parse_worker_devices,
    population_entry,
    PHASE_TRANSITION_RANK_CAP,
    rank_components,
    repair_genome,
    RELEASE_GEOMETRIC_RANK_CAP,
    SCALAR_GENES,
    release_local_metrics,
    release_timing_metrics,
    rare_release_score,
    resolve_device,
    run_with_motif,
    paired_causal_run,
    reproduce,
    target_dims,
    top_ablation_report,
)
from development.evolution.scoring import (
    CAUSAL_MODE_COMBINED,
    CAUSAL_MODE_GATE_STATE,
    CAUSAL_MODE_ROUTE_RELEASE,
    DYNAMIC_SELECTION_PROBE_RANK_MODE,
    ENGINEERED_TARGET_RANK_MODE,
    NATIVE_OBJECTIVE_COMBINED_DISCOVERY,
    NATIVE_OBJECTIVE_GATE_STATE,
    NATIVE_OBJECTIVE_MORPHOLOGY_LOW_DUTY,
    NATIVE_OBJECTIVE_MORPHOLOGY_ONLY,
    scalar_rank,
)
from development.evolution.artifacts import (
    load_summary,
    v10_summary_digest,
    validate_v10_summary,
)
from development.evolution.current import current_experiment_metadata
from development.evolution.diagnostics import generation_diagnostics as extracted_generation_diagnostics
from development.evolution.lineage import population_entry as extracted_population_entry
from development.evolution.scoring import rank_components as extracted_rank_components
from development.export_v9_5ch_evo_trajectory_3d import export_payload, records_from_run


def test_motif_suite_vectors_are_unit_scaled():
    motifs = motif_suite(hidden_size=12, seed=94)
    assert {motif["family"] for motif in motifs} >= {"basis", "block", "sine", "sparse", "gaussian"}
    for motif in motifs:
        assert motif["vector"].shape == (12,)
        assert torch.isclose(motif["vector"].norm(), torch.tensor(1.0), atol=1e-5)


def test_genome_shapes_survive_mutation():
    genome = default_genome(hidden_size=10, rank=2)
    child = mutate_genome(genome, random.Random(7), hidden_size=10, rank=2, sigma=1.0)
    for target, (out_dim, in_dim) in target_dims(10).items():
        assert len(child["matrices"][target]["u"]) == out_dim
        assert len(child["matrices"][target]["u"][0]) == 2
        assert len(child["matrices"][target]["v"]) == 2
        assert len(child["matrices"][target]["v"][0]) == in_dim
        assert len(child["matrices"][target]["obs"]) == 2


def test_delayed_observables_are_contract_fields_and_old_genomes_repair():
    assert "time_since_perturbation" in OBSERVABLE_KEYS
    assert "perturbation_magnitude" in OBSERVABLE_KEYS
    assert SCALAR_GENES["release_threshold"].high >= 0.85

    old_width = len(OBSERVABLE_KEYS) - 2
    genome = default_genome(hidden_size=10, rank=2)
    for gene in genome["matrices"].values():
        gene["obs"] = [row[:old_width] for row in gene["obs"]]

    repaired = repair_genome(genome, hidden_size=10, rank=2)
    for gene in repaired["matrices"].values():
        assert len(gene["obs"][0]) == len(OBSERVABLE_KEYS)
        assert gene["obs"][0][-2:] == [0.0, 0.0]


def test_causal_release_template_mutation_moves_gate_route_and_observables():
    genome = default_genome(hidden_size=10, rank=2)
    for name in (
        "release_to_fast_scale",
        "release_to_slow_scale",
        "release_to_control_scale",
        "release_to_message_scale",
        "release_to_carrier_scale",
    ):
        genome["scalars"][name] = 0.0
    genome["scalars"]["release_to_fast_scale"] = 1.0

    child = mutate_causal_release_template_genome(
        genome,
        random.Random(11),
        hidden_size=10,
        rank=2,
        sigma=1.0,
    )
    route_values = [
        child["scalars"][f"release_to_{channel}_scale"]
        for channel in ("fast", "slow", "control", "message", "carrier")
    ]
    release_obs = child["matrices"]["release_gate"]["obs"]
    pressure_idx = OBSERVABLE_KEYS.index("release_pressure")
    surface_delta_idx = OBSERVABLE_KEYS.index("surface_delta")

    assert child["scalars"]["release_threshold"] != pytest.approx(genome["scalars"]["release_threshold"])
    assert child["scalars"]["release_temperature"] != pytest.approx(genome["scalars"]["release_temperature"])
    assert child["scalars"]["release_gain"] != pytest.approx(genome["scalars"]["release_gain"])
    assert sum(route_values[1:]) > 0.0
    assert sum(route_values) == pytest.approx(1.0)
    assert all(row[pressure_idx] > 0.0 for row in release_obs)
    assert all(row[surface_delta_idx] > 0.0 for row in release_obs)


def test_delayed_eligibility_template_mutation_shapes_timing_without_zeroing_routes():
    genome = default_genome(hidden_size=10, rank=2)
    genome["scalars"]["release_threshold"] = 0.12
    genome["scalars"]["release_to_fast_scale"] = 0.2
    genome["scalars"]["release_to_slow_scale"] = 0.2
    genome["scalars"]["release_to_control_scale"] = 0.2
    genome["scalars"]["release_to_message_scale"] = 0.2
    genome["scalars"]["release_to_carrier_scale"] = 0.2

    child = mutate_delayed_eligibility_template_genome(
        genome,
        random.Random(17),
        hidden_size=10,
        rank=2,
        sigma=1.0,
    )
    release_obs = child["matrices"]["release_gate"]["obs"]
    timing_idx = OBSERVABLE_KEYS.index("time_since_perturbation")
    magnitude_idx = OBSERVABLE_KEYS.index("perturbation_magnitude")
    pressure_idx = OBSERVABLE_KEYS.index("release_pressure")
    surface_delta_idx = OBSERVABLE_KEYS.index("surface_delta")
    route_values = [
        child["scalars"][f"release_to_{channel}_scale"]
        for channel in ("fast", "slow", "control", "message", "carrier")
    ]

    assert 0.55 <= child["scalars"]["release_threshold"] <= 0.85
    assert child["scalars"]["message_decay"] < 0.98
    assert child["scalars"]["carrier_decay"] > 0.98
    assert sum(route_values) > 0.0
    assert all(row[timing_idx] > 0.0 for row in release_obs)
    assert all(row[magnitude_idx] > 0.0 for row in release_obs)
    assert all(row[pressure_idx] > 0.0 for row in release_obs)
    assert all(row[surface_delta_idx] > 0.0 for row in release_obs)


def test_adaptive_model_low_rank_delta_shapes():
    model = AdaptiveV9FiveChannel(8, default_genome(8, rank=2), rank=2)
    state = model.initial_state(1, torch.device("cpu"))
    assert len(state) == 5
    assert model._delta("message_gate").shape == (8, 8)
    assert model._delta("release_gate").shape == (1, 40)


def test_perturbation_observables_activate_after_recorded_perturbation_and_cap():
    model = AdaptiveV9FiveChannel(8, default_genome(8, rank=2), rank=2)
    state = model.initial_state(1, torch.device("cpu"))
    timing_idx = OBSERVABLE_KEYS.index("time_since_perturbation")
    magnitude_idx = OBSERVABLE_KEYS.index("perturbation_magnitude")

    model._step_index = 3
    before = model._observe(*state)
    assert before[timing_idx].item() == pytest.approx(0.0)
    assert before[magnitude_idx].item() == pytest.approx(0.0)

    model.record_perturbation(step=4, magnitude=0.7)
    model._step_index = 4
    at_perturb = model._observe(*state)
    assert at_perturb[timing_idx].item() == pytest.approx(0.0)
    assert at_perturb[magnitude_idx].item() == pytest.approx(0.0)

    model._step_index = 12
    after = model._observe(*state)
    assert after[timing_idx].item() > 0.0
    assert after[magnitude_idx].item() > 0.0
    assert after[magnitude_idx].item() < 0.7

    model._step_index = 80
    capped = model._observe(*state)
    assert capped[timing_idx].item() == pytest.approx(1.0)


def test_run_with_motif_smoke():
    genome = default_genome(8, rank=2)
    motif = motif_suite(8, seed=94)[0]
    run = run_with_motif(
        genome,
        hidden_size=8,
        steps=8,
        seed=94,
        perturb_step=4,
        perturb_scale=0.3,
        motif=motif,
        rank=2,
        device="cpu",
    )
    assert run["summary"]["regime_class"]
    assert len(run["trajectory"]) == 8
    assert "carrier_signature_tail" in run["metrics"]
    assert "release_duty_cycle" in run["metrics"]
    assert "release_local_causality" in run["metrics"]
    assert "release_local_displacement" in run["metrics"]
    assert "release_geometric_event" in run["metrics"]
    assert "phase_transition_score" in run["metrics"]
    assert "release_timing_score" in run["metrics"]
    assert "release_eligible_fraction" in run["metrics"]


def test_v9_1_channel_perturbations_and_release_targets():
    genome = default_genome(8, rank=2)
    genome["scalars"]["release_to_fast_scale"] = 0.0
    genome["scalars"]["release_to_slow_scale"] = 0.25
    genome["scalars"]["release_to_control_scale"] = 0.25
    genome["scalars"]["release_to_message_scale"] = 0.25
    genome["scalars"]["release_to_carrier_scale"] = 0.25
    motif = motif_suite(8, seed=95)[-1]
    run = run_with_motif(
        genome,
        hidden_size=8,
        steps=8,
        seed=95,
        perturb_step=4,
        perturb_scale=0.25,
        motif=motif,
        rank=2,
        device="cpu",
        perturb_channel="all",
        perturb_mode="carrier_pressure",
    )
    assert run["perturb_channel"] == "all"
    assert run["perturb_mode"] == "carrier_pressure"
    assert len(run["trajectory"]) == 8
    route_metrics = run["trajectory"][-1]["route_metrics"]
    assert "release_to_control_norm" in route_metrics
    assert "carrier_state_delta" in run["trajectory"][-1]


def test_release_routes_disabled_keeps_gate_and_zeros_route_biases():
    genome = default_genome(8, rank=2)
    genome["scalars"]["release_gain"] = 0.40
    genome["scalars"]["release_threshold"] = -0.40
    for name in (
        "release_to_fast_scale",
        "release_to_slow_scale",
        "release_to_control_scale",
        "release_to_message_scale",
        "release_to_carrier_scale",
    ):
        genome["scalars"][name] = 0.0
    motif = motif_suite(8, seed=94)[0]
    run = run_with_motif(
        genome,
        hidden_size=8,
        steps=8,
        seed=94,
        perturb_step=4,
        perturb_scale=0.25,
        motif=motif,
        rank=2,
        device="cpu",
        release_routes_disabled=True,
    )
    route_metrics = run["trajectory"][-1]["route_metrics"]
    assert route_metrics["release_strength_mean"] > 0.0
    assert route_metrics["release_to_fast_norm"] == pytest.approx(0.0)
    assert route_metrics["release_to_slow_norm"] == pytest.approx(0.0)
    assert route_metrics["release_to_control_norm"] == pytest.approx(0.0)
    assert route_metrics["release_to_message_norm"] == pytest.approx(0.0)
    assert route_metrics["release_to_carrier_norm"] == pytest.approx(0.0)


def test_paired_causal_divergence_uses_original_release_anchors():
    genome = default_genome(8, rank=2)
    genome["scalars"]["release_gain"] = 0.40
    genome["scalars"]["release_threshold"] = -0.40
    genome["scalars"]["release_to_fast_scale"] = 0.2
    genome["scalars"]["release_to_slow_scale"] = 0.2
    genome["scalars"]["release_to_control_scale"] = 0.2
    genome["scalars"]["release_to_message_scale"] = 0.2
    genome["scalars"]["release_to_carrier_scale"] = 0.2
    motif = motif_suite(8, seed=94)[0]
    original = run_with_motif(
        genome,
        hidden_size=8,
        steps=12,
        seed=94,
        perturb_step=4,
        perturb_scale=0.25,
        motif=motif,
        rank=2,
        device="cpu",
        capture_channel_states=True,
    )
    identical = run_with_motif(
        genome,
        hidden_size=8,
        steps=12,
        seed=94,
        perturb_step=4,
        perturb_scale=0.25,
        motif=motif,
        rank=2,
        device="cpu",
        capture_channel_states=True,
    )
    disabled = run_with_motif(
        genome,
        hidden_size=8,
        steps=12,
        seed=94,
        perturb_step=4,
        perturb_scale=0.25,
        motif=motif,
        rank=2,
        device="cpu",
        release_routes_disabled=True,
        capture_channel_states=True,
    )
    zero = causal_release_metrics(original, identical, anchor_offset=4)
    causal = causal_release_metrics(original, disabled, anchor_offset=4)
    assert zero["release_causal_anchor_count"] > 0
    assert zero["release_causal_divergence"] == pytest.approx(0.0, abs=1e-8)
    assert causal["release_causal_anchor_count"] == zero["release_causal_anchor_count"]
    assert causal["release_causal_divergence"] > 0.0
    for channel in ("fast", "slow", "control", "message", "carrier"):
        assert f"release_causal_divergence_{channel}" in causal


def test_paired_causal_run_records_three_variants_without_json_state_exports():
    genome = default_genome(8, rank=2)
    genome["scalars"]["release_gain"] = 0.40
    genome["scalars"]["release_threshold"] = -0.40
    motif = motif_suite(8, seed=94)[0]
    paired = paired_causal_run(
        genome,
        hidden_size=8,
        steps=10,
        seed=94,
        perturb_step=4,
        perturb_scale=0.25,
        motif=motif,
        rank=2,
        device="cpu",
    )
    assert {run["ablation"] for run in paired["runs"]} == {
        "original",
        "release_routes_disabled",
        "release_gain_zero",
    }
    original = next(run for run in paired["runs"] if run["ablation"] == "original")
    assert "release_causal_divergence" in original["metrics"]
    assert original["metrics"]["gain_zero_clean"] is True
    assert original["metrics"]["gain_zero_max_release_strength"] == pytest.approx(0.0)
    assert original["metrics"]["gain_zero_max_route_norm"] == pytest.approx(0.0)
    assert original["metrics"]["gain_zero_own_release_anchor_count"] == 0
    assert "_channel_states" not in original["trajectory"][0]


def test_evaluate_genome_smoke():
    result = evaluate_genome(
        default_genome(8, rank=2),
        hidden_size=8,
        steps=6,
        seeds=[94],
        perturb_step=3,
        perturb_scales=[0.25],
        rank=2,
        device="cpu",
    )
    assert "internal_richness" in result["aggregate"]
    assert "channel_separation" in result["aggregate"]
    assert "release_geometric_event" in result["aggregate"]
    assert "phase_transition_score" in result["aggregate"]
    assert "release_timing_score" in result["aggregate"]
    assert "mathematical_curiosity" in result["aggregate"]
    assert result["runs"]


def test_paired_causal_evaluate_genome_and_top_ablation_report():
    result = evaluate_genome(
        default_genome(8, rank=2),
        hidden_size=8,
        steps=10,
        seeds=[94],
        perturb_step=4,
        perturb_scales=[0.25],
        rank=2,
        device="cpu",
        paired_causal=True,
    )
    row = {
        "id": "candidate",
        "metrics": result["aggregate"],
        "paired_runs": result["paired_runs"],
    }
    report = top_ablation_report([row], generation=5, top_n=1)
    ablations = report["candidates"][0]["ablations"]
    assert {"original", "release_routes_disabled", "release_gain_zero"} <= set(ablations)
    assert "release_causal_divergence" in row["metrics"]
    assert "release_gain_zero_release_causal_divergence" in row["metrics"]


def test_rank_components_prioritize_machine_native_geometry():
    row = {
        "metrics": {
            "internal_richness": 0.5,
            "channel_separation": 0.2,
            "release_local_causality": 0.0004,
            "release_geometric_event": 0.012,
            "release_causal_divergence": 0.03,
            "phase_transition_score": 0.05,
            "release_timing_score": 0.2,
            "delayed_release_pressure": 0.08,
            "release_effectiveness": 0.0002,
            "mathematical_curiosity": 0.4,
            "boundedness": 1.0,
            "geometric_coherence": 0.7,
            "regime_bonus": 0.5,
            "release_duty_cycle": 0.1,
        }
    }
    components = rank_components(row)
    assert "diagnostic_release_effectiveness" not in components
    assert "diagnostic_release_timing" not in components
    assert components["release_causal_divergence"] > 0.0
    assert components["phase_transition"] > 0.0
    assert components["timing_bonus"] > 0.0
    assert components["channel_separation"] > 0.0
    assert components["mathematical_curiosity"] > 0.0
    assert components["duty_band_penalty"] == 0.0
    assert "release_event_duty" not in components
    assert "boundedness" not in components


def test_causal_modes_separate_route_release_from_gate_state_propagation():
    base_metrics = {
        "internal_richness": 0.0,
        "channel_separation": 0.0,
        "mathematical_curiosity": 0.0,
        "geometric_coherence": 0.0,
        "regime_bonus": 0.0,
        "release_duty_cycle": 0.10,
        "release_timing_score": 0.5,
        "phase_transition_score": 0.5,
        "release_causal_divergence": 0.30,
        "release_gain_zero_release_causal_divergence": 0.30,
        "gain_zero_clean_fraction": 1.0,
    }

    route = rank_components({"causal_mode": CAUSAL_MODE_ROUTE_RELEASE, "metrics": base_metrics})
    gate = rank_components({"causal_mode": CAUSAL_MODE_GATE_STATE, "metrics": base_metrics})
    combined = rank_components({"causal_mode": CAUSAL_MODE_COMBINED, "metrics": base_metrics})

    assert route["route_release_causal_divergence"] == 0.0
    assert route["gate_state_causal_divergence"] == pytest.approx(0.30)
    assert route["release_causal_divergence"] == 0.0
    assert route["timing_bonus"] == 0.0
    assert gate["release_causal_divergence"] > 0.0
    assert combined["route_release_causal_divergence"] == 0.0
    assert combined["gate_state_causal_divergence"] == pytest.approx(0.30)


def test_route_release_mode_requires_gain_zero_divergence_to_drop_out():
    metrics = {
        "internal_richness": 0.0,
        "channel_separation": 0.0,
        "mathematical_curiosity": 0.0,
        "geometric_coherence": 0.0,
        "regime_bonus": 0.0,
        "release_duty_cycle": 0.10,
        "release_timing_score": 0.5,
        "phase_transition_score": 0.5,
        "release_causal_divergence": 0.30,
        "release_gain_zero_release_causal_divergence": 0.0,
        "gain_zero_clean_fraction": 1.0,
    }

    route = rank_components({"causal_mode": CAUSAL_MODE_ROUTE_RELEASE, "metrics": metrics})
    gate = rank_components({"causal_mode": CAUSAL_MODE_GATE_STATE, "metrics": metrics})

    assert route["route_release_causal_divergence"] == pytest.approx(0.30)
    assert route["release_causal_divergence"] > 0.0
    assert route["timing_bonus"] > 0.0
    assert gate["gate_state_causal_divergence"] == 0.0
    assert gate["release_causal_divergence"] == 0.0


def test_v10_current_metadata_and_extracted_helpers():
    metadata = current_experiment_metadata()
    assert metadata["experiment_id"] == "demian-v1"
    assert metadata["experiment_name"] == "Demian v1"
    assert metadata["predecessor_evidence_id"] == "v10.0-frozen-evolution"
    assert metadata["substrate_baseline"] == "demian_native_v9"
    assert metadata["substrate_scaffold"].startswith("v9 five-channel")
    assert rank_components is extracted_rank_components
    assert population_entry is extracted_population_entry
    assert generation_diagnostics is extracted_generation_diagnostics


def test_v10_summary_contract_digest():
    summary = load_summary("data/evolution/v10_0_frozen_evolution_4island_20260510_summary.json")
    assert validate_v10_summary(summary) == []
    digest = v10_summary_digest(summary)
    assert digest["experiment"] == "v10.0-frozen-evolution"
    assert digest["candidate_count"] == 640
    assert digest["generation_count"] == 20
    assert digest["final_best_id"] == "gen019_candidate001"
    assert digest["final_best_event"] == pytest.approx(0.7896, abs=1e-4)


def test_rank_components_use_causal_release_and_phase_caps():
    base_metrics = {
        "internal_richness": 0.0,
        "channel_separation": 0.0,
        "mathematical_curiosity": 0.0,
        "boundedness": 0.0,
        "geometric_coherence": 0.0,
        "regime_bonus": 0.0,
        "release_duty_cycle": 0.0,
    }
    half_cap = rank_components(
        {
            "metrics": {
                **base_metrics,
                "release_geometric_event": RELEASE_GEOMETRIC_RANK_CAP * 2.0,
                "release_causal_divergence": RELEASE_GEOMETRIC_RANK_CAP / 2.0,
                "phase_transition_score": PHASE_TRANSITION_RANK_CAP / 2.0,
            }
        }
    )
    over_cap = rank_components(
        {
            "metrics": {
                **base_metrics,
                "release_geometric_event": 0.0,
                "release_causal_divergence": RELEASE_GEOMETRIC_RANK_CAP * 2.0,
                "phase_transition_score": PHASE_TRANSITION_RANK_CAP * 2.0,
            }
        }
    )
    assert half_cap["causal_multiplier"] == pytest.approx(1.0)
    assert half_cap["release_causal_divergence"] == pytest.approx(0.8)
    assert half_cap["phase_transition"] == pytest.approx(0.6)
    assert over_cap["release_causal_divergence"] == pytest.approx(1.6)
    assert over_cap["phase_transition"] == pytest.approx(1.2)
    assert over_cap["rank"] >= 0.0


def test_rank_components_gate_phase_by_causality_and_penalize_high_duty():
    base_metrics = {
        "internal_richness": 0.5,
        "channel_separation": 0.2,
        "release_geometric_event": 0.012,
        "release_causal_divergence": 0.03,
        "phase_transition_score": 0.05,
        "mathematical_curiosity": 0.4,
        "boundedness": 1.0,
        "geometric_coherence": 0.7,
        "regime_bonus": 0.5,
    }
    in_band = rank_components({"metrics": {**base_metrics, "release_duty_cycle": 0.10}})
    low_duty = rank_components({"metrics": {**base_metrics, "release_duty_cycle": 0.03}})
    high_duty = rank_components({"metrics": {**base_metrics, "release_duty_cycle": 0.30}})
    no_causal = rank_components({"metrics": {**base_metrics, "release_causal_divergence": 0.0, "release_duty_cycle": 0.10}})
    assert in_band["duty_band_penalty"] == 0.0
    assert low_duty["duty_band_penalty"] == 0.0
    assert low_duty["duty_band_multiplier"] == 0.0
    assert high_duty["duty_band_penalty"] < 0.0
    assert no_causal["release_causal_divergence"] == 0.0
    assert no_causal["phase_transition"] == 0.0
    assert no_causal["timing_bonus"] == 0.0


def test_rank_timing_bonus_requires_causality_and_delayed_concentration():
    base_metrics = {
        "internal_richness": 0.0,
        "channel_separation": 0.0,
        "phase_transition_score": 0.0,
        "mathematical_curiosity": 0.0,
        "geometric_coherence": 0.0,
        "regime_bonus": 0.0,
        "release_duty_cycle": 0.10,
    }
    no_causal = rank_components(
        {
            "metrics": {
                **base_metrics,
                "release_causal_divergence": 0.0,
                "release_timing_score": 0.8,
            }
        }
    )
    causal_no_timing = rank_components(
        {
            "metrics": {
                **base_metrics,
                "release_causal_divergence": RELEASE_GEOMETRIC_RANK_CAP,
                "release_timing_score": 0.0,
            }
        }
    )
    causal_timing = rank_components(
        {
            "metrics": {
                **base_metrics,
                "release_causal_divergence": RELEASE_GEOMETRIC_RANK_CAP,
                "release_timing_score": 0.5,
            }
        }
    )

    assert no_causal["timing_bonus"] == 0.0
    assert causal_no_timing["timing_bonus"] == 0.0
    assert causal_timing["timing_bonus"] == pytest.approx(0.8)


def test_engineered_rank_clamps_negative_score_b():
    row = {
        "metrics": {
            "internal_richness": 0.5,
            "channel_separation": 0.2,
            "mathematical_curiosity": 0.4,
            "geometric_coherence": 0.7,
            "release_causal_divergence": 0.0,
            "phase_transition_score": 0.0,
            "release_timing_score": 0.0,
            "regime_bonus": 0.0,
            "release_duty_cycle": 0.8,
        }
    }
    components = rank_components(
        row
    )
    assert components["score_b"] < 0.0
    assert components["rank"] == 0.0
    assert scalar_rank(row) == 0.0


def test_native_rank_excludes_discipline_terms():
    row = {
        "rank_mode": NATIVE_EMERGENCE_RANK_MODE,
        "metrics": {
            "internal_richness": 0.5,
            "channel_separation": 0.2,
            "release_causal_divergence": 0.03,
            "release_geometric_event": 0.01,
            "phase_transition_score": 0.05,
            "release_timing_score": 1.0,
            "mathematical_curiosity": 0.4,
            "geometric_coherence": 0.7,
            "regime_bonus": 1.0,
            "release_duty_cycle": 0.9,
        },
    }
    components = rank_components(row)
    assert "timing_bonus" not in components
    assert "duty_band_penalty" not in components
    assert "regime_bonus" not in components
    assert components["release_causal_divergence_raw"] == pytest.approx(0.03)


def test_native_objective_variants_rank_expected_fixture_order():
    base = {
        "internal_richness": 0.5,
        "channel_separation": 0.2,
        "mathematical_curiosity": 0.4,
        "geometric_coherence": 0.7,
        "release_causal_divergence": 0.2,
        "release_gain_zero_release_causal_divergence": 0.2,
        "gain_zero_clean_fraction": 1.0,
        "release_geometric_event": 0.1,
        "phase_transition_score": 0.1,
        "release_timing_score": 1.0,
        "regime_bonus": 1.0,
        "release_duty_cycle": 0.5,
    }
    morphology = {"rank_mode": NATIVE_EMERGENCE_RANK_MODE, "metrics": base}

    gate_state = scalar_rank({**morphology, "native_objective": NATIVE_OBJECTIVE_GATE_STATE})
    combined = scalar_rank({**morphology, "native_objective": NATIVE_OBJECTIVE_COMBINED_DISCOVERY})
    morphology_only = scalar_rank({**morphology, "native_objective": NATIVE_OBJECTIVE_MORPHOLOGY_ONLY})

    assert gate_state > combined > morphology_only


def test_morphology_only_ignores_causal_divergence_when_present():
    row = {
        "rank_mode": NATIVE_EMERGENCE_RANK_MODE,
        "native_objective": NATIVE_OBJECTIVE_MORPHOLOGY_ONLY,
        "metrics": {
            "internal_richness": 0.4,
            "channel_separation": 0.2,
            "mathematical_curiosity": 0.3,
            "geometric_coherence": 0.6,
            "release_causal_divergence": 100.0,
            "release_gain_zero_release_causal_divergence": 100.0,
            "gain_zero_clean_fraction": 1.0,
            "release_geometric_event": 100.0,
            "phase_transition_score": 100.0,
            "release_duty_cycle": 0.9,
        },
    }

    components = rank_components(row)

    assert set(components) == {
        "internal_richness",
        "channel_separation",
        "mathematical_curiosity",
        "geometric_coherence",
    }
    assert scalar_rank(row) == pytest.approx(1.45)


def test_low_duty_preference_is_capped_below_material_morphology_gain():
    weak_quiet = {
        "rank_mode": NATIVE_EMERGENCE_RANK_MODE,
        "native_objective": NATIVE_OBJECTIVE_MORPHOLOGY_LOW_DUTY,
        "metrics": {
            "internal_richness": 0.30,
            "channel_separation": 0.10,
            "mathematical_curiosity": 0.20,
            "geometric_coherence": 0.30,
            "release_duty_cycle": 0.0,
        },
    }
    better_loud = {
        **weak_quiet,
        "metrics": {
            "internal_richness": 0.40,
            "channel_separation": 0.15,
            "mathematical_curiosity": 0.30,
            "geometric_coherence": 0.40,
            "release_duty_cycle": 1.0,
        },
    }

    assert rank_components(weak_quiet)["low_duty_preference"] <= 0.15
    assert scalar_rank(better_loud) > scalar_rank(weak_quiet)


def test_dynamic_selection_probe_phase_one_uses_raw_causal_without_duty_or_timing():
    row = {
        "rank_mode": DYNAMIC_SELECTION_PROBE_RANK_MODE,
        "generation": 10,
        "metrics": {
            "internal_richness": 0.5,
            "channel_separation": 0.2,
            "mathematical_curiosity": 0.4,
            "geometric_coherence": 0.7,
            "release_causal_divergence": 0.03,
            "release_gain_zero_release_causal_divergence": 0.03,
            "gain_zero_clean_fraction": 1.0,
            "release_geometric_event": 0.2,
            "phase_transition_score": 0.1,
            "release_timing_score": 1.0,
            "release_duty_cycle": 0.9,
        },
    }

    components = rank_components(row)

    assert components["release_causal_divergence_raw"] == pytest.approx(0.045)
    assert components["release_geometric_event"] == pytest.approx(0.06)
    assert components["phase_transition"] == pytest.approx(0.03)
    assert "duty_band_penalty" not in components
    assert "timing_bonus" not in components


def test_dynamic_selection_probe_phase_two_gates_event_phase_and_penalizes_high_duty():
    row = {
        "rank_mode": DYNAMIC_SELECTION_PROBE_RANK_MODE,
        "generation": 25,
        "metrics": {
            "internal_richness": 0.5,
            "channel_separation": 0.2,
            "mathematical_curiosity": 0.4,
            "geometric_coherence": 0.7,
            "release_causal_divergence": 0.005,
            "release_geometric_event": 0.2,
            "phase_transition_score": 0.1,
            "release_timing_score": 1.0,
            "release_duty_cycle": 0.30,
        },
    }

    components = rank_components(row)

    assert components["causal_multiplier"] == pytest.approx(0.5)
    assert components["release_geometric_event"] == pytest.approx(0.03)
    assert components["phase_transition"] == pytest.approx(0.015)
    assert components["duty_band_penalty"] < 0.0
    assert "timing_bonus" not in components


def test_dynamic_selection_probe_phase_three_requires_causal_and_duty_band_for_timing():
    base_metrics = {
        "internal_richness": 0.0,
        "channel_separation": 0.0,
        "mathematical_curiosity": 0.0,
        "geometric_coherence": 0.0,
        "release_geometric_event": 0.0,
        "phase_transition_score": 0.0,
        "release_timing_score": 0.5,
    }
    in_band = {
        "rank_mode": DYNAMIC_SELECTION_PROBE_RANK_MODE,
        "generation": 45,
        "metrics": {
            **base_metrics,
            "release_causal_divergence": 0.01,
            "release_duty_cycle": 0.10,
        },
    }
    loud = {
        **in_band,
        "metrics": {
            **base_metrics,
            "release_causal_divergence": 0.01,
            "release_duty_cycle": 0.30,
        },
    }
    no_causal = {
        **in_band,
        "metrics": {
            **base_metrics,
            "release_causal_divergence": 0.0,
            "release_duty_cycle": 0.10,
        },
    }

    assert rank_components(in_band)["timing_bonus"] == pytest.approx(0.8)
    assert rank_components(loud)["timing_bonus"] == 0.0
    assert rank_components(no_causal)["timing_bonus"] == 0.0


def test_track_b_cli_records_native_objective_causal_mode_and_sweep_fields(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evolve_v9_5ch_release.py",
            "--rank-mode",
            NATIVE_EMERGENCE_RANK_MODE,
            "--native-objective",
            NATIVE_OBJECTIVE_MORPHOLOGY_ONLY,
            "--causal-mode",
            CAUSAL_MODE_GATE_STATE,
            "--generations",
            "20",
            "--perturb-step",
            "48",
            "--perturb-channels",
            "message,carrier",
            "--eval-seeds",
            "94,95",
        ],
    )

    args = parse_args()

    assert args.native_objective == NATIVE_OBJECTIVE_MORPHOLOGY_ONLY
    assert args.causal_mode == CAUSAL_MODE_GATE_STATE
    assert args.generations == 20
    assert args.perturb_step == 48
    assert args.perturb_channels == "message,carrier"


def test_dynamic_selection_probe_preset_records_experiment_shape(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evolve_v9_5ch_release.py", "--dynamic-selection-probe"])

    args = parse_args()

    assert args.population == 16
    assert args.generations == 60
    assert args.eval_seeds == "94,95"
    assert args.rank_mode == DYNAMIC_SELECTION_PROBE_RANK_MODE
    assert args.reproduction_mode == "native"
    assert args.no_default_seed is True
    assert args.no_elitism is True
    assert args.no_random_injection is True
    assert args.device == "cpu"
    assert args.torch_threads == 1


def test_delayed_release_pressure_prefers_sparse_post_perturbation_release():
    trajectory = [
        {
            "step": step,
            "message_signature_cosine": 0.1 if step < 64 else 0.2,
            "carrier_signature_cosine": 0.1 if step < 64 else 0.2,
            "route_metrics": {"release_pressure_mean": 0.2},
        }
        for step in range(1, 129)
    ]
    sparse_release_rows = [trajectory[71], trajectory[73]]
    early_noisy_rows = [
        trajectory[idx - 1] for idx in (12, 20, 32, 70, 91, 100, 108, 116)
    ]

    sparse = release_timing_metrics(trajectory, sparse_release_rows, perturb_step=64)
    noisy = release_timing_metrics(trajectory, early_noisy_rows, perturb_step=64)

    assert sparse["delayed_release_pressure"] > noisy["delayed_release_pressure"]
    assert sparse["release_early_fraction"] == 0.0
    assert noisy["release_early_fraction"] > 0.0


def test_release_local_phase_transition_uses_normalized_substrate_lifts():
    trajectory = []
    surfaces = []
    for step in range(1, 13):
        value = float(step) * 0.01
        if step >= 7:
            value += float(step - 6) * 0.04
        surface = torch.tensor([value, value * value], dtype=torch.float32)
        surfaces.append(surface)
        trajectory.append(
            {
                "step": step,
                "residual_delta": 0.004 if step < 7 else 0.012,
                "message_signature_cosine": 0.2,
                "carrier_signature_cosine": 0.1,
                "route_metrics": {
                    "release_pressure_mean": 0.3,
                    "release_strength_mean": 0.2,
                },
            }
        )
    signature = torch.tensor([1.0, 0.0], dtype=torch.float32)
    sparse_release = [trajectory[6]]
    flooding_release = trajectory[6:]

    sparse = release_local_metrics(
        trajectory, surfaces, signature, sparse_release, window=3
    )
    flooding = release_local_metrics(
        trajectory, surfaces, signature, flooding_release, window=3
    )

    assert sparse["phase_transition_score"] > 0.02
    assert flooding["phase_transition_score"] < sparse["phase_transition_score"]


def test_rare_release_score_prefers_sparse_band():
    assert rare_release_score(0.10) == 1.0
    assert rare_release_score(0.50) == 0.0
    assert rare_release_score(0.02) < rare_release_score(0.10)
    assert rare_release_score(0.22) < rare_release_score(0.16)


def test_lineage_deltas_and_stability_score():
    parent = {
        "id": "parent",
        "metrics": {
            "release_geometric_event": 0.004,
            "phase_transition_score": 0.03,
            "release_duty_cycle": 0.1,
            "mathematical_curiosity": 0.3,
            "channel_separation": 0.2,
        },
    }
    child_metrics = {
        "release_geometric_event": 0.005,
        "phase_transition_score": 0.04,
        "release_duty_cycle": 0.12,
        "mathematical_curiosity": 0.35,
        "channel_separation": 0.21,
    }
    deltas = lineage_deltas(child_metrics, ["parent"], {"parent": parent})
    assert deltas["delta_release_geometric_event"] > 0.0
    assert deltas["delta_phase_transition_score"] > 0.0
    assert deltas["delta_release_duty_cycle"] > 0.0
    assert 0.0 < lineage_stability_score(deltas) <= 1.0


def test_device_helpers_and_candidate_jobs(monkeypatch):
    assert parse_worker_devices("", "cpu") == ["cpu"]
    assert parse_worker_devices("cpu,cuda:0", "cpu") == ["cpu", "cuda:0"]
    assert resolve_device("cpu") == "cpu"
    assert normalize_workers_for_devices(4, ["cpu"]) == 4
    assert normalize_workers_for_devices(3, ["cuda:0"]) == 1
    assert normalize_workers_for_devices(4, ["cuda:0", "cuda:0"]) == 1
    assert normalize_workers_for_devices(4, ["cuda:0", "cuda:1"]) == 2
    assert normalize_workers_for_devices(4, ["cuda:0", "cpu"]) == 2
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert resolve_device("auto") == "cuda:0"
    assert parse_worker_devices("", "auto", workers=3) == ["cuda:0", "cuda:1"]
    args = type("Args", (), {
        "hidden_size": 8,
        "steps": 6,
        "perturb_step": 3,
        "rank": 2,
        "torch_threads": 1,
    })()
    population = [
        population_entry(default_genome(8, rank=2), "default_seed", []),
        population_entry(default_genome(8, rank=2), "mutation", ["parent"]),
    ]
    jobs = candidate_jobs(
        population,
        generation=2,
        args=args,
        seeds=[94],
        scales=[0.25],
        worker_devices=["cpu", "cpu"],
    )
    assert jobs[0]["candidate_id"] == "gen002_candidate000"
    assert jobs[0]["ancestor_ids"] == ["gen002_candidate000"]
    assert jobs[0]["generation_of_origin"] == 2
    assert jobs[0]["mutation_count"] == 0
    assert jobs[1]["parent_ids"] == ["parent"]
    assert jobs[1]["device"] == "cpu"


def test_native_reproduction_has_no_elitism_default_seed_templates_or_random_overlay():
    def row(idx: int, score: float) -> dict:
        return {
            "id": f"parent{idx}",
            "generation": 0,
            "generation_of_origin": 0,
            "mutation_count": 0,
            "ancestor_ids": [f"parent{idx}"],
            "rank_mode": NATIVE_EMERGENCE_RANK_MODE,
            "genome": default_genome(8, rank=2),
            "metrics": {
                "internal_richness": score,
                "channel_separation": 0.2,
                "mathematical_curiosity": 0.3,
                "geometric_coherence": 0.4,
                "release_causal_divergence": 0.01,
                "release_geometric_event": 0.01,
                "phase_transition_score": 0.01,
                "regime_bonus": 0.0,
                "release_duty_cycle": 0.1,
            },
        }

    parents = [row(0, 0.5), row(1, 0.4), row(2, 0.3)]
    population = reproduce(
        parents,
        parents,
        population_size=6,
        generation_of_origin=1,
        rng=random.Random(101),
        hidden_size=8,
        rank=2,
        mutation_sigma=1.0,
        reproduction_weights=NATIVE_REPRODUCTION_WEIGHTS,
        random_injection_rate=0.0,
        include_default_seed=False,
        elitism=False,
        allow_structured_operators=False,
        use_archive_bins_for_parents=False,
    )
    kinds = {entry["reproduction_kind"] for entry in population}
    assert "default_seed" not in kinds
    assert "elite_copy" not in kinds
    assert "random" not in kinds
    assert kinds <= {"mutation", "crossover"}


def test_generation_diagnostics_track_lineage_and_distribution():
    rows = []
    for idx, duty in enumerate((0.0, 0.10, 0.25, 0.70)):
        row = {
            "id": f"gen003_candidate{idx:03d}",
            "generation": 3,
            "parent_ids": ["parent"] if idx else [],
            "ancestor_ids": [f"ancestor{idx % 2}"],
            "generation_of_origin": idx,
            "mutation_count": idx,
            "reproduction_kind": "mutation",
            "rank_score": float(10 - idx),
            "metrics": {
                "release_duty_cycle": duty,
                "release_geometric_event": 0.01 * idx,
                "phase_transition_score": 0.02 * idx,
                "internal_richness": 0.5,
                "channel_separation": 0.4,
                "regimes": {"surface_fixed_accumulating": 1},
            },
        }
        rows.append(row)
    diagnostics = generation_diagnostics(rows, generation=3)
    assert diagnostics["top10_distinct_ancestor_count"] == 2
    assert diagnostics["duty_histogram"]["zero"] == 1
    assert diagnostics["duty_histogram"]["gt_0.06_le_0.14"] == 1
    assert diagnostics["duty_histogram"]["gt_0.18_le_0.30"] == 1
    assert diagnostics["duty_histogram"]["gt_0.60"] == 1
    assert diagnostics["event_phase_scatter_top20"][0]["id"] == "gen003_candidate000"


def test_cuda_lock_is_opt_in(monkeypatch):
    monkeypatch.delenv("DEMIAN_CUDA_SERIALIZE", raising=False)
    assert cuda_lock_enabled() is False
    monkeypatch.setenv("DEMIAN_CUDA_SERIALIZE", "1")
    assert cuda_lock_enabled() is True


def test_export_payload_contains_evo_fields():
    genome = default_genome(8, rank=2)
    motif = motif_suite(8, seed=94)[0]
    run = run_with_motif(
        genome,
        hidden_size=8,
        steps=6,
        seed=94,
        perturb_step=3,
        perturb_scale=0.25,
        motif=motif,
        rank=2,
        device="cpu",
    )
    candidate = {
        "id": "gen000_candidate000",
        "metrics": {"internal_richness": 0.1},
        "rank_score": 1.0,
    }
    config = {
        "eval_seeds": [94],
        "hidden_size": 8,
        "steps": 6,
        "perturb_step": 3,
        "perturb_scales": [0.25],
        "rank": 2,
        "device": "cpu",
    }
    records = records_from_run(candidate, run, config)
    payload = export_payload(records, config, [candidate])
    assert payload["metadata"]["schema_version"] == 2
    assert payload["points"][0]["candidate_id"] == "gen000_candidate000"
    assert "route_metrics.path_curvature_mean" in payload["metadata"]["projection_features"]
