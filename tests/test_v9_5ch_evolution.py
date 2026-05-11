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
    candidate_jobs,
    cuda_lock_enabled,
    default_genome,
    evaluate_genome,
    generation_diagnostics,
    lineage_deltas,
    lineage_stability_score,
    motif_suite,
    mutate_genome,
    normalize_workers_for_devices,
    parse_worker_devices,
    population_entry,
    PHASE_TRANSITION_RANK_CAP,
    rank_components,
    RELEASE_GEOMETRIC_RANK_CAP,
    release_local_metrics,
    release_timing_metrics,
    rare_release_score,
    resolve_device,
    run_with_motif,
    target_dims,
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


def test_adaptive_model_low_rank_delta_shapes():
    model = AdaptiveV9FiveChannel(8, default_genome(8, rank=2), rank=2)
    state = model.initial_state(1, torch.device("cpu"))
    assert len(state) == 5
    assert model._delta("message_gate").shape == (8, 8)
    assert model._delta("release_gate").shape == (1, 40)


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


def test_rank_components_prioritize_machine_native_geometry():
    row = {
        "metrics": {
            "internal_richness": 0.5,
            "channel_separation": 0.2,
            "release_local_causality": 0.0004,
            "release_geometric_event": 0.012,
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
    assert components["release_geometric_event"] > 0.0
    assert components["phase_transition"] > 0.0
    assert components["channel_separation"] > 0.0
    assert components["mathematical_curiosity"] > 0.0
    assert components["flood_penalty"] == 0.0
    assert "release_event_duty" not in components
    assert "boundedness" not in components


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


def test_rank_components_use_raised_release_and_phase_caps():
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
                "release_geometric_event": RELEASE_GEOMETRIC_RANK_CAP / 2.0,
                "phase_transition_score": PHASE_TRANSITION_RANK_CAP / 2.0,
            }
        }
    )
    over_cap = rank_components(
        {
            "metrics": {
                **base_metrics,
                "release_geometric_event": RELEASE_GEOMETRIC_RANK_CAP * 2.0,
                "phase_transition_score": PHASE_TRANSITION_RANK_CAP * 2.0,
            }
        }
    )
    assert half_cap["release_geometric_event"] == pytest.approx(0.8)
    assert half_cap["phase_transition"] == pytest.approx(0.6)
    assert over_cap["release_geometric_event"] == pytest.approx(1.6)
    assert over_cap["phase_transition"] == pytest.approx(1.2)


def test_rank_components_penalize_release_flooding_on_sparse_gate_domain():
    base_metrics = {
        "internal_richness": 0.5,
        "channel_separation": 0.2,
        "release_geometric_event": 0.012,
        "phase_transition_score": 0.05,
        "mathematical_curiosity": 0.4,
        "boundedness": 1.0,
        "geometric_coherence": 0.7,
        "regime_bonus": 0.5,
    }
    at_threshold = rank_components({"metrics": {**base_metrics, "release_duty_cycle": 0.18}})
    mid_flood = rank_components({"metrics": {**base_metrics, "release_duty_cycle": 0.30}})
    high_flood = rank_components({"metrics": {**base_metrics, "release_duty_cycle": 0.60}})
    assert at_threshold["flood_penalty"] == 0.0
    assert mid_flood["flood_penalty"] == pytest.approx(-0.4)
    assert high_flood["flood_penalty"] == pytest.approx(-1.4)


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
    sparse_release_rows = [trajectory[69], trajectory[71]]
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
