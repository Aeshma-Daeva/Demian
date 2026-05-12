"""Tests for research-lab tooling integrations."""

from __future__ import annotations

import json

import pyarrow.parquet as pq
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from development.lab_schemas import CandidateArtifact, EvolutionConfigArtifact, validate_artifact
from development.lab_tools import (
    create_study,
    log_candidate_to_mlflow,
    write_candidates_parquet,
    write_tensorboard_metrics,
)


def sample_candidate() -> dict:
    return {
        "id": "gen002_candidate003",
        "generation": 2,
        "index": 3,
        "genome": {"scalars": {}, "matrices": {}},
        "metrics": {
            "internal_richness": 0.4,
            "channel_separation": 0.2,
            "release_duty_cycle": 0.12,
            "release_causal_divergence": 0.03,
            "release_timing_score": 0.5,
            "phase_transition_score": 0.7,
            "geometric_coherence": 0.8,
            "regime_bonus": 0.25,
            "regimes": {"bounded_strange": 1},
        },
        "reproduction_kind": "delayed_eligibility_template_mutation",
        "parent_ids": ["gen001_candidate000"],
        "ancestor_ids": ["gen000_candidate000"],
        "rank_score": 3.5,
        "rank_components": {"timing_bonus": 0.2},
        "runs": [],
    }


def test_pydantic_candidate_and_config_artifact_validation(tmp_path):
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(sample_candidate()) + "\n")
    candidate = validate_artifact(candidate_path)
    assert isinstance(candidate, CandidateArtifact)
    assert candidate.metrics.release_duty_cycle == pytest.approx(0.12)

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "population": 8,
                "generations": 20,
                "seed": 2026051101,
                "eval_seeds": [94, 95],
                "hidden_size": 32,
                "steps": 128,
                "perturb_step": 64,
                "perturb_scales": [0.35, 0.7],
                "rank": 2,
                "experiment_version": "demian-v1-delayed-eligibility",
            }
        )
    )
    config = validate_artifact(config_path)
    assert isinstance(config, EvolutionConfigArtifact)


@given(
    duty=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    generation=st.integers(min_value=0, max_value=999),
)
@settings(max_examples=25, deadline=None)
def test_candidate_schema_accepts_valid_duty_fraction(duty, generation):
    payload = sample_candidate()
    payload["generation"] = generation
    payload["id"] = f"gen{generation:03d}_candidate000"
    payload["metrics"]["release_duty_cycle"] = duty
    parsed = CandidateArtifact.model_validate(payload)
    assert parsed.metrics.release_duty_cycle == pytest.approx(duty)


def test_lab_tracking_optimization_tensorboard_and_parquet_smoke(tmp_path):
    candidate = sample_candidate()

    study = create_study("smoke", tmp_path / "optuna.sqlite3")
    study.enqueue_trial({"release_threshold": 0.7})

    run_id = log_candidate_to_mlflow(
        candidate,
        tracking_dir=tmp_path / "mlruns",
        experiment_name="demian-smoke",
    )
    assert run_id

    tensorboard_dir = tmp_path / "tb"
    write_tensorboard_metrics(candidate["metrics"], tensorboard_dir, step=2)
    assert any(path.name.startswith("events.out.tfevents") for path in tensorboard_dir.iterdir())

    parquet_path = tmp_path / "candidates.parquet"
    write_candidates_parquet([candidate], parquet_path)
    table = pq.read_table(parquet_path)
    assert table.num_rows == 1
    assert "metric_release_timing_score" in table.column_names
