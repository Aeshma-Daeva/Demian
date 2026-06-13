"""Tests for identity-continuity characterization helpers."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.evolve_v9_5ch_release import default_genome
from development.gate_state_propagation_characterization import MotifSpec
from development.identity_continuity_characterization import (
    IdentityProbeConfig,
    apply_lesion,
    identity_metrics,
    identity_signature_from_window,
    run_boundary_source_probe,
    run_identity_characterization,
    run_identity_lifetime_probe,
    run_self_maintenance_lesion_probe,
    scramble_identity_channels,
)


def fixture_components(value: float = 1.0) -> dict[str, torch.Tensor]:
    return {
        "fast": torch.full((2,), value),
        "slow": torch.tensor([value, value + 1.0]),
        "control": torch.tensor([value + 2.0, value + 3.0]),
        "message": torch.full((2,), value + 4.0),
        "carrier": torch.tensor([value + 5.0, value + 6.0]),
    }


def write_candidate(path: Path) -> None:
    genome = default_genome(8, rank=2)
    payload = {
        "id": "gen000_candidate000",
        "genome": genome,
        "metrics": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def tiny_config(tmp_path: Path) -> IdentityProbeConfig:
    return IdentityProbeConfig(
        output_dir=tmp_path / "identity",
        seeds=[94],
        perturb_scales=[0.2],
        motifs=[MotifSpec(family="basis", index=0)],
        hidden_size=8,
        rank=2,
        steps=16,
        burn_in_steps=4,
        perturb_step=8,
        lesion_step=8,
        tail_window=4,
    )


def test_identity_signature_uses_slow_control_carrier_window():
    window = [fixture_components(1.0), fixture_components(2.0)]

    signature = identity_signature_from_window(window)

    assert signature.channels == ["slow", "control", "carrier"]
    assert len(signature.mean_vector) == 6
    assert set(signature.channel_pair_cosines) == {"slow:control", "slow:carrier", "control:carrier"}
    assert signature.channel_norms["slow"] > 0.0


def test_identity_metrics_report_exact_return_for_signature_components():
    components = fixture_components(1.0)
    signature = identity_signature_from_window([components])

    metrics = identity_metrics(components, signature)

    assert metrics["identity_distance"] == pytest.approx(0.0)
    assert metrics["identity_cosine"] == pytest.approx(1.0)


def test_scramble_and_lesion_conditions_modify_expected_identity_channels():
    state = tuple(torch.arange(1.0, 5.0).view(1, 4) + idx for idx in range(5))

    scrambled = scramble_identity_channels(state)
    lesioned = apply_lesion(state, "control_zero")

    assert torch.equal(scrambled[0], state[0])
    assert torch.equal(scrambled[1], -torch.flip(state[1], dims=(-1,)))
    assert torch.equal(scrambled[2], -torch.flip(state[2], dims=(-1,)))
    assert torch.equal(scrambled[4], -torch.flip(state[4], dims=(-1,)))
    assert torch.count_nonzero(lesioned[2]) == 0
    assert torch.equal(lesioned[1], state[1])


def test_identity_lifetime_probe_smoke_runs_preserved_and_reset(tmp_path):
    genome = default_genome(8, rank=2)
    config = tiny_config(tmp_path)
    motif = {"family": "basis", "index": 0, "vector": torch.nn.functional.normalize(torch.ones(8), dim=0)}

    preserved = run_identity_lifetime_probe(
        genome,
        candidate_id="fixture",
        config=config,
        condition="identity_preserved",
        seed=94,
        perturb_scale=0.2,
        motif=motif,
    )
    reset = run_identity_lifetime_probe(
        genome,
        candidate_id="fixture",
        config=config,
        condition="identity_reset",
        seed=94,
        perturb_scale=0.2,
        motif=motif,
    )

    assert preserved["summary"][0].probe == "lifetime"
    assert reset["summary"][0].condition == "identity_reset"
    assert len(preserved["rows"]) == config.steps - config.burn_in_steps


def test_boundary_and_lesion_probes_emit_source_and_condition_rows(tmp_path):
    genome = default_genome(8, rank=2)
    config = tiny_config(tmp_path)
    motif = {"family": "basis", "index": 0, "vector": torch.nn.functional.normalize(torch.ones(8), dim=0)}

    boundary = run_boundary_source_probe(
        genome,
        candidate_id="fixture",
        config=config,
        source="internal",
        seed=94,
        perturb_scale=0.2,
        motif=motif,
    )
    lesion = run_self_maintenance_lesion_probe(
        genome,
        candidate_id="fixture",
        config=config,
        condition="carrier_zero",
        seed=94,
        perturb_scale=0.2,
        motif=motif,
    )

    assert boundary["summary"][0].source == "internal"
    assert lesion["summary"][0].condition == "carrier_zero"


def test_identity_characterization_writes_loadable_artifacts(tmp_path):
    candidate_path = tmp_path / "archive" / "candidates" / "gen000_candidate000.json"
    write_candidate(candidate_path)
    config = tiny_config(tmp_path)
    config.candidate_path = candidate_path
    config.archive_root = candidate_path.parent.parent

    result = run_identity_characterization(
        default_genome(8, rank=2),
        candidate_id="gen000_candidate000",
        config=config,
        probes=("lifetime",),
    )

    summary = json.loads((config.output_dir / "identity_summary.json").read_text(encoding="utf-8"))
    assert result["row_count"] > 0
    assert summary["candidate_id"] == "gen000_candidate000"
    assert (config.output_dir / "identity_rows.jsonl").exists()
    assert (config.output_dir / "identity_summary.csv").exists()
