"""Tests for directional control setpoint probes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from development.control_capsule_maintenance import components_from_state
from development.control_setpoint_probe import (
    ControlSetpointConfig,
    apply_capsule_perturbation,
    capsule_direction_from_signature,
    directional_asymmetry,
    run_control_setpoint_probe,
    signed_projection,
)
from development.evolve_v9_5ch_release import default_genome
from development.gate_state_propagation_characterization import MotifSpec


def write_candidate(path: Path) -> None:
    payload = {
        "id": "gen000_candidate000",
        "genome": default_genome(8, rank=2),
        "metrics": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def capsule_delta_norm(state: tuple[torch.Tensor, ...], reference: tuple[torch.Tensor, ...]) -> float:
    delta = torch.cat(
        [
            (state[1] - reference[1]).view(-1),
            (state[3] - reference[3]).view(-1),
            (state[4] - reference[4]).view(-1),
        ]
    )
    return float(delta.norm())


def test_capsule_perturbations_are_equal_and_opposite_with_matched_norm():
    state = tuple(torch.zeros(1, 4) for _ in range(5))
    direction = capsule_direction_from_signature(torch.tensor([1.0, 2.0, -1.0, 0.5]))

    plus = apply_capsule_perturbation(state, direction, epsilon=0.25, sign="plus")
    minus = apply_capsule_perturbation(state, direction, epsilon=0.25, sign="minus")

    assert capsule_delta_norm(plus, state) == pytest.approx(0.25, rel=1e-6, abs=1e-6)
    assert capsule_delta_norm(minus, state) == pytest.approx(0.25, rel=1e-6, abs=1e-6)
    for channel_idx in (1, 3, 4):
        assert torch.allclose(plus[channel_idx], -minus[channel_idx])


def test_signed_projection_returns_opposite_initial_signs_for_plus_minus():
    state = tuple(torch.zeros(1, 4) for _ in range(5))
    direction = capsule_direction_from_signature(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    plus = apply_capsule_perturbation(state, direction, epsilon=0.2, sign="plus")
    minus = apply_capsule_perturbation(state, direction, epsilon=0.2, sign="minus")
    target = components_from_state(SimpleStateModel(), state)

    plus_projection = signed_projection(components_from_state(SimpleStateModel(), plus), target, direction, epsilon=0.2)
    minus_projection = signed_projection(components_from_state(SimpleStateModel(), minus), target, direction, epsilon=0.2)

    assert plus_projection == pytest.approx(1.0, rel=1e-6, abs=1e-6)
    assert minus_projection == pytest.approx(-1.0, rel=1e-6, abs=1e-6)


def test_symmetric_synthetic_decays_have_low_asymmetry():
    assert directional_asymmetry(0.50, 0.51) < 0.02


def test_asymmetric_synthetic_decays_have_high_asymmetry():
    assert directional_asymmetry(1.0, 0.1) > 0.8


def test_control_setpoint_probe_writes_artifacts(tmp_path):
    candidate_path = tmp_path / "archive" / "candidates" / "gen000_candidate000.json"
    write_candidate(candidate_path)
    config = ControlSetpointConfig(
        candidate_paths=[candidate_path],
        output_dir=tmp_path / "control_setpoint",
        seeds=[96],
        perturb_scales=[0.2],
        motifs=[MotifSpec(family="basis", index=0)],
        direction_motifs=[MotifSpec(family="basis", index=0), MotifSpec(family="gaussian", index=0)],
        hidden_size=8,
        rank=2,
        steps=12,
        perturb_step=4,
        capsule_pause_step=4,
        tail_window=4,
        epsilon=0.05,
    )

    payload = run_control_setpoint_probe(config)

    assert (config.output_dir / "control_setpoint_rows.jsonl").exists()
    assert (config.output_dir / "control_setpoint_summary.csv").exists()
    assert (config.output_dir / "control_setpoint_summary.json").exists()
    assert (config.output_dir / "README.md").exists()
    assert payload["row_count"] >= 12
    assert json.loads((config.output_dir / "control_setpoint_summary.json").read_text(encoding="utf-8"))[
        "experiment"
    ] == "control_setpoint_probe"
    assert {
        row["resume_kind"]
        for row in payload["by_candidate_direction_resume"]
    } == {"control_preserved", "control_zero_at_resume", "control_clamped_zero"}


class SimpleStateModel:
    def state_components(self, state: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
        return {
            "fast": state[0],
            "slow": state[1],
            "control": state[2],
            "message": state[3],
            "carrier": state[4],
        }
