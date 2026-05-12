"""Tests for Dynamic Selection Probe checkpoint planning."""

from __future__ import annotations

import json
from pathlib import Path

from development.evolution.scoring import DYNAMIC_SELECTION_PROBE_RANK_MODE
from development.run_dynamic_selection_probe import checkpoint_plan, select_checkpoint_candidates


def write_candidate(root: Path, generation: int, index: int, *, richness: float) -> None:
    payload = {
        "id": f"gen{generation:03d}_candidate{index:03d}",
        "generation": generation,
        "rank_mode": DYNAMIC_SELECTION_PROBE_RANK_MODE,
        "metrics": {
            "internal_richness": richness,
            "channel_separation": 0.2,
            "mathematical_curiosity": 0.3,
            "geometric_coherence": 0.4,
            "release_causal_divergence": 0.01,
            "release_geometric_event": 0.02,
            "phase_transition_score": 0.03,
            "release_timing_score": 0.4,
            "release_duty_cycle": 0.1,
        },
        "rank_components": {},
        "parent_ids": [f"parent_{index}"],
        "ancestor_ids": [f"ancestor_{index}"],
        "genome": {"scalars": {"release_gain": index}},
    }
    path = root / "candidates" / f"{payload['id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_select_checkpoint_candidates_ranks_each_checkpoint_generation(tmp_path):
    for generation in (19, 39, 59):
        for index, richness in enumerate((0.1, 0.8, 0.4)):
            write_candidate(tmp_path, generation, index, richness=richness)

    selected = select_checkpoint_candidates(tmp_path, checkpoints={19: 2, 39: 1, 59: 2})

    assert [row["candidate_id"] for row in selected] == [
        "gen019_candidate001",
        "gen019_candidate002",
        "gen039_candidate001",
        "gen059_candidate001",
        "gen059_candidate002",
    ]
    assert selected[0]["checkpoint_rank"] == 1
    assert selected[1]["checkpoint_rank"] == 2


def test_checkpoint_plan_emits_heldout_confirmation_commands(tmp_path):
    for generation in (19, 39, 59):
        write_candidate(tmp_path, generation, 0, richness=0.5)
    diagnostics = tmp_path / "diagnostics"

    plan = checkpoint_plan(tmp_path, diagnostics)
    command = plan["selected"][0]["confirmation_commands"][0]

    assert plan["held_out_seeds"] == [96, 97, 98]
    assert command[0] == "venv/bin/python"
    assert "development/gate_state_propagation_characterization.py" in command
    assert command[command.index("--seeds") + 1] == "96,97,98"
    assert command[command.index("--perturb-scales") + 1] == "0.2,0.35,0.7"
    assert command[command.index("--output-dir") + 1].startswith(diagnostics.as_posix())
