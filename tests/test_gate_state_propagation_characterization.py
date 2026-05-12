"""Tests for gate-state propagation characterization helpers."""

from __future__ import annotations

import json
import math
import os
import sys

import pyarrow.parquet as pq
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.gate_state_propagation_characterization import (
    AblationRow,
    CapsuleRow,
    CharacterizationConfig,
    MotifSpec,
    clamp_channel_state,
    parameter_signature_rows,
    run_characterization,
    run_condition,
)
from development.evolve_v9_5ch_release import default_genome, gain_zero_diagnostics, release_gain_zero_genome


def test_channel_disabled_clamp_zeros_only_requested_channel():
    state = tuple(torch.full((1, 4), float(idx + 1)) for idx in range(5))

    clamped = clamp_channel_state(state, "message")

    assert torch.count_nonzero(clamped[3]) == 0
    for idx in (0, 1, 2, 4):
        assert torch.equal(clamped[idx], state[idx])


def test_gain_zero_emits_zero_release_strength_and_routes():
    config = CharacterizationConfig(
        seeds=[94],
        perturb_scales=[0.2],
        motifs=[MotifSpec(family="basis", index=0)],
        conditions=["gain_zero"],
        hidden_size=8,
        steps=10,
        perturb_step=4,
        capsule_pause_step=4,
    )
    genome = default_genome(8, rank=2)
    run = run_condition(
        release_gain_zero_genome(genome),
        candidate_id="fixture",
        config=config,
        condition="gain_zero",
        seed=94,
        perturb_scale=0.2,
        motif={"family": "basis", "index": 0, "vector": torch.nn.functional.normalize(torch.ones(8), dim=0)},
    )

    checks = gain_zero_diagnostics(run)

    assert checks["gain_zero_clean"] is True
    assert checks["gain_zero_max_release_strength"] == 0.0
    assert checks["gain_zero_max_route_norm"] == 0.0


@given(value=st.floats(allow_nan=False, allow_infinity=False, width=32))
def test_ablation_schema_preserves_extra_fields_and_rejects_nonfinite(value):
    row = AblationRow(
        candidate_id="fixture",
        condition="original",
        seed=94,
        perturb_scale=0.2,
        motif_family="basis",
        motif_index=0,
        step=1,
        residual_norm=abs(float(value)),
        residual_delta=0.0,
        temporal_coherence=1.0,
        release_strength_mean=0.0,
        release_open_mean=0.0,
        release_pressure_mean=0.0,
        fast_state_norm=0.1,
        slow_state_norm=0.2,
        control_state_norm=0.3,
        message_state_norm=0.4,
        carrier_state_norm=0.5,
        extra_probe="kept",
    )

    assert row.model_extra["extra_probe"] == "kept"

    with pytest.raises(ValueError):
        AblationRow(
            **{
                **row.model_dump(),
                "residual_norm": math.inf,
            }
        )


def test_capsule_schema_accepts_expected_row():
    row = CapsuleRow(
        candidate_id="fixture",
        seed=94,
        perturb_scale=0.2,
        motif_family="basis",
        motif_index=0,
        resume_kind="surface_only",
        final_cosine=0.1,
        final_l2_gap=0.2,
        mean_step_gap=0.3,
        fast_gap=0.4,
        slow_gap=0.5,
        control_gap=0.6,
        message_gap=0.7,
        carrier_gap=0.8,
        diagnostic_note="kept",
    )

    assert row.model_extra["diagnostic_note"] == "kept"


def write_candidate(path, candidate_id: str, rank_metric: float, scalar: float, ancestor: str) -> None:
    payload = {
        "id": candidate_id,
        "generation": int(candidate_id[3:6]),
        "index": 0,
        "rank_mode": "native_emergence",
        "genome": {
            "scalars": {
                **default_genome(8, rank=2)["scalars"],
                "release_gain": scalar,
            },
            "matrices": default_genome(8, rank=2)["matrices"],
        },
        "metrics": {
            "internal_richness": rank_metric,
            "channel_separation": 0.1,
            "mathematical_curiosity": 0.1,
            "geometric_coherence": 0.1,
            "release_causal_divergence": 0.1,
            "release_geometric_event": 0.1,
            "phase_transition_score": 0.1,
        },
        "reproduction_kind": "mutation",
        "parent_ids": [ancestor],
        "ancestor_ids": [ancestor],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_parameter_signature_extraction_top_and_distinct_lineages(tmp_path):
    root = tmp_path / "archive"
    candidates = root / "candidates"
    candidates.mkdir(parents=True)
    for idx in range(6):
        write_candidate(
            candidates / f"gen00{idx}_candidate000.json",
            f"gen00{idx}_candidate000",
            rank_metric=float(idx),
            scalar=0.1 + idx,
            ancestor=f"lineage-{idx % 3}",
        )

    rows, summary = parameter_signature_rows(root, top_n=3)

    assert summary["candidate_count"] == 6
    assert {row.selection for row in rows} == {"top_native_rank", "top_distinct_lineage"}
    assert len([row for row in rows if row.selection == "top_native_rank"]) == 3
    assert "release_gain" in summary["scalar_summary"]


def test_smoke_characterization_outputs_loadable_artifacts(tmp_path):
    archive = tmp_path / "archive"
    candidates = archive / "candidates"
    candidates.mkdir(parents=True)
    candidate_path = candidates / "gen000_candidate000.json"
    write_candidate(candidate_path, "gen000_candidate000", 0.5, 0.3, "root")
    out_dir = tmp_path / "diagnostics"
    config = CharacterizationConfig(
        candidate_path=candidate_path,
        archive_root=archive,
        output_dir=out_dir,
        seeds=[94],
        perturb_scales=[0.2],
        motifs=[MotifSpec(family="basis", index=0)],
        conditions=["original", "gain_zero"],
        hidden_size=8,
        steps=10,
        perturb_step=4,
        capsule_pause_step=4,
        write_parameter_signatures=False,
    )

    summary = run_characterization(config)

    assert summary.run_count == 2
    assert json.loads((out_dir / "summary.json").read_text())["candidate_id"] == "gen000_candidate000"
    summary_payload = json.loads((out_dir / "summary.json").read_text())
    assert summary_payload["held_out_seeds"] == [94]
    assert summary_payload["perturb_steps"] == [4]
    assert "capsule_result_flags" in summary_payload
    assert "probe_summary" in summary_payload
    assert len(pq.read_table(out_dir / "ablation_results.parquet")) > 0
    assert len(pq.read_table(out_dir / "capsule_results.parquet")) > 0
