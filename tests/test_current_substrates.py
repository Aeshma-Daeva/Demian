"""Tests for the compact current substrate workbench."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.substrates.current import (
    CURRENT_BASELINE,
    CURRENT_COMPARISON,
    CURRENT_SUBSTRATES,
    CURRENT_TARGET,
    compare_current_target,
    current_entrypoints,
    current_substrate_specs,
    export_current_trajectory_3d,
    load_current_experiment_digest,
    load_latest_evolution_summary,
    load_manifest_run,
    make_current_substrate,
)
from development.substrates.legacy import (
    DemianNativeV9Substrate,
    SelfLoopRunner,
    compare_native_v9_vs_v8,
)


def test_current_substrate_constants():
    assert CURRENT_TARGET == "demian_native_v9"
    assert CURRENT_COMPARISON == "demian_native_v8"
    assert CURRENT_BASELINE == "demian_native_v7.4"
    assert CURRENT_SUBSTRATES == (
        "demian_native_v9",
        "demian_native_v8",
        "demian_native_v7.4",
    )


def test_legacy_boundary_exports_active_symbols():
    assert DemianNativeV9Substrate.__name__ == "DemianNativeV9Substrate"
    assert SelfLoopRunner.__name__ == "SelfLoopRunner"
    assert SelfLoopRunner.__module__ == "development.substrates.legacy"
    result = compare_native_v9_vs_v8(
        hidden_size=16,
        steps=8,
        perturb_step=4,
        seeds=[94],
        device="cpu",
    )
    assert result["seeds"] == [94]
    assert "aggregate" in result


def test_make_current_substrate_v9():
    substrate = make_current_substrate(CURRENT_TARGET, hidden_size=16)
    assert substrate.__class__.__name__ == "DemianNativeV9Substrate"


def test_current_workbench_describes_active_surface():
    specs = current_substrate_specs()
    names = {spec["name"] for spec in specs}
    assert names == set(CURRENT_SUBSTRATES)
    assert all(spec["implementation_path"] == "development/substrate_lab.py" for spec in specs)

    entrypoints = current_entrypoints()
    assert "docs/SUBSTRATE_ANATOMY.md" in entrypoints["docs"]
    assert entrypoints["legacy_lab"] == "development/substrate_lab.py"
    assert entrypoints["runtime"] == "development/substrates/runtime.py"
    assert entrypoints["legacy_boundary"] == "development/substrates/legacy.py"
    assert entrypoints["current_experiment"]["experiment_id"] == "demian-v1"
    assert entrypoints["current_experiment"]["experiment_name"] == "Demian v1"
    assert (
        entrypoints["current_experiment"]["predecessor_evidence_id"]
        == "v10.0-frozen-evolution"
    )
    assert entrypoints["current_experiment"]["substrate_baseline"] == CURRENT_TARGET
    assert entrypoints["artifacts"]["demian_v1_predecessor"].endswith(
        "v10_0_frozen_evolution_4island_20260510_summary.json"
    )


def test_current_artifact_loaders():
    summary = load_latest_evolution_summary()
    assert summary["experiment"] == "v10.0-frozen-evolution"
    assert summary["candidate_count"] == 640
    digest = load_current_experiment_digest()
    assert digest["current_experiment"] == "demian-v1"
    assert digest["predecessor_evidence"] == "v10.0-frozen-evolution"
    assert digest["final_best_id"] == "gen019_candidate001"
    assert digest["final_best_duty"] == 0.171875

    run = load_manifest_run("v10_0_frozen_evolution_20260510")
    assert run["status"] == "active_evidence"
    assert run["primary_summary"].endswith("v10_0_frozen_evolution_4island_20260510_summary.json")


def test_compare_current_target_smoke():
    result = compare_current_target(
        hidden_size=16,
        steps=16,
        perturb_step=8,
        seeds=[94],
        device="cpu",
    )
    assert result["seeds"] == [94]
    assert "aggregate" in result
    assert "mean_v9_recovery_1.0" in result["aggregate"]


def test_export_current_trajectory_3d_smoke(tmp_path):
    out_path = export_current_trajectory_3d(
        out_dir=tmp_path,
        run_name="tiny",
        hidden_size=16,
        steps=16,
        perturb_step=8,
        seeds=[94],
        perturb_scales=[0.25],
        device="cpu",
    )

    import json

    payload = json.loads(out_path.read_text())
    assert set(payload) >= {"metadata", "points", "metrics", "events", "summaries"}
    assert payload["metadata"]["substrates"] == [CURRENT_TARGET, CURRENT_COMPARISON]
    assert payload["projection"]["axis_loadings"]
    assert {
        point["substrate"] for point in payload["points"]
    } == {CURRENT_TARGET, CURRENT_COMPARISON}
    assert payload["events"]
    assert payload["summaries"]["runs"]

    for point in payload["points"]:
        assert isinstance(point["x"], (int, float))
        assert isinstance(point["y"], (int, float))
        assert isinstance(point["z"], (int, float))
