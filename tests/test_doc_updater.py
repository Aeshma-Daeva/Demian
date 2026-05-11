"""Smoke tests for generated documentation updater."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from development import update_docs


def test_substrate_registry_parse():
    registry = update_docs.parse_substrate_registry()
    assert "demian_native_v6" in registry
    assert "demian_native_v7" in registry
    assert "demian_native_v7.1" in registry
    assert "demian_native_v7.4" in registry
    assert "dual_gru_v3b" in registry
    print("  PASS test_substrate_registry_parse")


def test_latest_native_substrate():
    latest = update_docs.latest_native_substrate(
        [
            "demian_native_v0",
            "demian_native_v5.3",
            "demian_native_v6",
            "demian_native_v7",
            "demian_native_v7.1",
            "demian_native_v7.4",
            "demian_native_v8",
            "demian_native_v9",
            "demian_native_v90",
            "demian_native_v52c",
        ]
    )
    assert latest == "demian_native_v9"
    print("  PASS test_latest_native_substrate")


def test_validate_links_runs():
    issues = update_docs.validate_links()
    assert isinstance(issues, list)
    print("  PASS test_validate_links_runs")


def test_manifest_validation_runs():
    issues = update_docs.validate_manifest()
    assert isinstance(issues, list)
    assert not issues
    paths = update_docs.manifest_artifact_paths()
    assert "data/evolution/v10_0_frozen_evolution_4island_20260510_summary.json" in paths
    print("  PASS test_manifest_validation_runs")


if __name__ == "__main__":
    tests = [
        test_substrate_registry_parse,
        test_latest_native_substrate,
        test_validate_links_runs,
        test_manifest_validation_runs,
    ]
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - standalone no-pytest runner
            failed += 1
            print(f"  FAIL {test.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    raise SystemExit(1 if failed else 0)
