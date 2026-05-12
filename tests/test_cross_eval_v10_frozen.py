"""Tests for v10 frozen cross-evaluation helpers."""

from __future__ import annotations

import copy
import json

import pytest

from development.cross_eval_v10_frozen import (
    ablated_genome,
    find_candidate,
    flatten_summary,
    parse_candidate_spec,
)


def sample_genome() -> dict:
    return {
        "scalars": {
            "release_gain": 0.25,
            "release_to_fast_scale": 1.0,
            "release_to_slow_scale": 0.5,
            "release_to_control_scale": 0.25,
            "release_to_message_scale": 0.125,
            "release_to_carrier_scale": 0.0625,
        },
        "matrices": {},
    }


def test_parse_candidate_spec_requires_label_archive_and_id():
    spec = parse_candidate_spec("island_1:data/evolution/example:gen001_candidate002")

    assert spec["label"] == "island_1"
    assert str(spec["archive_dir"]) == "data/evolution/example"
    assert spec["candidate_id"] == "gen001_candidate002"
    with pytest.raises(ValueError):
        parse_candidate_spec("missing-parts")


def test_find_candidate_reads_list_archives(tmp_path):
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    (archive_dir / "archive.json").write_text(
        json.dumps([{"id": "candidate-a"}, {"id": "candidate-b"}]),
        encoding="utf-8",
    )

    assert find_candidate(archive_dir, "candidate-b") == {"id": "candidate-b"}
    with pytest.raises(KeyError):
        find_candidate(archive_dir, "missing")


def test_ablated_genome_is_deep_copy_and_supports_release_ablations():
    genome = sample_genome()
    original = copy.deepcopy(genome)

    gain_zero = ablated_genome(genome, "release_gain_zero")
    routes_disabled = ablated_genome(genome, "release_routes_disabled")

    assert genome == original
    assert gain_zero["scalars"]["release_gain"] == 0.0
    assert gain_zero["scalars"]["release_to_fast_scale"] == 1.0
    assert routes_disabled["scalars"]["release_gain"] == 0.25
    assert routes_disabled["scalars"]["release_to_fast_scale"] == 1.0
    assert routes_disabled["scalars"]["release_to_carrier_scale"] == 0.0625


def test_flatten_summary_prefers_top_level_then_metrics():
    row = {
        "label": "island_1",
        "island": "island_1",
        "candidate_id": "gen018_candidate003",
        "ablation": "original",
        "run_count": 144,
        "rank_score": 2.0,
        "metrics": {
            "release_duty_cycle": 0.45,
            "release_strength_mean": 0.05,
            "internal_richness": 0.8,
        },
    }

    flat = flatten_summary(row)

    assert flat["label"] == "island_1"
    assert flat["run_count"] == 144
    assert flat["release_duty_cycle"] == 0.45
    assert flat["internal_richness"] == 0.8
