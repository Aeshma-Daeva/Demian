"""Tests for v9 five-channel evolution summary rebuilding."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from development.summarize_v9_5ch_evolution import load_candidates, summarize


def test_rebuild_v10_summary_core_fields():
    roots = [
        Path(f"data/evolution/v10_0_frozen_evolution_island_{index}_20260510")
        for index in range(1, 5)
    ]
    candidates = load_candidates(roots)
    summary = summarize(candidates, "v10.0-frozen-evolution", 94)

    assert summary["candidate_count"] == 640
    assert summary["generation_count"] == 20
    assert summary["final_generation"] == 19
    assert summary["final_best"]["id"] == "gen019_candidate001"
    assert summary["final_best"]["island"] == "island_1"
    assert summary["final_best"]["release_duty_cycle"] == 0.171875
    assert summary["curve_summary"]["duty_delta_last5_minus_first5"] < 0
    assert summary["curve_summary"]["event_delta_last5_minus_first5"] > 0
    assert summary["curve_summary"]["phase_delta_last5_minus_first5"] > 0
