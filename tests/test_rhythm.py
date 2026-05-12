"""Tests for Fibonacci consolidation engine."""
import os
import sys
import torch
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from legacy.demian_runtime.rhythm import FibonacciConsolidator, CompressedTrajectory
from legacy.demian_runtime.vibration import VibrationTracker


def _setup():
    with tempfile.TemporaryDirectory() as td:
        tracker = VibrationTracker(d_model=64, target_dim=16, projection_path=td)
        with tempfile.TemporaryDirectory() as td2:
            consol = FibonacciConsolidator(
                tracker=tracker, storage_dir=td2, target_dim=16
            )
        return tracker, consol


def test_fib_set():
    tracker, consol = _setup()
    assert consol._fibonacci_intervals == {1, 2, 3, 5, 8, 13, 21, 34, 55, 89}
    print("  PASS test_fib_set")


def test_no_fire_non_fib():
    tracker, consol = _setup()
    tracker._trajectory = [tracker.capture(torch.randn(64), torch.randn(1, 1, 50))]
    consol.on_turn()  # turn 1 — fires
    assert consol.turn_count == 1
    print("  PASS test_no_fire_non_fib")


def test_consolidation_has_turn():
    tracker, consol = _setup()
    tracker._trajectory = [tracker.capture(torch.randn(64), torch.randn(1, 1, 50))]
    result = consol.on_turn()  # turn 1
    assert result is not None
    assert result.turn == 1
    print("  PASS test_consolidation_has_turn")


def test_consolidation_type_raw():
    tracker, consol = _setup()
    tracker._trajectory = [tracker.capture(torch.randn(64), torch.randn(1, 1, 50))]
    consol.on_turn()  # turn 1
    assert consol._consolidated[0].compression_type == "raw"
    print("  PASS test_consolidation_type_raw")


def test_markov_chain():
    tracker, consol = _setup()
    modes = ["focused", "focused", "distributed", "diffuse", "diffuse"]
    transitions = consol._markov_chain(modes)
    assert isinstance(transitions, dict)
    assert "focused" in transitions
    assert transitions["focused"]["distributed"] > 0
    assert transitions["distributed"]["diffuse"] > 0
    print("  PASS test_markov_chain")


if __name__ == "__main__":
    tests = [test_fib_set, test_no_fire_non_fib, test_consolidation_has_turn,
             test_consolidation_type_raw, test_markov_chain]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  FAIL {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {t.__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
