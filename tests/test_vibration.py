"""Tests for vibration tracker."""
import torch
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demian.vibration import VibrationTracker


def _tracker():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        from demian.vibration import VibrationTracker
        return VibrationTracker(d_model=64, target_dim=16, projection_path=td)


def test_capture():
    t = _tracker()
    residual = torch.randn(64)
    logits = torch.randn(1, 1, 50)
    snap = t.capture(residual, logits)
    assert snap.step == 1
    assert len(snap.projected_state) == 16
    assert snap.attention is snap.shape
    assert snap.shape.attention_dim > 0.0
    assert snap.shape.peakiness > 0.0
    print("  PASS test_capture")


def test_first_coh_one():
    t = _tracker()
    snap = t.capture(torch.randn(64), torch.randn(1, 1, 50))
    assert snap.temporal_coherence == 1.0
    print("  PASS test_first_coh_one")


def test_coh_changes():
    t = _tracker()
    t.capture(torch.randn(64), torch.randn(1, 1, 50))
    snap2 = t.capture(torch.randn(64), torch.randn(1, 1, 50))
    assert -1.0 <= snap2.temporal_coherence <= 1.0
    print("  PASS test_coh_changes")


def test_focused_attention():
    t = _tracker()
    logits = torch.zeros(1, 1, 50)
    logits[0, 0, 10] = 10.0  # one dominant token
    snap = t.capture(torch.randn(64), logits)
    assert snap.shape.peakiness > 0.99
    assert snap.shape.dominance_ratio > 50.0
    assert snap.shape.attention_dim < 2.0
    print("  PASS test_focused_attention")


def test_diffuse_attention():
    t = _tracker()
    logits = torch.randn(1, 1, 50) * 0.01  # nearly uniform
    snap = t.capture(torch.randn(64), logits)
    assert snap.shape.peakiness < 0.03
    assert snap.shape.dominance_ratio < 1.1
    assert snap.shape.attention_dim > 45.0
    print("  PASS test_diffuse_attention")


def test_reset():
    t = _tracker()
    t.capture(torch.randn(64), torch.randn(1, 1, 50))
    assert t.step_count == 1
    t.reset()
    assert t.step_count == 0
    assert len(t.get_trajectory()) == 0
    print("  PASS test_reset")


if __name__ == "__main__":
    tests = [test_capture, test_first_coh_one, test_coh_changes,
             test_focused_attention, test_diffuse_attention, test_reset]
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
