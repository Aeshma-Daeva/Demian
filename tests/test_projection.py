"""Tests for random projection engine."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import tempfile
from legacy.demian_runtime.noise import generate_projection, load_or_create_projection


def test_shape():
    P = generate_projection(d_model=3584, target_dim=128)
    assert P.shape == (128, 3584), f"Expected (128, 3584), got {P.shape}"
    print("  PASS test_shape")

def test_smaller():
    P = generate_projection(d_model=64, target_dim=4)
    assert P.shape == (4, 64), f"Expected (4, 64), got {P.shape}"
    print("  PASS test_smaller")

def test_orthogonality():
    P = generate_projection(d_model=256, target_dim=32)
    G = P @ P.T
    off_diag = G - torch.diag(torch.diag(G))
    mean = off_diag.abs().mean().item()
    assert mean < 5.0, f"Off-diag mean {mean} too high"
    print(f"  PASS test_orthogonality (off-diag: {mean:.4f})")

def test_structure():
    P = generate_projection(d_model=256, target_dim=32)
    x = torch.randn(256)
    y = x + 0.01 * torch.randn(256)
    ratio = torch.norm(P @ x - P @ y) / torch.norm(x - y)
    assert 0.1 < ratio < 10.0, f"Ratio {ratio} out of bounds"
    print(f"  PASS test_structure (ratio: {ratio:.3f})")

def test_cache():
    with tempfile.TemporaryDirectory() as td:
        P1 = load_or_create_projection(128, 16, cache_dir=td)
        P2 = load_or_create_projection(128, 16, cache_dir=td)
        assert torch.allclose(P1, P2)
    print("  PASS test_cache")

def test_seed():
    P1 = generate_projection(64, 8, seed=42)
    P2 = generate_projection(64, 8, seed=42)
    assert torch.allclose(P1, P2)
    print("  PASS test_seed")

if __name__ == "__main__":
    tests = [test_shape, test_smaller, test_orthogonality, test_structure, test_cache, test_seed]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
