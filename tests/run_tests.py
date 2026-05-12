#!/usr/bin/env python
"""Simple test runner - no pytest dependency."""
import os, sys
# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
import torch
from legacy.demian_runtime.noise import generate_projection, load_or_create_projection
import tempfile

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
    print(f"  PASS test_orthogonality (off-diag mean: {mean:.4f})")

def test_structure():
    P = generate_projection(d_model=256, target_dim=32)
    x = torch.randn(256)
    y = x + 0.01 * torch.randn(256)
    dist_orig = torch.norm(x - y)
    dist_proj = torch.norm(P @ x - P @ y)
    ratio = dist_proj / dist_orig
    assert 0.1 < ratio < 10.0, f"Ratio {ratio} out of bounds"
    print(f"  PASS test_structure (ratio: {ratio:.3f})")

def test_cache():
    with tempfile.TemporaryDirectory() as td:
        P1 = load_or_create_projection(128, 16, cache_dir=td)
        P2 = load_or_create_projection(128, 16, cache_dir=td)
        assert torch.allclose(P1, P2), "Cached projections differ"
    print("  PASS test_cache")

def test_seed():
    P1 = generate_projection(64, 8, seed=42)
    P2 = generate_projection(64, 8, seed=42)
    assert torch.allclose(P1, P2), "Seeded projections differ"
    print("  PASS test_seed")

if __name__ == "__main__":
    tests = [test_shape, test_smaller, test_orthogonality, test_structure, test_cache, test_seed]
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
    sys.exit(1 if failed else 0)
