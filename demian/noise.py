"""Random projection: blind dimensionality reduction.

The projection matrix is generated ONCE per model dimensionality,
stored, and never changed. Not learned, not human-selected.
A random orthogonal matrix — every direction is equally likely.

Johnson-Lindenstrauss: pairwise distances in projected space approximate
distances in original space within epsilon with high probability.
"""
import numpy as np
import torch
from pathlib import Path


def generate_projection(
    d_model: int,
    target_dim: int = 128,
    seed: int | None = None,
) -> torch.Tensor:
    """Generate a random projection matrix via QR decomposition.

    Args:
        d_model: source dimensionality (model hidden_size)
        target_dim: projected dimensionality
        seed: optional seed for reproducibility

    Returns:
        Projection matrix of shape (target_dim, d_model)
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    # Random Gaussian matrix with normalized rows
    # Each row is a random unit vector in d_model dimensions
    M = rng.randn(target_dim, d_model)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    M = M / norms

    # Scale to preserve expected norm under projection
    scale = np.sqrt(d_model / target_dim)
    M *= scale

    return torch.tensor(M, dtype=torch.float32)


def load_or_create_projection(
    d_model: int,
    target_dim: int,
    cache_dir: str | Path = "data/projections",
) -> torch.Tensor:
    """Load a saved projection or generate a new one.

    Once you project with matrix A, you always use matrix A.
    Otherwise the meaning of projected dimensions changes mid-flight.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = cache_dir / f"proj_{d_model}_{target_dim}.pt"

    if fname.exists():
        return torch.load(fname, weights_only=True)

    proj = generate_projection(d_model, target_dim)
    torch.save(proj, fname)
    return proj
