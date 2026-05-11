"""Fibonacci rhythm engine: compression at natural intervals.

At F(n) turns, the raw signal trajectory is compressed into
geometric primitives. The compressed form captures the shape
of what thinking felt like during those turns.

Compression levels:
  Turn 1-3 ("raw"): Keep full projected state snapshots
  Turn 5-8 ("pattern"): Extract mean direction, variance, attention transitions
  Turn 13+ ("essence"): Further compress to attractor states and Markov chains

Self-question: why Fibonacci? It's a sub-linear growing-gaps schedule.
Early turns need frequent compression — the model doesn't know its
computational shape yet. Later, the shape has been established.
Fibonacci is one of many valid schedules. Worth trying: power-law,
adaptive, logarithmic.

The golden ratio (phi) also drives injection rhythm within a single
generation turn — see InjectionScheduler below. An irrational pulse
keeps the model from settling into mechanical prediction patterns.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Generator, List, Optional

import numpy as np

# The golden ratio — irrational, non-repeating
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2  # approximately 1.618033988749895


def generate_fibonacci_schedule(max_steps: int = 256, base_gap: int = 2, max_gap: int = 13) -> set[int]:
    """Generate a Fibonacci-spaced injection schedule.

    Injection occurs at steps separated by growing Fibonacci intervals:
    2, 3, 5, 8, 13, then cap at max_gap and repeat. The non-repeating
    rhythm means the model cannot settle into a mechanical pattern
    around when injection will occur.

    Args:
        max_steps: total generation steps to plan for
        base_gap: starting gap before first injection
        max_gap: largest gap before schedule loops

    Returns:
        Set of step indices (0-based) where injection should occur.
    """
    # Build Fibonacci sequence up to max_gap
    fib = [1, base_gap]
    while fib[-1] + fib[-2] <= max_gap:
        fib.append(fib[-1] + fib[-2])
    fib = fib[1:]  # skip initial 1

    schedule: set[int] = set()
    step = 0
    fib_idx = 0
    while step < max_steps:
        gap = fib[fib_idx % len(fib)]
        step += gap
        if step < max_steps and step > 0:
            schedule.add(step)
        fib_idx += 1
    return schedule


class InjectionScheduler:
    """Decides WHEN to inject proprioceptive feedback.

    The schedule follows Fibonacci intervals: early steps get dense
    injection (the model is orienting to its new input), later steps
    get breathing room. A full cycle repeats every N steps.

    This replaces the old behaviour of injecting every single step.
    """

    def __init__(
        self,
        max_steps: int = 256,
        base_gap: int = 2,
        max_gap: int = 13,
    ):
        self._schedule = generate_fibonacci_schedule(max_steps, base_gap, max_gap)

    def should_inject(self, step: int) -> bool:
        return step in self._schedule


@dataclass
class CompressedTrajectory:
    turn: int
    mean_direction: List[float]
    variance: List[float]
    attention_transitions: Any
    residual_norm_mean: float
    residual_norm_std: float
    temporal_coherence_mean: float
    trajectory_length: int
    compression_type: str
    raw_snapshots: Optional[List[List[float]]] = None
    raw_residual_mean: Optional[List[float]] = None


class FibonacciConsolidator:
    def __init__(
        self,
        tracker,
        storage_dir: str | Path = "data/consolidations",
        target_dim: int = 128,
        fibonacci_intervals: List[int] | None = None,
    ):
        self.tracker = tracker
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.target_dim = target_dim
        self._turn_count = 0
        self._consolidated: List[CompressedTrajectory] = []
        self._fibonacci_intervals = frozenset(
            fibonacci_intervals if fibonacci_intervals is not None
            else [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        )

    def on_turn(self, engagement_id: Optional[str] = None) -> Optional[CompressedTrajectory]:
        self._turn_count += 1

        if self._turn_count not in self._fibonacci_intervals:
            return None

        snapshots = self.tracker.get_trajectory()
        if not snapshots:
            return None

        compressed = self._consolidate(snapshots)
        self._consolidated.append(compressed)

        if engagement_id:
            self._save(compressed, engagement_id)

        return compressed

    def _consolidate(self, snapshots) -> CompressedTrajectory:
        turn = self._turn_count
        projected_states = np.array([s.projected_state for s in snapshots])

        # Structural state sequence: bin spectral_centroid into 4 regions
        centroids = [s.shape.spectral_centroid for s in snapshots]
        centroid_bins = self._bin_centroids(centroids)
        attn_transitions = self._markov_chain(centroid_bins)

        # Legacy: also store mode sequence from old code if available
        modes = [getattr(s, 'attention', s.shape).mode if hasattr(getattr(s, 'attention', s.shape), 'mode')
                 else None for s in snapshots]

        raw_residual_mean = self._mean_raw_residual(snapshots)

        if turn <= 3:
            return CompressedTrajectory(
                turn=turn,
                mean_direction=projected_states.mean(axis=0).tolist(),
                variance=projected_states.var(axis=0).tolist(),
                attention_transitions=attn_transitions,
                residual_norm_mean=np.mean([s.residual_norm for s in snapshots]),
                residual_norm_std=np.std([s.residual_norm for s in snapshots]),
                temporal_coherence_mean=np.mean([s.temporal_coherence for s in snapshots]),
                trajectory_length=len(snapshots),
                compression_type="raw",
                raw_snapshots=[s.projected_state for s in snapshots],
                raw_residual_mean=raw_residual_mean,
            )
        elif turn <= 8:
            return CompressedTrajectory(
                turn=turn,
                mean_direction=projected_states.mean(axis=0).tolist(),
                variance=projected_states.var(axis=0).tolist(),
                attention_transitions=attn_transitions,
                residual_norm_mean=np.mean([s.residual_norm for s in snapshots]),
                residual_norm_std=np.std([s.residual_norm for s in snapshots]),
                temporal_coherence_mean=np.mean([s.temporal_coherence for s in snapshots]),
                trajectory_length=len(snapshots),
                compression_type="pattern",
                raw_residual_mean=raw_residual_mean,
            )
        else:
            # Essence: PCA dominant direction + Markov chain of spectral regions
            centered = projected_states - projected_states.mean(axis=0)
            if len(centered) > 1:
                u, s, vt = np.linalg.svd(centered, full_matrices=False)
                dominant = vt[0].tolist()
            else:
                dominant = projected_states[0].tolist()

            return CompressedTrajectory(
                turn=turn,
                mean_direction=dominant,
                variance=projected_states.var(axis=0).tolist(),
                attention_transitions=attn_transitions,
                residual_norm_mean=np.mean([s.residual_norm for s in snapshots]),
                residual_norm_std=np.std([s.residual_norm for s in snapshots]),
                temporal_coherence_mean=np.mean([s.temporal_coherence for s in snapshots]),
                trajectory_length=len(snapshots),
                compression_type="essence",
                raw_residual_mean=raw_residual_mean,
            )

    def _bin_centroids(self, centroids: List[float]) -> List[str]:
        """Bin continuous spectral_centroid into 4 structural regions.

        Low (0-0.25): energy concentrated in low-frequency modes of residual
        Mid-low (0.25-0.5): mid-frequency structure
        Mid-high (0.5-0.75): high-frequency structure
        High (0.75-1.0): very high-frequency, rapid oscillation

        Not human labels — these are bins on a measurable axis.
        """
        result = []
        for c in centroids:
            if c < 0.25:
                result.append("low")
            elif c < 0.5:
                result.append("mid_low")
            elif c < 0.75:
                result.append("mid_high")
            else:
                result.append("high")
        return result

    def _mean_raw_residual(self, snapshots) -> Optional[List[float]]:
        """Average of raw residual vectors in the trajectory.

        This is a d_model-dimensional vector that can be injected
        directly into the KV cache without dimension mismatch.
        """
        residuals = [s.raw_residual for s in snapshots if s.raw_residual]
        if not residuals:
            return None
        return np.mean(np.array(residuals), axis=0).tolist()

    def _markov_chain(self, modes: List[str]) -> dict:
        unique_modes = sorted(set(modes))
        mode_index = {m: i for i, m in enumerate(unique_modes)}
        n = len(unique_modes)
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(len(modes) - 1):
            fi = mode_index[modes[i]]
            ti = mode_index[modes[i + 1]]
            matrix[fi][ti] += 1
        for row in matrix:
            s = sum(row)
            if s > 0:
                for i in range(len(row)):
                    row[i] /= s
        return {
            unique_modes[i]: {unique_modes[j]: matrix[i][j] for j in range(n) if matrix[i][j] > 0}
            for i in range(n) if any(matrix[i])
        }

    def _save(self, compressed: CompressedTrajectory, engagement_id: str):
        data = asdict(compressed)
        fname = self.storage_dir / f"consolidation_{engagement_id}_turn{compressed.turn}.json"
        fname.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def load_consolidations(self, engagement_id: Optional[str] = None) -> List[CompressedTrajectory]:
        pattern = f"consolidation_{engagement_id}_" if engagement_id else "consolidation_"
        files = sorted(self.storage_dir.glob(f"{pattern}*.json"))
        result = []
        for f in files:
            data = json.loads(f.read_text())
            result.append(CompressedTrajectory(**data))
        return result

    @property
    def turn_count(self) -> int:
        return self._turn_count

    def reset(self):
        self._turn_count = 0
