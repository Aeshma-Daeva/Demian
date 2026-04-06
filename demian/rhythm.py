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
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Optional

import numpy as np


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

        modes = [s.attention.mode for s in snapshots]
        attn_transitions = self._markov_chain(modes)

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
            )
        else:
            # Essence: PCA dominant direction + Markov chain of attention modes
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
            )

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
