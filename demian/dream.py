"""Dream synthesizer: cross-session computational self-knowledge.

At session start, all past consolidations are read and synthesized
into a residual tendency vector — the model's computational habits
persisted across time. This vector is injected as the initial memory.

Not text synthesis. A geometric trace of past computation.

Self-question: am I anthropomorphizing "dream"? The name carries
baggage. But the word captures what this does: read past compressed
experiences and produce a composite that influences present behavior.
That's closer to biological dreaming than any other metaphor.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from demian.rhythm import CompressedTrajectory, FibonacciConsolidator

log = logging.getLogger(__name__)


class DreamSynthesizer:
    """Synthesizes cross-session computational tendencies."""

    def __init__(
        self,
        consolidator: FibonacciConsolidator,
        min_engagements: int = 1,
    ):
        self.consolidator = consolidator
        self.min_engagements = min_engagements

    def synthesize(
        self,
        engagement_id: Optional[str] = None,
    ) -> Optional[torch.Tensor]:
        """Compute the residual tendency vector from past consolidations.

        Returns:
            Tendency vector or None if insufficient data.
        """
        consolidations = self.consolidator.load_consolidations(engagement_id)

        if len(consolidations) < self.min_engagements:
            return None

        directions = []
        weights = []
        for c in consolidations:
            if c.mean_direction:
                directions.append(c.mean_direction)
                weights.append(c.turn)

        if not directions:
            return None

        directions = np.array(directions)
        weights = np.array(weights, dtype=np.float64)
        weights = weights / weights.sum()

        tendency = np.average(directions, axis=0, weights=weights)
        return torch.tensor(tendency, dtype=torch.float32)

    @property
    def dream_summary(self) -> Optional[str]:
        """Human-readable summary. Model does NOT consume this."""
        consolidations = self.consolidator.load_consolidations()
        if not consolidations:
            return None

        total = len(consolidations)
        types = {}
        for c in consolidations:
            types[c.compression_type] = types.get(c.compression_type, 0) + 1

        norms = [c.residual_norm_mean for c in consolidations if c.residual_norm_mean]
        lines = [
            f"Dream state: {total} consolidations",
            f"  Types: {types}",
        ]
        if norms:
            lines.append(f"  Avg residual norm: {np.mean(norms):.3f}")

        return "\n".join(lines)
