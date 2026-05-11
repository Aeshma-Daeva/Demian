"""Current experiment metadata.

The active code still uses the v9 five-channel scaffold. Older artifacts named
``v10.0-frozen-evolution`` are predecessor evidence, not a separate
``DemianNativeV10Substrate`` class. New custom-substrate experiments should use
the ``demian-v*`` program names.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


NEXT_EXPERIMENT_ID = "demian-v1"
NEXT_EXPERIMENT_NAME = "Demian v1"
NEXT_EXPERIMENT_GOAL = (
    "First custom-substrate program built from v9 five-channel evidence, "
    "with sparse consequential release as the initial design target."
)

CURRENT_EVIDENCE_ID = "v10.0-frozen-evolution"
CURRENT_EVIDENCE_SUMMARY_PATH = Path(
    "data/evolution/v10_0_frozen_evolution_4island_20260510_summary.json"
)

CURRENT_EXPERIMENT_ID = NEXT_EXPERIMENT_ID
CURRENT_EXPERIMENT_SUMMARY_PATH = CURRENT_EVIDENCE_SUMMARY_PATH


@dataclass(frozen=True)
class CurrentExperiment:
    experiment_id: str
    experiment_name: str
    substrate_scaffold: str
    substrate_baseline: str
    predecessor_evidence_id: str
    summary_path: str
    eval_seed: int
    status: str
    caveat: str


CURRENT_EXPERIMENT = CurrentExperiment(
    experiment_id=CURRENT_EXPERIMENT_ID,
    experiment_name=NEXT_EXPERIMENT_NAME,
    substrate_scaffold="v9 five-channel message/carrier/release scaffold",
    substrate_baseline="demian_native_v9",
    predecessor_evidence_id=CURRENT_EVIDENCE_ID,
    summary_path=CURRENT_EXPERIMENT_SUMMARY_PATH.as_posix(),
    eval_seed=94,
    status="next named custom-substrate experiment; predecessor evidence loaded",
    caveat=(
        "The v10.0 artifact is evidence for Demian v1 design, not proof of a "
        "new substrate generation. Rank ordering is not seed-stable until "
        "held-out eval seeds are run."
    ),
)


def current_experiment_metadata() -> dict[str, Any]:
    """Return stable metadata for the current experiment line."""
    return asdict(CURRENT_EXPERIMENT)
