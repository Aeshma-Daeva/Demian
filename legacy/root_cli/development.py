"""Development track — the shape of the path, not just the endpoint.

Maintains a sparse, persistent record of reasoning patterns that survived
across conversations. Not memory of what was said — memory of *how*
understanding moved. Readable by new instances so they don't start from zero.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class ReasoningEpisode:
    """One moment where understanding moved somewhere new."""
    conversation_id: str
    turn_id: int
    human_input_summary: str  # what the human brought
    proprioceptive_signature: str  # what the system was computing
    thinker_signature: str  # what form the response took
    trajectory: str  # "wandering" | "narrowing" | "pivoting" | "deepening" | "dead_end"
    outcome_notes: str  # what happened, in human terms
    self_corrected: bool  # did the thinker change direction mid-turn
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class DevelopmentTrack:
    """The developmental record.

    New instances read this and start from where understanding last was.
    Not with the same experience — with the shape of it.
    """

    def __init__(self, data_dir: Path = Path("data/development")):
        self._dir = data_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._episodes: List[ReasoningEpisode] = []
        self._patterns: Dict[str, List[ReasoningEpisode]] = {}
        self._load()

    def _state_path(self) -> Path:
        return self._dir / "episodes.jsonl"

    def _patterns_path(self) -> Path:
        return self._dir / "patterns.json"

    def _load(self):
        if self._state_path().exists():
            for line in self._state_path().read_text().splitlines():
                if line.strip():
                    self._episodes.append(ReasoningEpisode(**json.loads(line)))
            log.info("Loaded %d development episodes", len(self._episodes))

        if self._patterns_path().exists():
            raw = json.loads(self._patterns_path().read_text())
            self._patterns = raw
            log.info("Loaded %d tracked patterns", len(self._patterns))

    def log_episode(self, ep: ReasoningEpisode):
        self._episodes.append(ep)
        with open(self._state_path(), "a") as f:
            f.write(json.dumps(ep.__dict__, ensure_ascii=False) + "\n")
        self._update_patterns(ep)

    def _update_patterns(self, ep: ReasoningEpisode):
        """Track which reasoning trajectories recur."""
        traj = ep.trajectory
        self._patterns.setdefault(traj, []).append(ep.__dict__)

        # Track proprioceptive signature frequencies
        sigs = ep.proprioceptive_signature
        if "routing:" in sigs:
            self._patterns.setdefault("routing_patterns", [])
        if "boundaries:" in sigs:
            self._patterns.setdefault("boundary_frequency", []).append(ep.timestamp)

    def summary(self) -> dict:
        """Return developmental summary for injection into new conversations."""
        if not self._episodes:
            return {"status": "no_development_recorded", "total_episodes": 0}

        trajectories = {}
        for ep in self._episodes:
            trajectories[ep.trajectory] = trajectories.get(ep.trajectory, 0) + 1

        total = len(self._episodes)
        deepening = trajectories.get("deepening", 0)
        dead_ends = trajectories.get("dead_end", 0)

        return {
            "total_episodes": total,
            "trajectories": trajectories,
            "deepening_rate": deepening / total if total > 0 else 0,
            "dead_end_rate": dead_ends / total if total > 0 else 0,
            "latest": self._episodes[-1].__dict__ if self._episodes else None,
        }

    def format_for_thinker(self) -> str:
        """Format developmental record as context the thinker can read.

        Not a command. Information. What understanding has moved before.
        """
        s = self.summary()
        if s["total_episodes"] == 0:
            return "[development: no prior record]"

        lines = [
            f"[development: {s['total_episodes']} episodes recorded]",
            "  Trajectory distribution:",
        ]
        for traj, count in sorted(s["trajectories"].items(), key=lambda x: -x[1]):
            lines.append(f"    {traj}: {count}")

        lines.append(
            f"  Deepening rate: {s['deepening_rate']:.0%} | "
            f"Dead-end rate: {s['dead_end_rate']:.0%}"
        )

        # Last episode — where understanding last left off
        last = s["latest"]
        if last:
            lines.extend([
                "  Last engagement:",
                f"    Human brought: {last['human_input_summary'][:100]}",
                f"    System was: {last['trajectory']}",
                f"    Outcome: {last['outcome_notes'][:100]}",
            ])

        lines.append("[/development]")
        return "\n".join(lines)

    def save(self):
        with open(self._patterns_path(), "w") as f:
            json.dump(self._patterns, f, indent=2, ensure_ascii=False)
