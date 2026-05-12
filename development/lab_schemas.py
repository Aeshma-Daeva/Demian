"""Pydantic contracts for Demian research-lab artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, NonNegativeInt, field_validator, model_validator

ArtifactKind = Literal["candidate", "evolution_config", "evolution_summary"]


class FlexibleArtifact(BaseModel):
    """Base model that validates known fields while preserving older artifact extras."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())


class Metrics(FlexibleArtifact):
    """Shared metric surface for candidate and summary rows."""

    internal_richness: float | None = None
    channel_separation: float | None = None
    release_duty_cycle: NonNegativeFloat | None = None
    release_geometric_event: float | None = None
    release_causal_divergence: float | None = None
    release_gain_zero_release_causal_divergence: float | None = None
    release_timing_score: float | None = None
    phase_transition_score: float | None = None
    mathematical_curiosity: float | None = None
    geometric_coherence: float | None = None
    regime_bonus: float | None = None
    regimes: dict[str, int] = Field(default_factory=dict)

    @field_validator("release_duty_cycle")
    @classmethod
    def duty_is_fraction(cls, value: float | None) -> float | None:
        if value is not None and value > 1.0:
            raise ValueError("release_duty_cycle must be a fraction in [0, 1]")
        return value


class CandidateArtifact(FlexibleArtifact):
    """Single evolved candidate JSON artifact."""

    id: str
    generation: NonNegativeInt
    index: NonNegativeInt | None = None
    genome: dict[str, Any]
    metrics: Metrics
    reproduction_kind: str
    parent_ids: list[str] = Field(default_factory=list)
    ancestor_ids: list[str] = Field(default_factory=list)
    rank_score: float | None = None
    rank_components: dict[str, float] = Field(default_factory=dict)
    runs: list[dict[str, Any]] = Field(default_factory=list)
    paired_runs: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def candidate_id_matches_generation(self) -> CandidateArtifact:
        if self.id.startswith("gen"):
            prefix = f"gen{int(self.generation):03d}_"
            if not self.id.startswith(prefix):
                raise ValueError(f"candidate id {self.id!r} does not match generation {self.generation}")
        return self


class EvolutionConfigArtifact(FlexibleArtifact):
    """Configuration emitted by evolution scripts."""

    population: NonNegativeInt
    generations: NonNegativeInt
    seed: int
    eval_seeds: list[int]
    hidden_size: NonNegativeInt
    steps: NonNegativeInt
    perturb_step: NonNegativeInt
    perturb_scales: list[float]
    rank: NonNegativeInt
    experiment_version: str

    @model_validator(mode="after")
    def perturb_step_within_run(self) -> EvolutionConfigArtifact:
        if self.perturb_step > self.steps:
            raise ValueError("perturb_step must be <= steps")
        return self


class EvolutionSummaryArtifact(FlexibleArtifact):
    """Cross-island or single-run summary JSON."""

    experiment: str
    candidate_count: NonNegativeInt
    generation_count: NonNegativeInt
    final_generation: NonNegativeInt | None = None
    best_overall: dict[str, Any] | None = None
    final_best: dict[str, Any] | None = None
    per_generation: list[dict[str, Any]] = Field(default_factory=list)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def infer_artifact_kind(path: Path, payload: Any) -> ArtifactKind:
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object artifact")
    if "genome" in payload and "metrics" in payload and "id" in payload:
        return "candidate"
    if "population" in payload and "generations" in payload and "eval_seeds" in payload:
        return "evolution_config"
    if "experiment" in payload and "candidate_count" in payload and "generation_count" in payload:
        return "evolution_summary"
    raise ValueError(f"cannot infer artifact kind for {path}")


def validate_artifact(path: Path, *, kind: ArtifactKind | None = None) -> FlexibleArtifact:
    payload = load_json(path)
    artifact_kind = kind or infer_artifact_kind(path, payload)
    if artifact_kind == "candidate":
        return CandidateArtifact.model_validate(payload)
    if artifact_kind == "evolution_config":
        return EvolutionConfigArtifact.model_validate(payload)
    if artifact_kind == "evolution_summary":
        return EvolutionSummaryArtifact.model_validate(payload)
    raise ValueError(f"unknown artifact kind: {artifact_kind}")
