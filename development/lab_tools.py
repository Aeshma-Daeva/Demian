"""Research-lab adapters for tracking, optimization, and searchable exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import optuna
import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.tensorboard import SummaryWriter


def numeric_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Return metrics that experiment trackers can store as scalar values."""
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }


def create_study(study_name: str, storage_path: Path, *, direction: str = "maximize") -> optuna.Study:
    """Create or reopen a local SQLite-backed Optuna study."""
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    return optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{storage_path}",
        direction=direction,
        load_if_exists=True,
    )


def log_candidate_to_mlflow(
    candidate: dict[str, Any],
    *,
    tracking_dir: Path,
    experiment_name: str,
) -> str:
    """Log a candidate's scalar metrics and compact metadata to local MLflow tracking."""
    tracking_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = tracking_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{tracking_dir / 'mlflow.sqlite3'}")
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=artifact_dir.as_uri())
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=str(candidate.get("id", "candidate"))) as run:
        mlflow.log_params(
            {
                "candidate_id": str(candidate.get("id", "")),
                "generation": int(candidate.get("generation", 0)),
                "reproduction_kind": str(candidate.get("reproduction_kind", "")),
            }
        )
        for key, value in numeric_metrics(candidate.get("metrics", {})).items():
            mlflow.log_metric(key, value)
        mlflow.log_text(
            json.dumps(
                {
                    "parent_ids": candidate.get("parent_ids", []),
                    "ancestor_ids": candidate.get("ancestor_ids", []),
                },
                indent=2,
                sort_keys=True,
            ),
            "lineage.json",
        )
        return run.info.run_id


def write_tensorboard_metrics(metrics: dict[str, Any], log_dir: Path, *, step: int = 0) -> None:
    """Write scalar metrics to a TensorBoard event file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(log_dir)) as writer:
        for key, value in numeric_metrics(metrics).items():
            writer.add_scalar(key, value, global_step=step)


def flatten_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    row = {
        "id": candidate.get("id"),
        "generation": candidate.get("generation"),
        "rank_score": candidate.get("rank_score"),
        "reproduction_kind": candidate.get("reproduction_kind"),
    }
    for key, value in numeric_metrics(candidate.get("metrics", {})).items():
        row[f"metric_{key}"] = value
    return row


def write_candidates_parquet(candidates: list[dict[str, Any]], out_path: Path) -> None:
    """Write flattened candidate metrics to Parquet for fast cross-run search."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [flatten_candidate(candidate) for candidate in candidates]
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_path)
