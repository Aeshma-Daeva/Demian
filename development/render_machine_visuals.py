#!/usr/bin/env python3
"""Render machine-native diagnostic visuals from trajectory_3d.json exports."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_INPUT = Path("data/substrate_lab/v9_release_gate_trajectory_3d_20260509/trajectory_3d.json")
DEFAULT_OUT_DIR = Path("docs/assets/machine_visuals")
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

CORE_DELTA_METRICS = (
    "residual_delta",
    "fast_state_delta",
    "slow_state_delta",
    "message_state_delta",
    "route_metrics.release_pressure_mean",
    "route_metrics.release_open_mean",
    "route_metrics.release_strength_mean",
    "route_metrics.carrier_residual_norm",
)
EVENT_METRICS = (
    "route_metrics.release_pressure_mean",
    "route_metrics.release_open_mean",
    "route_metrics.release_strength_mean",
    "residual_delta",
    "route_metrics.carrier_residual_norm",
    "release_local_displacement",
    "release_local_signature_push",
)
PHASE_PAIRS = (
    ("route_metrics.release_pressure_mean", "route_metrics.carrier_residual_norm"),
    ("route_metrics.release_pressure_mean", "residual_delta"),
    ("fast_state_norm", "slow_state_norm"),
)
PARETO_METRICS = (
    "internal_richness",
    "release_duty_cycle",
    "geometric_coherence",
    "phase_transition_score",
)
COMPARISON_FIELDS = (
    "substrate",
    "candidate_id",
    "seed",
    "run_kind",
    "perturb_scale",
    "perturb_family",
    "motif_index",
    "release_duty_fraction",
    "release_strength_max",
    "release_pressure_max",
    "carrier_residual_final",
    "residual_delta_max",
    "post_event_residual_delta_max",
    "pressure_to_post_delta_ratio",
)


def load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def joined_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    metrics_by_id = {row["point_id"]: row for row in payload.get("metrics", [])}
    records = []
    for point in payload.get("points", []):
        metric_row = metrics_by_id.get(point["point_id"], {})
        records.append({**point, "metrics": metric_row})
    return sorted(records, key=record_sort_key)


def record_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    scale = record.get("perturb_scale")
    scale_value = -1.0 if scale is None else float(scale)
    return (
        str(record.get("substrate", "")),
        str(record.get("candidate_id", "")),
        int(record.get("seed", 0)),
        str(record.get("run_kind", "")),
        scale_value,
        str(record.get("perturb_family", "")),
        int(record.get("motif_index", 0) or 0),
        int(record.get("step", 0)),
    )


def run_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("substrate"),
        record.get("candidate_id"),
        record.get("seed"),
        record.get("run_kind"),
        record.get("perturb_scale"),
        record.get("perturb_family"),
        record.get("motif_index"),
    )


def group_runs(records: list[dict[str, Any]]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[run_key(record)].append(record)
    return {key: sorted(rows, key=lambda row: int(row.get("step", 0))) for key, rows in grouped.items()}


def filter_records(records: list[dict[str, Any]], run_filter: str, seed: str) -> list[dict[str, Any]]:
    filtered = records
    if run_filter != "all":
        filtered = [
            row
            for row in filtered
            if row.get("substrate") == run_filter or row.get("candidate_id") == run_filter
        ]
    if seed != "all":
        seed_value = int(seed)
        filtered = [row for row in filtered if int(row.get("seed", -1)) == seed_value]
    return filtered


def representative_run(runs: dict[tuple[Any, ...], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    if not runs:
        return []

    def score(item: tuple[tuple[Any, ...], list[dict[str, Any]]]) -> tuple[Any, ...]:
        key, rows = item
        first = rows[0]
        scale = first.get("perturb_scale")
        scale_value = -1.0 if scale is None else float(scale)
        is_perturbed = 1 if first.get("run_kind") == "perturbed" else 0
        return (is_perturbed, scale_value, -int(first.get("seed", 0)), tuple(str(part) for part in key))

    return max(runs.items(), key=score)[1]


def run_summary(rows: list[dict[str, Any]], events: list[dict[str, Any]], *, window: int = 12) -> dict[str, Any]:
    first = rows[0]
    release_open = metric_series(rows, "route_metrics.release_open_mean")
    release_strength = metric_series(rows, "route_metrics.release_strength_mean")
    release_pressure = metric_series(rows, "route_metrics.release_pressure_mean")
    carrier_residual = metric_series(rows, "route_metrics.carrier_residual_norm")
    residual_delta = metric_series(rows, "residual_delta")
    event_window = min(window, max(0, (len(rows) - 1) // 2))
    indexes = event_indices(rows, events, event_window) if event_window else []
    post_event_max = 0.0
    if residual_delta.size and indexes:
        chunks = [residual_delta[index : min(len(rows), index + window + 1)] for index in indexes]
        post_event_max = max((float(np.max(chunk)) for chunk in chunks if chunk.size), default=0.0)
    pressure_max = float(np.max(release_pressure)) if release_pressure.size else 0.0
    return {
        "substrate": first.get("substrate"),
        "candidate_id": first.get("candidate_id"),
        "seed": first.get("seed"),
        "run_kind": first.get("run_kind"),
        "perturb_scale": first.get("perturb_scale"),
        "perturb_family": first.get("perturb_family"),
        "motif_index": first.get("motif_index"),
        "release_duty_fraction": float(np.mean(release_strength > 1e-12)) if release_strength.size else 0.0,
        "release_strength_max": float(np.max(release_strength)) if release_strength.size else 0.0,
        "release_pressure_max": pressure_max,
        "carrier_residual_final": float(carrier_residual[-1]) if carrier_residual.size else 0.0,
        "residual_delta_max": float(np.max(residual_delta)) if residual_delta.size else 0.0,
        "post_event_residual_delta_max": post_event_max,
        "pressure_to_post_delta_ratio": pressure_max / max(post_event_max, 1e-9),
    }


def comparison_rows(
    runs: dict[tuple[Any, ...], list[dict[str, Any]]],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    events_by_point = {event.get("point_id"): event for event in events}
    rows = []
    for run_rows in runs.values():
        run_events = [
            events_by_point[row["point_id"]]
            for row in run_rows
            if row["point_id"] in events_by_point
        ]
        rows.append(run_summary(run_rows, run_events))
    rows.sort(
        key=lambda row: (
            str(row["substrate"]),
            int(row["seed"] or 0),
            str(row["run_kind"]),
            -1.0 if row["perturb_scale"] is None else float(row["perturb_scale"]),
            str(row["perturb_family"]),
            int(row["motif_index"] or 0),
        )
    )
    return rows


def metric_value(record: dict[str, Any], metric_path: str) -> float | None:
    current: Any = record.get("metrics", {})
    for part in metric_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    try:
        value = float(current)
    except (TypeError, ValueError):
        return None
    if math.isfinite(value):
        return value
    return None


def available_metrics(rows: list[dict[str, Any]], metric_paths: tuple[str, ...]) -> list[str]:
    available = []
    for metric_path in metric_paths:
        if any(metric_value(row, metric_path) is not None for row in rows):
            available.append(metric_path)
    return available


def metric_series(rows: list[dict[str, Any]], metric_path: str) -> np.ndarray:
    values = [metric_value(row, metric_path) for row in rows]
    return np.asarray([0.0 if value is None else value for value in values], dtype=float)


def normalize_columns(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    safe_stds = np.where(stds <= 1e-12, 1.0, stds)
    return (matrix - means) / safe_stds


def correlation_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    normalized = normalize_columns(matrix)
    denom = max(1, normalized.shape[0] - 1)
    corr = (normalized.T @ normalized) / float(denom)
    corr = np.clip(corr, -1.0, 1.0)
    return np.nan_to_num(corr, nan=0.0)


def recurrence_matrix(rows: list[dict[str, Any]], metric_paths: list[str]) -> np.ndarray:
    vectors = np.column_stack([metric_series(rows, metric) for metric in metric_paths])
    vectors = normalize_columns(vectors)
    diff = vectors[:, None, :] - vectors[None, :, :]
    return np.linalg.norm(diff, axis=2)


def row_scaled_matrix(rows: list[dict[str, Any]], metric_paths: list[str]) -> np.ndarray:
    matrix = np.vstack([metric_series(rows, metric) for metric in metric_paths])
    scaled = []
    for row in matrix:
        max_abs = float(np.max(np.abs(row))) if row.size else 0.0
        scaled.append(row / max_abs if max_abs > 1e-12 else row)
    return np.asarray(scaled)


def event_indices(rows: list[dict[str, Any]], events: list[dict[str, Any]], window: int) -> list[int]:
    point_to_index = {row["point_id"]: index for index, row in enumerate(rows)}
    indices = [
        point_to_index[event["point_id"]]
        for event in events
        if event.get("point_id") in point_to_index and event.get("event_kind") == "perturbation"
    ]
    pressure = metric_series(rows, "route_metrics.release_pressure_mean")
    if pressure.size and float(np.max(pressure)) > 1e-12:
        indices.append(int(np.argmax(pressure)))
    usable = sorted({index for index in indices if window <= index < len(rows) - window})
    return usable


def collect_event_windows(
    rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
    metric_paths: list[str],
    window: int,
) -> dict[str, np.ndarray]:
    indices = event_indices(rows, events, window)
    windows: dict[str, np.ndarray] = {}
    for metric_path in metric_paths:
        series = metric_series(rows, metric_path)
        chunks = [series[index - window : index + window + 1] for index in indices]
        if chunks:
            windows[metric_path] = np.vstack(chunks)
    return windows


def _import_matplotlib() -> tuple[Any, Any, Any]:
    mpl_config_dir = Path(tempfile.gettempdir()) / "demian-matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm

    return plt, PowerNorm, matplotlib


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    sidecar = path.with_suffix(path.suffix + ".json")
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_skipped(path: Path, reason: str, context: dict[str, Any]) -> None:
    write_metadata(path, {"status": "skipped", "reason": reason, **context})


def write_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in COMPARISON_FIELDS})


def render_event_aligned_release_windows(
    rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
    path: Path,
    context: dict[str, Any],
    *,
    dpi: int,
    window: int = 12,
) -> None:
    metric_paths = available_metrics(rows, EVENT_METRICS)
    windows = collect_event_windows(rows, events, metric_paths, window)
    if not windows:
        save_skipped(path, "No usable perturbation or pressure-peak event windows.", context)
        return
    plt, _power_norm, _matplotlib = _import_matplotlib()
    offsets = np.arange(-window, window + 1)
    fig, ax = plt.subplots(figsize=(10.0, 5.2), constrained_layout=True)
    for metric_path, matrix in windows.items():
        mean = matrix.mean(axis=0)
        max_abs = float(np.max(np.abs(mean)))
        if max_abs > 1e-12:
            mean = mean / max_abs
        ax.plot(offsets, mean, label=metric_path.replace("route_metrics.", ""))
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.55)
    ax.set_title("Event-aligned release windows")
    ax.set_xlabel("step offset")
    ax.set_ylabel("mean, normalized per metric")
    ax.legend(loc="best", fontsize=8)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    write_metadata(path, {"status": "rendered", "window": window, "metrics": list(windows), **context})


def render_release_variant_comparison(
    rows: list[dict[str, Any]],
    path: Path,
    context: dict[str, Any],
    *,
    dpi: int,
) -> None:
    if not rows:
        save_skipped(path, "No runs available for release variant comparison.", context)
        return
    csv_path = path.with_suffix(".csv")
    write_comparison_csv(csv_path, rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("run_kind") == "perturbed":
            grouped[str(row.get("substrate"))].append(row)
    if not grouped:
        save_skipped(path, "No perturbed runs available for release variant comparison.", context)
        return
    variants = sorted(grouped)
    metrics = (
        "release_duty_fraction",
        "release_strength_max",
        "release_pressure_max",
        "post_event_residual_delta_max",
        "pressure_to_post_delta_ratio",
    )
    matrix = np.asarray(
        [
            [float(np.mean([row[metric] for row in grouped[variant]])) for variant in variants]
            for metric in metrics
        ],
        dtype=float,
    )
    row_scaled = []
    for row in matrix:
        max_value = float(np.max(np.abs(row))) if row.size else 0.0
        row_scaled.append(row / max_value if max_value > 1e-12 else row)
    display = np.asarray(row_scaled)

    plt, _power_norm, _matplotlib = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(max(8.5, len(variants) * 1.1), 4.8), constrained_layout=True)
    image = ax.imshow(display, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title("Release variant comparison, mean over seeds and perturb scales")
    ax.set_xticks(range(len(variants)), variants, rotation=35, ha="right")
    ax.set_yticks(range(len(metrics)), metrics)
    fig.colorbar(image, ax=ax, shrink=0.78, label="row-scaled mean")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    write_metadata(
        path,
        {
            "status": "rendered",
            "metrics": list(metrics),
            "variants": variants,
            "normalization": "row_scaled_mean_over_perturbed_runs",
            "csv": str(csv_path),
            **context,
        },
    )


def render_state_delta_heatmap(rows: list[dict[str, Any]], path: Path, context: dict[str, Any], *, dpi: int) -> None:
    metric_paths = available_metrics(rows, CORE_DELTA_METRICS)
    if not metric_paths:
        save_skipped(path, "No delta metrics found.", context)
        return
    matrix = row_scaled_matrix(rows, metric_paths)
    plt, power_norm, _matplotlib = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(10.0, 4.8), constrained_layout=True)
    image = ax.imshow(
        np.abs(matrix),
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        norm=power_norm(gamma=0.7, vmin=0.0, vmax=1.0),
    )
    ax.set_title("State delta heatmap, row-scaled")
    ax.set_xlabel("step")
    ax.set_yticks(range(len(metric_paths)), [metric.replace("route_metrics.", "") for metric in metric_paths])
    fig.colorbar(image, ax=ax, shrink=0.78, label="within-row absolute fraction")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    write_metadata(path, {"status": "rendered", "metrics": metric_paths, "normalization": "row_abs_max", **context})


def render_recurrence_distance(rows: list[dict[str, Any]], path: Path, context: dict[str, Any], *, dpi: int) -> None:
    metric_paths = available_metrics(rows, (*CORE_DELTA_METRICS, "fast_state_norm", "slow_state_norm", "message_state_norm"))
    if len(rows) < 2 or not metric_paths:
        save_skipped(path, "Need at least two steps and one metric for recurrence.", context)
        return
    matrix = recurrence_matrix(rows, metric_paths)
    plt, _power_norm, _matplotlib = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
    image = ax.imshow(matrix, aspect="equal", interpolation="nearest", cmap="viridis")
    ax.set_title("Recurrence distance")
    ax.set_xlabel("step")
    ax.set_ylabel("step")
    fig.colorbar(image, ax=ax, shrink=0.78, label="normalized metric distance")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    write_metadata(path, {"status": "rendered", "metrics": metric_paths, **context})


def render_channel_separation_covariance(
    rows: list[dict[str, Any]], path: Path, context: dict[str, Any], *, dpi: int
) -> None:
    metric_paths = available_metrics(
        rows,
        (
            "fast_state_norm",
            "slow_state_norm",
            "message_state_norm",
            "route_metrics.carrier_residual_norm",
            "route_metrics.fast_update_mean",
            "route_metrics.slow_write_mean",
            "route_metrics.message_write_mean",
            "route_metrics.release_pressure_mean",
        ),
    )
    if len(metric_paths) < 2:
        save_skipped(path, "Need at least two channel metrics for covariance.", context)
        return
    matrix = np.column_stack([metric_series(rows, metric) for metric in metric_paths])
    corr = correlation_matrix(matrix)
    plt, _power_norm, _matplotlib = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(7.2, 6.4), constrained_layout=True)
    image = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="RdBu_r", interpolation="nearest")
    labels = [metric.replace("route_metrics.", "") for metric in metric_paths]
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("Channel separation correlation")
    fig.colorbar(image, ax=ax, shrink=0.78, label="Pearson r")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    write_metadata(path, {"status": "rendered", "metrics": metric_paths, **context})


def summary_metric_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows_by_key: dict[str, dict[str, Any]] = {}
    for item in payload.get("summaries", {}).get("runs", []):
        metrics = item.get("metrics") or item.get("summary") or {}
        row = {**item, "metrics": metrics}
        key = str(row.get("candidate_id") or row.get("substrate") or len(rows_by_key))
        rows_by_key[key] = row
    candidate_metrics = payload.get("metadata", {}).get("candidate_metrics") or {}
    for candidate_id, metrics in candidate_metrics.items():
        rows_by_key.setdefault(str(candidate_id), {"candidate_id": candidate_id, "metrics": metrics})
    return [rows_by_key[key] for key in sorted(rows_by_key)]


def summary_metric_value(row: dict[str, Any], metric: str) -> float | None:
    value = row.get("metrics", {}).get(metric)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def first_summary_metric_value(row: dict[str, Any], metrics: tuple[str, ...]) -> float | None:
    for metric in metrics:
        value = summary_metric_value(row, metric)
        if value is not None:
            return value
    return None


def render_sweep_surface(payload: dict[str, Any], path: Path, context: dict[str, Any], *, dpi: int) -> None:
    rows = summary_metric_rows(payload)
    usable = [
        row
        for row in rows
        if summary_metric_value(row, "release_duty_cycle") is not None
        and first_summary_metric_value(row, ("release_geometric_event", "release_effectiveness")) is not None
    ]
    if not usable:
        save_skipped(path, "No candidate summary release/geometric metrics found.", context)
        return
    usable.sort(key=lambda row: str(row.get("candidate_id", row.get("substrate", ""))))
    matrix = np.asarray(
        [
            [summary_metric_value(row, "release_duty_cycle") or 0.0 for row in usable],
            [
                first_summary_metric_value(row, ("release_geometric_event", "release_effectiveness")) or 0.0
                for row in usable
            ],
            [
                first_summary_metric_value(row, ("phase_transition_score", "geometric_coherence")) or 0.0
                for row in usable
            ],
        ],
        dtype=float,
    )
    plt, _power_norm, _matplotlib = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(max(7.0, len(usable) * 0.45), 3.8), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_title("Release versus geometry summary surface")
    ax.set_yticks(range(3), ["release duty", "release effect", "phase/coherence"])
    ax.set_xticks(range(len(usable)), [str(row.get("candidate_id", row.get("substrate", idx))) for idx, row in enumerate(usable)], rotation=45, ha="right")
    fig.colorbar(image, ax=ax, shrink=0.78)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    write_metadata(path, {"status": "rendered", "rows": len(usable), **context})


def render_phase_portraits(rows: list[dict[str, Any]], path: Path, context: dict[str, Any], *, dpi: int) -> None:
    pairs = [
        pair for pair in PHASE_PAIRS if all(any(metric_value(row, metric) is not None for row in rows) for metric in pair)
    ]
    if not pairs:
        save_skipped(path, "No phase portrait metric pairs found.", context)
        return
    plt, _power_norm, _matplotlib = _import_matplotlib()
    fig, axes = plt.subplots(1, len(pairs), figsize=(5.0 * len(pairs), 4.2), constrained_layout=True)
    axes_array = np.atleast_1d(axes)
    steps = np.asarray([int(row.get("step", idx)) for idx, row in enumerate(rows)])
    for ax, (x_metric, y_metric) in zip(axes_array, pairs):
        image = ax.scatter(metric_series(rows, x_metric), metric_series(rows, y_metric), c=steps, cmap="viridis", s=18)
        ax.set_xlabel(x_metric.replace("route_metrics.", ""))
        ax.set_ylabel(y_metric.replace("route_metrics.", ""))
        ax.set_title("Phase portrait")
    fig.colorbar(image, ax=axes_array.tolist(), shrink=0.75, label="step")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    write_metadata(path, {"status": "rendered", "pairs": pairs, **context})


def render_pareto_frontier(payload: dict[str, Any], path: Path, context: dict[str, Any], *, dpi: int) -> None:
    rows = [
        row
        for row in summary_metric_rows(payload)
        if summary_metric_value(row, "internal_richness") is not None
        and summary_metric_value(row, "geometric_coherence") is not None
    ]
    if not rows:
        save_skipped(path, "No candidate summary metrics found for Pareto plot.", context)
        return
    x = np.asarray([summary_metric_value(row, "internal_richness") or 0.0 for row in rows])
    y = np.asarray([summary_metric_value(row, "geometric_coherence") or 0.0 for row in rows])
    color = np.asarray([summary_metric_value(row, "release_duty_cycle") or 0.0 for row in rows])
    size = np.asarray([summary_metric_value(row, "phase_transition_score") or 0.0 for row in rows])
    plt, _power_norm, _matplotlib = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(6.8, 5.2), constrained_layout=True)
    image = ax.scatter(x, y, c=color, s=50 + 160 * size, cmap="viridis", alpha=0.82)
    ax.set_xlabel("internal richness")
    ax.set_ylabel("geometric coherence")
    ax.set_title("Candidate frontier")
    fig.colorbar(image, ax=ax, shrink=0.78, label="release duty cycle")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    write_metadata(path, {"status": "rendered", "metrics": list(PARETO_METRICS), "rows": len(rows), **context})


def render_spectral_channel_power(rows: list[dict[str, Any]], path: Path, context: dict[str, Any], *, dpi: int) -> None:
    metric_paths = available_metrics(rows, ("fast_state_norm", "slow_state_norm", "message_state_norm", "route_metrics.release_pressure_mean"))
    if len(rows) < 4 or not metric_paths:
        save_skipped(path, "Need at least four steps and one metric for spectral power.", context)
        return
    spectra = []
    for metric in metric_paths:
        series = metric_series(rows, metric)
        centered = series - series.mean()
        power = np.abs(np.fft.rfft(centered)) ** 2
        spectra.append(power)
    max_len = max(len(row) for row in spectra)
    matrix = np.zeros((len(spectra), max_len), dtype=float)
    for index, row in enumerate(spectra):
        max_value = float(np.max(row))
        matrix[index, : len(row)] = row / max_value if max_value > 1e-12 else row
    plt, power_norm, _matplotlib = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="magma", norm=power_norm(gamma=0.5, vmin=0.0, vmax=1.0))
    ax.set_title("Spectral channel power")
    ax.set_xlabel("frequency bin")
    ax.set_yticks(range(len(metric_paths)), [metric.replace("route_metrics.", "") for metric in metric_paths])
    fig.colorbar(image, ax=ax, shrink=0.78, label="within-row power fraction")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    write_metadata(path, {"status": "rendered", "metrics": metric_paths, **context})


def render_all(args: argparse.Namespace) -> list[Path]:
    payload = load_payload(Path(args.input))
    records = filter_records(joined_records(payload), args.run_filter, args.seed)
    runs = group_runs(records)
    selected = representative_run(runs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    context = {
        "input": str(args.input),
        "run_filter": args.run_filter,
        "seed": args.seed,
        "selected_run": list(run_key(selected[0])) if selected else None,
    }
    events = payload.get("events", [])
    selected_event_ids = {row["point_id"] for row in selected}
    selected_events = [event for event in events if event.get("point_id") in selected_event_ids]
    comparison = comparison_rows(runs, events)
    outputs = [
        out_dir / "event_aligned_release_windows.png",
        out_dir / "release_variant_comparison.png",
        out_dir / "state_delta_heatmap.png",
        out_dir / "recurrence_distance.png",
        out_dir / "channel_separation_covariance.png",
        out_dir / "sweep_surface_release_vs_geometry.png",
        out_dir / "phase_portraits.png",
        out_dir / "pareto_frontier.png",
        out_dir / "spectral_channel_power.png",
    ]
    if not selected:
        for output in outputs:
            save_skipped(output, "No records matched filters.", context)
        return outputs

    render_event_aligned_release_windows(selected, selected_events, outputs[0], context, dpi=args.dpi)
    render_release_variant_comparison(comparison, outputs[1], context, dpi=args.dpi)
    render_state_delta_heatmap(selected, outputs[2], context, dpi=args.dpi)
    render_recurrence_distance(selected, outputs[3], context, dpi=args.dpi)
    render_channel_separation_covariance(selected, outputs[4], context, dpi=args.dpi)
    render_sweep_surface(payload, outputs[5], context, dpi=args.dpi)
    render_phase_portraits(selected, outputs[6], context, dpi=args.dpi)
    render_pareto_frontier(payload, outputs[7], context, dpi=args.dpi)
    render_spectral_channel_power(selected, outputs[8], context, dpi=args.dpi)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--run-filter", default="all")
    parser.add_argument("--seed", default="all")
    parser.add_argument("--format", choices=("png", "svg", "all"), default="png")
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.format not in {"png", "all"}:
        raise SystemExit("Only PNG output is implemented for machine diagnostics.")
    for output in render_all(args):
        print(f"saved: {output}")


if __name__ == "__main__":
    main()
