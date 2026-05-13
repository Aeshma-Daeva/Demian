#!/usr/bin/env python3
"""Render compact 2D publication visuals for the Demian README.

The figures are SVG so GitHub can render them directly. Color intensity uses a
quadratic mapping: normalized magnitude is squared before being mapped to fill
color. This keeps low-amplitude background activity visually quiet and makes
strong neuron/gate activations visible without inventing categorical labels.

The script also emits Matplotlib PNG heatmaps using standard colormaps. Those
are useful for publication/export contexts where a real colormap and colorbar
are clearer than hand-authored SVG cells.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from development.probe_v9_message_carrier_strange import ExperimentalV9MessageCarrier


CHANNELS = ("fast", "slow", "control", "message", "carrier")
GATES = (
    ("fast_update_mean", "fast"),
    ("slow_write_mean", "slow"),
    ("control_write_mean", "control"),
    ("message_write_mean", "message"),
    ("carrier_write_mean", "carrier"),
    ("release_open_mean", "release open"),
    ("release_strength_mean", "release strength"),
    ("release_pressure_mean", "pressure"),
)

SVG_BG = "#0d1117"
SVG_CELL_BG = "#20242b"
SVG_TEXT = "#f0f6fc"
SVG_MUTED = "#8b949e"
SVG_AXIS = "#c9d1d9"
SVG_GRID = "#30363d"
SIGNED_POSITIVE = (45, 212, 191)
SIGNED_NEGATIVE = (251, 113, 133)
UNSIGNED_HIGH = (96, 165, 250)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def mix_channel(start: int, end: int, amount: float) -> int:
    return round(start + (end - start) * clamp01(amount))


def rgb_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def signed_quadratic_color(value: float, scale: float) -> str:
    if scale <= 1e-12:
        return SVG_CELL_BG
    strength = clamp01(abs(value) / scale) ** 2
    base = (32, 36, 43)
    target = SIGNED_POSITIVE if value >= 0 else SIGNED_NEGATIVE
    return rgb_hex(tuple(mix_channel(base[i], target[i], strength) for i in range(3)))


def signed_readable_color(value: float, scale: float) -> str:
    """High-contrast signed color for reader-facing overview SVGs."""
    if scale <= 1e-12:
        return SVG_CELL_BG
    strength = clamp01(abs(value) / scale) ** 0.72
    base = (32, 36, 43)
    target = SIGNED_POSITIVE if value >= 0 else SIGNED_NEGATIVE
    return rgb_hex(tuple(mix_channel(base[i], target[i], strength) for i in range(3)))


def percentile_abs_scale(values: list[float], percentile: float) -> float:
    """Return a robust signed-display scale so one outlier does not mute a figure."""
    magnitudes = sorted(abs(value) for value in values)
    if not magnitudes:
        return 1.0
    index = min(len(magnitudes) - 1, max(0, math.ceil(len(magnitudes) * percentile) - 1))
    scale = magnitudes[index]
    return scale if scale > 1e-12 else 1.0


def unsigned_quadratic_color(value: float, scale: float) -> str:
    if scale <= 1e-12:
        return SVG_CELL_BG
    strength = clamp01(value / scale) ** 2
    base = (32, 36, 43)
    target = UNSIGNED_HIGH
    return rgb_hex(tuple(mix_channel(base[i], target[i], strength) for i in range(3)))


def escape_xml(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def trace_v9_five_channel(
    *,
    hidden_size: int,
    seed: int,
    steps: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = ExperimentalV9MessageCarrier(
        hidden_size,
        initial_message_scale=0.0,
        initial_carrier_scale=0.0,
        release_policy="learned",
        release_gain=0.0,
        binding_start_step=1,
    )
    device = torch.device("cpu")
    state = model.initial_state(1, device)
    channel_rows: dict[str, list[list[float]]] = {name: [] for name in CHANNELS}
    gate_rows: dict[str, list[float]] = {name: [] for name, _label in GATES}

    with torch.no_grad():
        for _step in range(steps):
            state = model.step(state)
            components = model.state_components(state)
            for name in CHANNELS:
                channel_rows[name].append(
                    components[name].view(-1).detach().float().cpu().tolist()
                )
            aux = model.step_aux()
            for name, _label in GATES:
                gate_rows[name].append(float(aux.get(name, 0.0)))

    return {
        "config": {
            "hidden_size": hidden_size,
            "seed": seed,
            "steps": steps,
            "substrate": "v9_five_channel",
        },
        "channels": channel_rows,
        "gates": gate_rows,
    }


def neuron_heatmap_matrix(trace: dict[str, Any]) -> list[list[float]]:
    """Return channel-stacked rows of signed neuron activation values."""
    channels: dict[str, list[list[float]]] = trace["channels"]
    rows: list[list[float]] = []
    for name in CHANNELS:
        channel_rows = channels[name]
        channel_width = len(channel_rows[0]) if channel_rows else 0
        for neuron_index in range(channel_width):
            rows.append([float(step_values[neuron_index]) for step_values in channels[name]])
    return rows


def gate_heatmap_matrix(trace: dict[str, Any]) -> list[list[float]]:
    """Return route/release gate metrics as rows over time."""
    gates: dict[str, list[float]] = trace["gates"]
    return [[float(value) for value in gates[name]] for name, _label in GATES]


def row_normalized_matrix(matrix: list[list[float]]) -> list[list[float]]:
    """Normalize each row independently to preserve within-metric structure."""
    rows: list[list[float]] = []
    for row in matrix:
        max_value = max(row, default=0.0)
        if max_value <= 1e-12:
            rows.append([0.0 for _value in row])
        else:
            rows.append([float(value) / max_value for value in row])
    return rows


def selected_channel_neurons(trace: dict[str, Any], limit: int) -> dict[str, list[int]]:
    """Select the most active neuron rows per channel for reader-facing views."""
    channels: dict[str, list[list[float]]] = trace["channels"]
    selected: dict[str, list[int]] = {}
    for name in CHANNELS:
        channel_rows = channels[name]
        channel_width = len(channel_rows[0]) if channel_rows else 0
        scores = []
        for neuron_index in range(channel_width):
            score = max((abs(float(step_values[neuron_index])) for step_values in channel_rows), default=0.0)
            scores.append((score, neuron_index))
        scores.sort(reverse=True)
        selected[name] = [neuron_index for _score, neuron_index in scores[:limit]]
    return selected


def neuron_overview_matrix(trace: dict[str, Any], selected: dict[str, list[int]]) -> list[list[float]]:
    """Return only selected neuron rows, grouped by channel, for readable plots."""
    channels: dict[str, list[list[float]]] = trace["channels"]
    rows: list[list[float]] = []
    for name in CHANNELS:
        for neuron_index in selected[name]:
            rows.append([float(step_values[neuron_index]) for step_values in channels[name]])
    return rows


def _import_matplotlib() -> tuple[Any, Any, Any, Any]:
    try:
        mpl_config_dir = Path(tempfile.gettempdir()) / "demian-matplotlib"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap, PowerNorm, TwoSlopeNorm
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Matplotlib is required for PNG heatmap rendering. "
            "Install project requirements first: ./venv/bin/pip install -r requirements.txt"
        ) from exc
    return plt, LinearSegmentedColormap, PowerNorm, TwoSlopeNorm


def style_dark_matplotlib_figure(fig: Any, ax: Any) -> None:
    fig.patch.set_facecolor(SVG_BG)
    ax.set_facecolor(SVG_CELL_BG)
    ax.title.set_color(SVG_TEXT)
    ax.xaxis.label.set_color(SVG_MUTED)
    ax.yaxis.label.set_color(SVG_MUTED)
    ax.tick_params(colors=SVG_MUTED)
    for spine in ax.spines.values():
        spine.set_color(SVG_GRID)


def style_dark_colorbar(colorbar: Any) -> None:
    colorbar.ax.set_facecolor(SVG_BG)
    colorbar.outline.set_edgecolor(SVG_GRID)
    colorbar.ax.yaxis.label.set_color(SVG_MUTED)
    colorbar.ax.tick_params(colors=SVG_MUTED)


def render_neuron_heatmap_png(trace: dict[str, Any], path: Path, *, dpi: int = 200) -> None:
    """Render a reader-facing signed neuron activation overview on a dark canvas."""
    plt, linear_segmented_colormap, _power_norm, two_slope_norm = _import_matplotlib()
    selected = selected_channel_neurons(trace, 4)
    matrix = neuron_overview_matrix(trace, selected)
    max_abs = percentile_abs_scale([value for row in matrix for value in row], 0.92)
    norm = two_slope_norm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    config = trace["config"]
    steps = int(config["steps"])
    channel_widths = [len(selected[name]) for name in CHANNELS]

    fig_height = max(5.0, len(matrix) * 0.18 + 2.7)
    fig_width = max(9.6, steps * 0.095 + 3.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    style_dark_matplotlib_figure(fig, ax)
    cmap = linear_segmented_colormap.from_list(
        "demian_signed_dark",
        [rgb_hex(SIGNED_NEGATIVE), SVG_CELL_BG, rgb_hex(SIGNED_POSITIVE)],
    )
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    centers: list[float] = []
    row_offset = 0
    for width in channel_widths:
        centers.append(row_offset + (width - 1) / 2 if width else row_offset)
        row_offset += width
    ax.set_yticks(centers, [f"{name}: loudest 4 neurons" for name in CHANNELS])
    ax.set_xlabel("time step")
    ax.set_ylabel("channel overview")
    ax.set_title(
        "Neuron activity overview: teal pushes up, pink pushes down"
    )
    ax.set_xticks([0, steps // 2, steps - 1], [1, steps // 2 + 1, steps])
    boundary_row = 0
    for width in channel_widths[:-1]:
        boundary_row += width
        ax.axhline(boundary_row - 0.5, color=SVG_GRID, linewidth=0.5, alpha=0.8)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.78)
    colorbar.set_label("activation strength")
    style_dark_colorbar(colorbar)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_gate_heatmap_png(trace: dict[str, Any], path: Path, *, dpi: int = 200) -> None:
    """Render unsigned gate metrics with viridis and quadratic PowerNorm."""
    plt, _linear_segmented_colormap, power_norm, _two_slope_norm = _import_matplotlib()
    matrix = gate_heatmap_matrix(trace)
    max_value = max((value for row in matrix for value in row), default=0.0)
    if max_value <= 1e-12:
        max_value = 1.0
    config = trace["config"]
    steps = int(config["steps"])

    fig_width = max(8.0, steps * 0.08 + 2.8)
    fig, ax = plt.subplots(figsize=(fig_width, 4.6), constrained_layout=True)
    style_dark_matplotlib_figure(fig, ax)
    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        norm=power_norm(gamma=2.0, vmin=0.0, vmax=max_value),
    )
    ax.set_yticks(range(len(GATES)), [label for _name, label in GATES])
    ax.set_xlabel("step")
    ax.set_ylabel("metric")
    ax.set_title(
        f"v9 five-channel gating activations: seed={config['seed']}, hidden={config['hidden_size']}, steps={steps}"
    )
    ax.set_xticks([0, steps - 1], [1, steps])
    colorbar = fig.colorbar(image, ax=ax, shrink=0.78)
    colorbar.set_label("metric value, PowerNorm gamma=2.0")
    style_dark_colorbar(colorbar)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_gate_row_normalized_heatmap_png(
    trace: dict[str, Any], path: Path, *, dpi: int = 200
) -> None:
    """Render gate metrics with each row normalized to its own maximum."""
    plt, _linear_segmented_colormap, power_norm, _two_slope_norm = _import_matplotlib()
    matrix = row_normalized_matrix(gate_heatmap_matrix(trace))
    config = trace["config"]
    steps = int(config["steps"])

    fig_width = max(8.0, steps * 0.08 + 2.8)
    fig, ax = plt.subplots(figsize=(fig_width, 4.6), constrained_layout=True)
    style_dark_matplotlib_figure(fig, ax)
    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        norm=power_norm(gamma=2.0, vmin=0.0, vmax=1.0),
    )
    ax.set_yticks(range(len(GATES)), [label for _name, label in GATES])
    ax.set_xlabel("step")
    ax.set_ylabel("metric")
    ax.set_title(
        "v9 five-channel gating activations, row-normalized: "
        f"seed={config['seed']}, hidden={config['hidden_size']}, steps={steps}"
    )
    ax.set_xticks([0, steps - 1], [1, steps])
    colorbar = fig.colorbar(image, ax=ax, shrink=0.78)
    colorbar.set_label("within-row fraction of max, PowerNorm gamma=2.0")
    style_dark_colorbar(colorbar)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_neuron_svg(trace: dict[str, Any]) -> str:
    config = trace["config"]
    channels: dict[str, list[list[float]]] = trace["channels"]
    steps = int(config["steps"])
    hidden_size = int(config["hidden_size"])
    visible_neurons = min(4, hidden_size)
    cell = 11
    gap = 0
    left = 202
    top = 152
    panel_gap = 48
    label_gap = 20
    panel_h = visible_neurons * cell + (visible_neurons - 1) * gap
    plot_w = steps * cell + (steps - 1) * gap
    width = left + plot_w + 124
    height = top + len(CHANNELS) * panel_h + (len(CHANNELS) - 1) * panel_gap + 150
    values = [
        float(value)
        for rows in channels.values()
        for step_values in rows
        for value in step_values
    ]
    display_abs = percentile_abs_scale(values, 0.82)
    true_max_abs = max((abs(value) for value in values), default=display_abs)
    selected_neurons = selected_channel_neurons(trace, visible_neurons)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">v9 five-channel neuron activation trace</title>',
        '<desc id="desc">Beginner-readable overview of the strongest neuron activations for fast, slow, control, message, and carrier channels. Teal cells are positive activations, pink cells are negative activations, and darker cells are quiet.</desc>',
        f'<rect width="{width}" height="{height}" fill="{SVG_BG}"/>',
        f'<text x="34" y="42" font-family="Arial, sans-serif" font-size="25" font-weight="700" fill="{SVG_TEXT}">Neuron activity: the readable version</text>',
        f'<text x="34" y="72" font-family="Arial, sans-serif" font-size="15" fill="{SVG_AXIS}">Each channel shows its 4 loudest neurons. Read left to right as time passes.</text>',
        f'<text x="34" y="98" font-family="Arial, sans-serif" font-size="15" fill="{SVG_MUTED}">Teal means push up, pink means push down, dark means quiet. Brighter color means stronger activity.</text>',
        f'<text x="{left}" y="{top - 18}" font-family="Arial, sans-serif" font-size="12" fill="{SVG_MUTED}">step 1</text>',
        f'<text x="{left + plot_w // 2 - 18}" y="{top - 18}" font-family="Arial, sans-serif" font-size="12" fill="{SVG_MUTED}">step {steps // 2 + 1}</text>',
        f'<text x="{left + plot_w - 42}" y="{top - 18}" font-family="Arial, sans-serif" font-size="12" fill="{SVG_MUTED}">step {steps}</text>',
    ]
    for channel_index, name in enumerate(CHANNELS):
        y0 = top + channel_index * (panel_h + panel_gap)
        parts.append(
            f'<rect x="{left}" y="{y0}" width="{plot_w}" height="{panel_h}" fill="{SVG_CELL_BG}" stroke="{SVG_GRID}" stroke-width="1"/>'
        )
        channel_values = [
            float(step_values[neuron_index])
            for step_values in channels[name]
            for neuron_index in selected_neurons[name]
        ]
        max_channel = max(channel_values, default=0.0)
        min_channel = min(channel_values, default=0.0)
        if abs(max_channel) >= abs(min_channel):
            summary = f"main signal pushes up to +{max_channel:.1f}"
        else:
            summary = f"main signal pushes down to {min_channel:.1f}"
        parts.append(
            f'<text x="34" y="{y0 + label_gap}" font-family="Arial, sans-serif" font-size="15" font-weight="700" fill="{SVG_TEXT}">{escape_xml(name)}</text>'
        )
        parts.append(
            f'<text x="34" y="{y0 + label_gap + 22}" font-family="Arial, sans-serif" font-size="12" fill="{SVG_MUTED}">{escape_xml(summary)}</text>'
        )
        for tick_step in (0, steps // 2, steps - 1):
            x = left + tick_step * (cell + gap)
            parts.append(
                f'<line x1="{x}" y1="{y0}" x2="{x}" y2="{y0 + panel_h}" stroke="{SVG_GRID}" stroke-width="1" opacity="0.75"/>'
            )
        for step_index, values in enumerate(channels[name]):
            x = left + step_index * (cell + gap)
            for row_index, neuron_index in enumerate(selected_neurons[name]):
                value = values[neuron_index]
                y = y0 + row_index * (cell + gap)
                fill = signed_readable_color(float(value), display_abs)
                parts.append(
                    f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}"><title>{escape_xml(name)} step {step_index + 1} neuron {neuron_index}: {float(value):.6f}</title></rect>'
                )
        parts.append(
            f'<text x="{left + plot_w + 14}" y="{y0 + panel_h // 2 + 4}" font-family="Arial, sans-serif" font-size="11" fill="{SVG_MUTED}">top 4</text>'
        )
    legend_y = height - 38
    parts.extend(
        [
            f'<rect x="34" y="{legend_y - 44}" width="18" height="18" fill="{signed_readable_color(-display_abs, display_abs)}"/>',
            f'<text x="62" y="{legend_y - 30}" font-family="Arial, sans-serif" font-size="14" fill="{SVG_AXIS}">pink: pushes down</text>',
            f'<rect x="224" y="{legend_y - 44}" width="18" height="18" fill="{signed_readable_color(0.0, display_abs)}"/>',
            f'<text x="252" y="{legend_y - 30}" font-family="Arial, sans-serif" font-size="14" fill="{SVG_AXIS}">dark: quiet</text>',
            f'<rect x="360" y="{legend_y - 44}" width="18" height="18" fill="{signed_readable_color(display_abs, display_abs)}"/>',
            f'<text x="388" y="{legend_y - 30}" font-family="Arial, sans-serif" font-size="14" fill="{SVG_AXIS}">teal: pushes up</text>',
            f'<text x="34" y="{legend_y + 2}" font-family="Arial, sans-serif" font-size="12" fill="{SVG_MUTED}">This is an overview, not every neuron. Full trace: seed={config["seed"]}, hidden={hidden_size}, steps={steps}; display scale +/-{display_abs:.2f}, true max |activation|={true_max_abs:.2f}</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def render_gate_svg(trace: dict[str, Any]) -> str:
    config = trace["config"]
    gates: dict[str, list[float]] = trace["gates"]
    steps = int(config["steps"])
    cell = 9
    gap = 0
    left = 154
    top = 106
    row_h = 18
    plot_w = steps * cell + (steps - 1) * gap
    width = left + plot_w + 44
    height = top + len(GATES) * row_h + 86
    max_pressure = max(gates["release_pressure_mean"]) if gates["release_pressure_mean"] else 1.0

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">v9 five-channel gate activation trace</title>',
        '<desc id="desc">Quadratic-color heatmap of route and release gate metrics over time.</desc>',
        f'<rect width="{width}" height="{height}" fill="{SVG_BG}"/>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{len(GATES) * row_h - 6}" fill="{SVG_CELL_BG}"/>',
        f'<text x="34" y="42" font-family="Arial, sans-serif" font-size="23" font-weight="700" fill="{SVG_TEXT}">Gating activations, quadratic color</text>',
        f'<text x="34" y="70" font-family="Arial, sans-serif" font-size="13" fill="{SVG_MUTED}">v9 five-channel, seed={config["seed"]}, hidden={config["hidden_size"]}, steps={steps}; intensity=(metric/max)^2</text>',
    ]
    for row_index, (name, label) in enumerate(GATES):
        y = top + row_index * row_h
        scale = max_pressure if name == "release_pressure_mean" else 1.0
        parts.append(
            f'<text x="34" y="{y + 11}" font-family="Arial, sans-serif" font-size="12" font-weight="700" fill="{SVG_TEXT}">{escape_xml(label)}</text>'
        )
        for step_index, value in enumerate(gates[name]):
            x = left + step_index * (cell + gap)
            fill = unsigned_quadratic_color(float(value), scale)
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="12" rx="1" fill="{fill}"><title>{escape_xml(label)} step {step_index + 1}: {float(value):.6f}</title></rect>'
            )
    legend_y = height - 42
    parts.extend(
        [
            f'<rect x="34" y="{legend_y - 10}" width="12" height="12" rx="1" fill="{unsigned_quadratic_color(0.0, 1.0)}"/>',
            f'<text x="54" y="{legend_y}" font-family="Arial, sans-serif" font-size="12" fill="{SVG_MUTED}">low</text>',
            f'<rect x="104" y="{legend_y - 10}" width="12" height="12" rx="1" fill="{unsigned_quadratic_color(0.5, 1.0)}"/>',
            f'<text x="124" y="{legend_y}" font-family="Arial, sans-serif" font-size="12" fill="{SVG_MUTED}">mid maps to 25% intensity</text>',
            f'<rect x="310" y="{legend_y - 10}" width="12" height="12" rx="1" fill="{unsigned_quadratic_color(1.0, 1.0)}"/>',
            f'<text x="330" y="{legend_y}" font-family="Arial, sans-serif" font-size="12" fill="{SVG_MUTED}">high</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def render_anatomy_svg() -> str:
    width = 920
    height = 420
    nodes = {
        "fast": (178, 188, "#137970"),
        "message": (352, 120, "#2a578d"),
        "carrier": (552, 120, "#6d5c9c"),
        "slow": (724, 188, "#a84e4b"),
        "control": (452, 286, "#8a6a23"),
    }
    edges = [
        ("fast", "message", 0.25),
        ("message", "carrier", 0.35),
        ("carrier", "slow", 0.25),
        ("slow", "control", 0.35),
        ("control", "fast", 0.62),
        ("message", "fast", 0.12),
        ("carrier", "fast", 0.08),
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">v9 five-channel anatomy</title>',
        '<desc id="desc">2D routing diagram for fast, slow, control, message, and carrier channels with quadratic edge opacity.</desc>',
        f'<rect width="{width}" height="{height}" fill="#f7f7f4"/>',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#33383d"/></marker></defs>',
        '<text x="42" y="54" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#202124">v9 five-channel anatomy</text>',
        '<text x="42" y="84" font-family="Arial, sans-serif" font-size="13" fill="#4c535a">routing scaffold for Demian v1 work; edge opacity uses squared route scale</text>',
    ]
    max_weight = max(weight for _src, _dst, weight in edges)
    for src, dst, weight in edges:
        x1, y1, _c1 = nodes[src]
        x2, y2, _c2 = nodes[dst]
        opacity = 0.16 + 0.78 * (weight / max_weight) ** 2
        width_px = 1.5 + 5.0 * (weight / max_weight) ** 2
        parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#33383d" stroke-width="{width_px:.2f}" opacity="{opacity:.3f}" marker-end="url(#arrow)"><title>{src} to {dst}: scale {weight}</title></line>'
        )
    for name, (x, y, color) in nodes.items():
        parts.extend(
            [
                f'<circle cx="{x}" cy="{y}" r="48" fill="{color}" opacity="0.94"/>',
                f'<text x="{x}" y="{y + 5}" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" font-weight="700" fill="#ffffff">{escape_xml(name)}</text>',
            ]
        )
    parts.extend(
        [
            '<text x="42" y="368" font-family="Arial, sans-serif" font-size="12" fill="#4c535a">The visual is an anatomy map, not a performance claim. It shows explicit state owners and route directions used by the current v9 five-channel probe.</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=94)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--out-dir", default="docs/assets")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--formats",
        choices=("svg", "png", "all"),
        default="all",
        help="Which visual formats to render. Anatomy is SVG-only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace = trace_v9_five_channel(
        hidden_size=args.hidden_size,
        seed=args.seed,
        steps=args.steps,
    )
    if args.formats in {"svg", "all"}:
        files = {
            "v9_5ch_neuron_activations.svg": render_neuron_svg(trace),
            "v9_5ch_neuron_activations_dark.svg": render_neuron_svg(trace),
            "v9_5ch_neuron_activations_solid.svg": render_neuron_svg(trace),
            "v9_5ch_neuron_activity_overview.svg": render_neuron_svg(trace),
            "v9_5ch_gating_activations.svg": render_gate_svg(trace),
            "v9_5ch_gating_activations_dark.svg": render_gate_svg(trace),
            "v9_5ch_gating_activations_solid.svg": render_gate_svg(trace),
            "v9_5ch_anatomy.svg": render_anatomy_svg(),
        }
        for name, svg in files.items():
            path = out_dir / name
            path.write_text(svg, encoding="utf-8")
            print(f"saved: {path}")

    if args.formats in {"png", "all"}:
        png_files = {
            "v9_5ch_neuron_heatmap.png": render_neuron_heatmap_png,
            "v9_5ch_neuron_heatmap_solid.png": render_neuron_heatmap_png,
            "v9_5ch_neuron_activity_overview.png": render_neuron_heatmap_png,
            "v9_5ch_gate_heatmap.png": render_gate_heatmap_png,
            "v9_5ch_gate_heatmap_solid.png": render_gate_heatmap_png,
            "v9_5ch_gate_heatmap_row_normalized.png": render_gate_row_normalized_heatmap_png,
            "v9_5ch_gate_heatmap_row_normalized_solid.png": render_gate_row_normalized_heatmap_png,
        }
        for name, renderer in png_files.items():
            path = out_dir / name
            renderer(trace, path, dpi=args.dpi)
            print(f"saved: {path}")


if __name__ == "__main__":
    main()
