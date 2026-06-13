from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/demian-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/demian-cache")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
EVIDENCE = ROOT / "evidence_summary.csv"


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _rows() -> list[dict[str, str]]:
    with EVIDENCE.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d6dce5", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def render_protocol() -> None:
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    ax.axis("off")

    boxes = [
        (0.04, 0.58, 0.20, 0.24, "Surface readout\n$y_t = R(s_t)$"),
        (0.40, 0.58, 0.22, 0.24, "Surface label\nFIXED_POINT"),
        (0.74, 0.58, 0.22, 0.24, "Insufficient verdict\nwithout state probes"),
        (0.04, 0.12, 0.20, 0.24, "Internal state\n$s_t$"),
        (0.40, 0.12, 0.22, 0.24, "Interior probes\nchannels, routes,\nresume, ablation"),
        (0.74, 0.12, 0.22, 0.24, "Structured basin\nor true collapse"),
    ]

    for x, y, w, h, text in boxes:
        ax.add_patch(
            plt.Rectangle(
                (x, y),
                w,
                h,
                facecolor="#f6f8fb",
                edgecolor="#364152",
                linewidth=1.2,
                joinstyle="round",
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    arrows = [
        ((0.25, 0.70), (0.39, 0.70)),
        ((0.63, 0.70), (0.73, 0.70)),
        ((0.25, 0.24), (0.39, 0.24)),
        ((0.63, 0.24), (0.73, 0.24)),
        ((0.14, 0.56), (0.14, 0.38)),
        ((0.51, 0.56), (0.51, 0.38)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#233142"})

    ax.text(
        0.50,
        0.94,
        "A fixed surface label is a starting point, not the whole state.",
        ha="center",
        va="center",
        fontsize=13,
        weight="bold",
    )
    _save(fig, "surface_internal_protocol.png")


def render_dual_gru(_rows: list[dict[str, str]]) -> None:
    variants = ["dual_gru_v3b:current", "dual_gru_v3b:tight", "dual_gru_v3b:threshold", "dual_gru_v3"]
    labels = ["v3b current", "v3b tight", "v3b threshold", "v3"]
    message_norm = {
        "dual_gru_v3b:current": 23.1086,
        "dual_gru_v3b:tight": 0.0624,
        "dual_gru_v3b:threshold": 0.0753,
        "dual_gru_v3": 0.0624,
    }
    entropy = {
        "dual_gru_v3b:current": 1.7779,
        "dual_gru_v3b:tight": 0.3423,
        "dual_gru_v3b:threshold": 0.5120,
        "dual_gru_v3": 0.5069,
    }
    class_count = {
        "dual_gru_v3b:current": 1,
        "dual_gru_v3b:tight": 2,
        "dual_gru_v3b:threshold": 2,
        "dual_gru_v3": 2,
    }

    x = np.arange(len(variants))
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.7), constrained_layout=True)
    axes[0].bar(x, [class_count[v] for v in variants], color="#3b6ea8")
    axes[0].set_title("Interior classes")
    axes[0].set_ylim(0, 2.5)
    axes[0].set_ylabel("count")

    axes[1].bar(x, [message_norm[v] for v in variants], color="#9f5f37")
    axes[1].set_yscale("log")
    axes[1].set_title("Message norm")
    axes[1].set_ylabel("mean, log scale")

    axes[2].bar(x, [entropy[v] for v in variants], color="#3f7f5f")
    axes[2].set_title("Bottleneck entropy")
    axes[2].set_ylabel("mean")

    for ax in axes:
        _style(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.text(0.5, -0.37, "All rows: surface label FIXED_POINT in 8/8 runs", transform=ax.transAxes, ha="center")

    fig.suptitle("Same fixed-point surface, different hidden interiors", fontsize=14, weight="bold")
    _save(fig, "dual_gru_interiors.png")


def render_capsule(_rows: list[dict[str, str]]) -> None:
    substrates = ["demian_native_v9", "v9_five_channel"]
    labels = ["native v9", "v9 five-channel"]
    surface_only = {
        "demian_native_v9": 0.2262487313710153,
        "v9_five_channel": 0.27455845288932323,
    }
    full = {name: 0.0 for name in substrates}

    x = np.arange(len(substrates))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.bar(x - width / 2, [full[s] for s in substrates], width, label="full capsule", color="#3b6ea8")
    ax.bar(x + width / 2, [surface_only[s] for s in substrates], width, label="surface only", color="#b35445")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("minimum mean continuation gap")
    ax.set_title("Full state resumes; surface-only state does not")
    ax.legend(frameon=False)
    _style(ax)
    for i, value in enumerate([surface_only[s] for s in substrates]):
        ax.text(i + width / 2, value + 0.008, f"{value:.3f}", ha="center", fontsize=9)
    ax.text(x[0] - width / 2, 0.008, "0.0", ha="center", fontsize=9)
    ax.text(x[1] - width / 2, 0.008, "0.0", ha="center", fontsize=9)
    _save(fig, "capsule_resume_gap.png")


def main() -> None:
    rows = _rows()
    render_protocol()
    render_dual_gru(rows)
    render_capsule(rows)


if __name__ == "__main__":
    main()
