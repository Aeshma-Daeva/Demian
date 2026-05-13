"""Tests for GitHub-readable publication SVG generation."""

import pytest

from development.render_publication_2d_visuals import (
    gate_heatmap_matrix,
    neuron_heatmap_matrix,
    render_gate_heatmap_png,
    render_gate_row_normalized_heatmap_png,
    render_gate_svg,
    render_neuron_heatmap_png,
    render_neuron_svg,
    row_normalized_matrix,
    signed_quadratic_color,
    trace_v9_five_channel,
    unsigned_quadratic_color,
)


def test_quadratic_color_maps_midpoint_to_quiet_intensity():
    low = unsigned_quadratic_color(0.0, 1.0)
    mid = unsigned_quadratic_color(0.5, 1.0)
    high = unsigned_quadratic_color(1.0, 1.0)

    assert low == "#20242b"
    assert mid not in {low, high}
    assert high == "#60a5fa"
    assert signed_quadratic_color(-1.0, 1.0) == "#fb7185"
    assert signed_quadratic_color(1.0, 1.0) == "#2dd4bf"


def test_publication_visual_renderers_emit_svg_from_trace():
    trace = trace_v9_five_channel(hidden_size=4, seed=94, steps=4)
    neuron_svg = render_neuron_svg(trace)
    gate_svg = render_gate_svg(trace)

    assert "<svg" in neuron_svg
    assert "Neuron activity: the readable version" in neuron_svg
    assert "Each channel shows its 4 loudest neurons" in neuron_svg
    assert "This is an overview, not every neuron" in neuron_svg
    assert "fast step 1 neuron 0" in neuron_svg
    assert "<svg" in gate_svg
    assert "Gating activations, quadratic color" in gate_svg
    assert "release open step 1" in gate_svg


def test_publication_heatmap_matrices_preserve_trace_shape():
    trace = trace_v9_five_channel(hidden_size=4, seed=94, steps=5)

    neuron_matrix = neuron_heatmap_matrix(trace)
    gate_matrix = gate_heatmap_matrix(trace)

    assert len(neuron_matrix) == 20
    assert all(len(row) == 5 for row in neuron_matrix)
    assert len(gate_matrix) == 8
    assert all(len(row) == 5 for row in gate_matrix)
    assert neuron_matrix[0][0] == trace["channels"]["fast"][0][0]
    assert gate_matrix[0][0] == trace["gates"]["fast_update_mean"][0]


def test_row_normalized_matrix_preserves_zero_rows_and_scales_active_rows():
    matrix = row_normalized_matrix([[0.0, 0.0], [0.25, 0.5], [2.0, 1.0]])

    assert matrix == [[0.0, 0.0], [0.5, 1.0], [1.0, 0.5]]


def test_publication_heatmap_renderers_emit_png_files(tmp_path):
    pytest.importorskip("matplotlib")
    trace = trace_v9_five_channel(hidden_size=4, seed=94, steps=5)
    neuron_path = tmp_path / "neurons.png"
    gate_path = tmp_path / "gates.png"
    row_normalized_gate_path = tmp_path / "gates-row-normalized.png"

    render_neuron_heatmap_png(trace, neuron_path, dpi=80)
    render_gate_heatmap_png(trace, gate_path, dpi=80)
    render_gate_row_normalized_heatmap_png(trace, row_normalized_gate_path, dpi=80)

    assert neuron_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert gate_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert row_normalized_gate_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
