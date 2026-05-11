"""Tests for GitHub-readable publication SVG generation."""

from development.render_publication_2d_visuals import (
    render_gate_svg,
    render_neuron_svg,
    signed_quadratic_color,
    trace_v9_five_channel,
    unsigned_quadratic_color,
)


def test_quadratic_color_maps_midpoint_to_quiet_intensity():
    low = unsigned_quadratic_color(0.0, 1.0)
    mid = unsigned_quadratic_color(0.5, 1.0)
    high = unsigned_quadratic_color(1.0, 1.0)

    assert low == "#f1f2ef"
    assert mid not in {low, high}
    assert high == "#2a578d"
    assert signed_quadratic_color(-1.0, 1.0) == "#a84e4b"
    assert signed_quadratic_color(1.0, 1.0) == "#137970"


def test_publication_visual_renderers_emit_svg_from_trace():
    trace = trace_v9_five_channel(hidden_size=4, seed=94, steps=4)
    neuron_svg = render_neuron_svg(trace)
    gate_svg = render_gate_svg(trace)

    assert "<svg" in neuron_svg
    assert "Neuron activations, quadratic color" in neuron_svg
    assert "fast step 1 neuron 0" in neuron_svg
    assert "<svg" in gate_svg
    assert "Gating activations, quadratic color" in gate_svg
    assert "release open step 1" in gate_svg
