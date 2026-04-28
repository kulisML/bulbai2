"""Tests for the inline Pareto-plot SVG renderer.

The night report embeds a 2D scatter plot of objective[0] (resistance
proxy) vs objective[1] (volume delta) for the Pareto front, with the
winner highlighted. These tests exercise the pure-stdlib SVG generator
under three regimes: no data, regular candidates, and mixed candidates
where some entries are sentinel-penalty (objective[0] >= 1e8) and must
be dropped silently.
"""
from __future__ import annotations

import re

from bulbopt.reporting.pareto_plot import render_pareto_plot_svg


def test_render_pareto_plot_svg_handles_empty_candidates() -> None:
    svg = render_pareto_plot_svg([])

    assert isinstance(svg, str)
    assert svg.startswith("<svg")
    assert svg.endswith("</svg>")
    # Empty plot still tells the reader why it's empty.
    assert "no data" in svg.lower()
    # No data circles should be drawn.
    assert "<circle" not in svg


def test_render_pareto_plot_svg_renders_two_candidates() -> None:
    candidates = [
        {"objectives": [1.20, 0.05, 0.10], "candidate_id": "candidate-001"},
        {"objectives": [1.05, 0.12, 0.15], "candidate_id": "candidate-002"},
    ]
    svg = render_pareto_plot_svg(candidates)

    assert svg.startswith("<svg")
    # Two distinct Pareto points → two circles.
    circles = re.findall(r"<circle\b", svg)
    assert len(circles) == 2


def test_render_pareto_plot_svg_drops_penalty_entries() -> None:
    candidates = [
        {"objectives": [1.20, 0.05, 0.10], "candidate_id": "real-001"},
        {"objectives": [1e9, 1e9, 1e9], "candidate_id": "penalty-001"},
        {"objectives": [1.05, 0.12, 0.15], "candidate_id": "real-002"},
        {"objectives": [1.0e8, 1.0e8, 1.0e8], "candidate_id": "penalty-002"},
    ]
    svg = render_pareto_plot_svg(candidates)

    # Only the two real candidates should be plotted.
    circles = re.findall(r"<circle\b", svg)
    assert len(circles) == 2
    # No sentinel-y values leak into the SVG markup.
    assert "1e+09" not in svg
    assert "1000000000" not in svg


def test_render_pareto_plot_svg_handles_all_penalty_candidates() -> None:
    """All-penalty front collapses to an empty plot with a friendly note."""
    candidates = [
        {"objectives": [1e9, 1e9, 1e9], "candidate_id": "p-001"},
        {"objectives": [1e9, 1e9, 1e9], "candidate_id": "p-002"},
    ]
    svg = render_pareto_plot_svg(candidates)

    assert svg.startswith("<svg")
    assert "<circle" not in svg
    assert "no data" in svg.lower() or "no valid" in svg.lower()
