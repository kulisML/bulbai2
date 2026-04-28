"""Hand-rolled SVG renderer for the night-run Pareto-front scatter plot.

Audit A finding (2026-04-26): the night-run HTML report previously
showed Pareto candidates only as a numeric table. This module emits an
inline ``<svg>`` so the report contains a 2D scatter of objective[0]
(resistance proxy, x-axis) vs objective[1] (volume delta, y-axis) with
the winner highlighted.

The SVG is generated with pure stdlib string formatting — no matplotlib
dependency. Penalty-flag candidates (``objectives[0] >= 1e8``) are the
sentinel value used elsewhere in the optimisation gates and are dropped
from the plot before scaling so they cannot collapse the axis range.
"""
from __future__ import annotations

from typing import Iterable, Sequence

# Sentinel used by the mid/high-fidelity gates for unevaluable candidates.
# Any objective[0] at or above this is treated as a penalty entry.
_PENALTY_THRESHOLD = 1e8


def _format_number(value: float) -> str:
    """Compact, locale-free numeric format for axis labels."""
    if value == 0:
        return "0"
    abs_value = abs(value)
    if abs_value >= 1000 or abs_value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def _is_real_candidate(candidate: dict) -> bool:
    objectives = candidate.get("objectives") or []
    if len(objectives) < 2:
        return False
    try:
        primary = float(objectives[0])
        secondary = float(objectives[1])
    except (TypeError, ValueError):
        return False
    return primary < _PENALTY_THRESHOLD and secondary < _PENALTY_THRESHOLD


def _empty_svg(width: int, height: int, title: str, message: str) -> str:
    """Return a minimal but well-formed SVG with a single text label."""
    cx = width / 2
    cy = height / 2
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" '
        f'class="pareto-plot-svg" role="img" '
        f'aria-label="{title}">'
        f'<title>{title}</title>'
        f'<rect x="0" y="0" width="{width}" height="{height}" '
        f'fill="#fafafa" stroke="#ccc" />'
        f'<text x="{cx}" y="{cy}" text-anchor="middle" '
        f'dominant-baseline="middle" '
        f'font-family="system-ui, sans-serif" font-size="14" '
        f'fill="#666">{message}</text>'
        f'</svg>'
    )


def render_pareto_plot_svg(
    candidates: Sequence[dict],
    width: int = 480,
    height: int = 320,
    title: str = "Pareto front: resistance vs volume",
) -> str:
    """Return an inline SVG string scattering Pareto candidates.

    ``candidates`` is a list of dicts shaped like
    ``{"objectives": [resistance, volume_delta, mesh_quality],
       "candidate_id": str | None}``.

    The first non-penalty entry (``objectives[0] < 1e8``) is treated as
    the winner and drawn in a different colour. Penalty entries are
    silently dropped. Empty input or all-penalty input renders an empty
    plot with a "no data" note.
    """
    real = [c for c in candidates if _is_real_candidate(c)]
    if not real:
        return _empty_svg(
            width=width,
            height=height,
            title=title,
            message="no data",
        )

    xs = [float(c["objectives"][0]) for c in real]
    ys = [float(c["objectives"][1]) for c in real]

    # Axis bounds with a small padding so points aren't on the edge.
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max - x_min < 1e-9:
        x_max = x_min + max(abs(x_min), 1.0) * 0.1 + 1e-3
    if y_max - y_min < 1e-9:
        y_max = y_min + max(abs(y_min), 1.0) * 0.1 + 1e-3
    x_pad = (x_max - x_min) * 0.08
    y_pad = (y_max - y_min) * 0.08
    x_lo, x_hi = x_min - x_pad, x_max + x_pad
    y_lo, y_hi = y_min - y_pad, y_max + y_pad

    # Plot area inside the SVG (leave room for axis labels).
    margin_left = 56
    margin_right = 16
    margin_top = 28
    margin_bottom = 44
    plot_w = max(1, width - margin_left - margin_right)
    plot_h = max(1, height - margin_top - margin_bottom)

    def to_px(value: float, lo: float, hi: float, length: float) -> float:
        if hi - lo < 1e-12:
            return length / 2.0
        return (value - lo) / (hi - lo) * length

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" '
        f'class="pareto-plot-svg" role="img" '
        f'aria-label="{title}">'
    )
    parts.append(f'<title>{title}</title>')
    # Background.
    parts.append(
        f'<rect x="0" y="0" width="{width}" height="{height}" '
        f'fill="#ffffff" stroke="#ccc" />'
    )
    # Title.
    parts.append(
        f'<text x="{width / 2:.1f}" y="{margin_top - 10:.1f}" '
        f'text-anchor="middle" font-family="system-ui, sans-serif" '
        f'font-size="13" fill="#222">{title}</text>'
    )

    # Plot frame (axes).
    plot_x0 = margin_left
    plot_y0 = margin_top
    plot_x1 = margin_left + plot_w
    plot_y1 = margin_top + plot_h
    parts.append(
        f'<rect x="{plot_x0}" y="{plot_y0}" width="{plot_w}" '
        f'height="{plot_h}" fill="#fafafa" stroke="#888" />'
    )
    # X axis (bottom) tick labels.
    parts.append(
        f'<line x1="{plot_x0}" y1="{plot_y1}" x2="{plot_x1}" y2="{plot_y1}" '
        f'stroke="#444" stroke-width="1" />'
    )
    parts.append(
        f'<line x1="{plot_x0}" y1="{plot_y0}" x2="{plot_x0}" y2="{plot_y1}" '
        f'stroke="#444" stroke-width="1" />'
    )
    # Min / max ticks on each axis.
    parts.append(
        f'<text x="{plot_x0}" y="{plot_y1 + 14:.1f}" text-anchor="start" '
        f'font-family="system-ui, sans-serif" font-size="10" '
        f'fill="#444">{_format_number(x_lo)}</text>'
    )
    parts.append(
        f'<text x="{plot_x1}" y="{plot_y1 + 14:.1f}" text-anchor="end" '
        f'font-family="system-ui, sans-serif" font-size="10" '
        f'fill="#444">{_format_number(x_hi)}</text>'
    )
    parts.append(
        f'<text x="{plot_x0 - 6:.1f}" y="{plot_y1 + 4:.1f}" text-anchor="end" '
        f'font-family="system-ui, sans-serif" font-size="10" '
        f'fill="#444">{_format_number(y_lo)}</text>'
    )
    parts.append(
        f'<text x="{plot_x0 - 6:.1f}" y="{plot_y0 + 8:.1f}" text-anchor="end" '
        f'font-family="system-ui, sans-serif" font-size="10" '
        f'fill="#444">{_format_number(y_hi)}</text>'
    )
    # Axis labels.
    parts.append(
        f'<text x="{(plot_x0 + plot_x1) / 2:.1f}" y="{height - 8:.1f}" '
        f'text-anchor="middle" font-family="system-ui, sans-serif" '
        f'font-size="11" fill="#222">resistance proxy</text>'
    )
    parts.append(
        f'<text x="14" y="{(plot_y0 + plot_y1) / 2:.1f}" '
        f'text-anchor="middle" font-family="system-ui, sans-serif" '
        f'font-size="11" fill="#222" '
        f'transform="rotate(-90 14 {(plot_y0 + plot_y1) / 2:.1f})">'
        f'volume delta</text>'
    )

    # Scatter points. The first real candidate is the "winner".
    for index, candidate in enumerate(real):
        cx = plot_x0 + to_px(float(candidate["objectives"][0]), x_lo, x_hi, plot_w)
        cy = plot_y1 - to_px(float(candidate["objectives"][1]), y_lo, y_hi, plot_h)
        is_winner = index == 0
        radius = 6 if is_winner else 4
        fill = "#d24a4a" if is_winner else "#1e7cbf"
        stroke = "#7a1f1f" if is_winner else "#0e4f7f"
        cid = candidate.get("candidate_id") or f"candidate-{index + 1:03d}"
        parts.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{radius}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1" '
            f'opacity="0.9"><title>{cid}: '
            f'r={_format_number(float(candidate["objectives"][0]))}, '
            f'v={_format_number(float(candidate["objectives"][1]))}</title>'
            f'</circle>'
        )

    parts.append('</svg>')
    return "".join(parts)
