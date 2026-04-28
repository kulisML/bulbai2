"""Tests for the night-report extras: inline Pareto SVG plot and
fuel-savings projection (audit A finding 2026-04-26).

The night-run HTML must include both a `<section class="pareto-plot">`
with an inline SVG scatter, and a `<section class="fuel-savings">`
showing drag reduction %, t/year and USD/year for the winner.

These tests run a small night-optimization end-to-end with a tiny
population so the whole flow stays well under a second, and assert that
the rendered HTML contains the two new CSS-class anchors.
"""
from __future__ import annotations

from pathlib import Path

import trimesh

from bulbopt.application.contracts.models import CreateCaseCommand
from bulbopt.application.use_cases.run_night_optimization import (
    NightOptimizationConfig,
    run_night_optimization,
)


def _write_watertight_stl(path: Path) -> None:
    mesh = trimesh.creation.box(extents=(4.0, 1.5, 1.0))
    path.write_bytes(trimesh.exchange.stl.export_stl(mesh))


def test_render_night_report_includes_pareto_plot_section(tmp_path: Path) -> None:
    """The rendered HTML must include the new pareto-plot SVG section
    and the fuel-savings section anchors so downstream consumers (UI,
    PDFs, audits) can find them by CSS class."""
    source_path = tmp_path / "hull.stl"
    _write_watertight_stl(source_path)

    summary = run_night_optimization(
        project_root=tmp_path / "projects",
        command=CreateCaseCommand(
            case_name="night-extras",
            source_path=str(source_path),
            vessel_length_m=142.0,
            vessel_beam_m=19.1,
            vessel_draft_m=6.0,
            displacement_t=8420.0,
            speed_knots=[18.0, 20.0],
        ),
        config=NightOptimizationConfig(
            population=6,
            generations=2,
            high_fidelity_budget=1,
            runtime_budget_hours=1.0,
            seed=7,
            mid_gate_estimated_seconds_per_eval=0.001,
            high_gate_estimated_seconds_per_eval=0.005,
        ),
    )

    case_dir = tmp_path / "projects" / summary.case_id
    report_path = case_dir / "outputs" / "reports" / "night_report.html"
    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")

    # Two new sections must be present.
    assert "pareto-plot" in report_text
    assert "fuel-savings" in report_text
    # The plot must be an inline SVG element.
    assert "<svg" in report_text and "</svg>" in report_text
