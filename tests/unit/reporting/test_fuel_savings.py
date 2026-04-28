"""Tests for the annual fuel-savings projection.

Industry rule of thumb: a 1% reduction in drag yields roughly a 1%
reduction in fuel consumption. The project_fuel_savings helper applies
that linear approximation to a baseline / winner resistance pair plus a
configurable annual fuel burn and fuel price.
"""
from __future__ import annotations

import math

from bulbopt.reporting.fuel_savings import project_fuel_savings


def test_project_fuel_savings_positive_improvement() -> None:
    """5% drag reduction → ~5% fuel reduction → 1500 t/year @ 30k t baseline."""
    result = project_fuel_savings(
        baseline_resistance=100.0,
        winner_resistance=95.0,
        annual_fuel_burn_t=30_000.0,
        fuel_price_usd_per_t=600.0,
    )

    assert math.isclose(result["drag_reduction_pct"], 5.0, abs_tol=1e-6)
    assert math.isclose(result["fuel_savings_t_per_year"], 1500.0, abs_tol=1e-6)
    assert math.isclose(result["fuel_savings_usd_per_year"], 900_000.0, abs_tol=1e-6)
    assert math.isclose(result["drag_proxy_baseline"], 100.0)
    assert math.isclose(result["drag_proxy_winner"], 95.0)


def test_project_fuel_savings_no_improvement() -> None:
    """Winner equal to baseline → exactly zero savings."""
    result = project_fuel_savings(
        baseline_resistance=42.0,
        winner_resistance=42.0,
    )

    assert result["drag_reduction_pct"] == 0.0
    assert result["fuel_savings_t_per_year"] == 0.0
    assert result["fuel_savings_usd_per_year"] == 0.0


def test_project_fuel_savings_negative_improvement() -> None:
    """Winner worse than baseline returns negative numbers — no clamping."""
    result = project_fuel_savings(
        baseline_resistance=100.0,
        winner_resistance=110.0,
        annual_fuel_burn_t=30_000.0,
        fuel_price_usd_per_t=600.0,
    )

    # 10/100 = -10%
    assert math.isclose(result["drag_reduction_pct"], -10.0, abs_tol=1e-6)
    assert result["fuel_savings_t_per_year"] < 0
    assert result["fuel_savings_usd_per_year"] < 0
    assert math.isclose(
        result["fuel_savings_t_per_year"], -3000.0, abs_tol=1e-6
    )


def test_project_fuel_savings_zero_baseline_returns_zero() -> None:
    """Avoid divide-by-zero when the baseline resistance proxy is zero."""
    result = project_fuel_savings(
        baseline_resistance=0.0,
        winner_resistance=0.0,
    )
    assert result["drag_reduction_pct"] == 0.0
    assert result["fuel_savings_t_per_year"] == 0.0
    assert result["fuel_savings_usd_per_year"] == 0.0
