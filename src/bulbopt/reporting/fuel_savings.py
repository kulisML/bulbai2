"""Annual fuel-savings projection from a baseline / winner drag pair.

Audit A finding (2026-04-26): the night-run report quantified resistance
proxies but never told the engineer how that translated into operating
cost. This helper applies the textbook industry rule of thumb that a 1%
reduction in vessel drag yields roughly a 1% reduction in fuel
consumption, then multiplies by a configurable annual fuel burn and
fuel price to produce an annual savings figure in tons and USD.

Defaults are tuned for a Panamax-class container ship (30,000 t/year of
heavy fuel oil at 600 USD/t). Callers can override both for other
vessel classes.
"""
from __future__ import annotations


def project_fuel_savings(
    baseline_resistance: float,
    winner_resistance: float,
    annual_fuel_burn_t: float = 30_000.0,
    fuel_price_usd_per_t: float = 600.0,
) -> dict:
    """Return the projected annual fuel savings for a winner candidate.

    Parameters
    ----------
    baseline_resistance:
        Drag proxy / resistance number for the undeformed baseline hull.
        Should be in the same units as ``winner_resistance``; the only
        thing that matters is the ratio.
    winner_resistance:
        Drag proxy for the winning candidate. If higher than baseline
        the function returns negative savings — engineering-honest, no
        clamping. The template layer is expected to render the
        ``no improvement`` message when that happens.
    annual_fuel_burn_t:
        Operator-supplied annual fuel burn in metric tons. Defaults to
        30,000 t/year (Panamax container ship).
    fuel_price_usd_per_t:
        Fuel price in USD per metric ton. Defaults to 600 USD/t (heavy
        fuel oil reference price 2026 Q1).

    Returns
    -------
    dict
        ``{"drag_reduction_pct": float,
            "fuel_savings_t_per_year": float,
            "fuel_savings_usd_per_year": float,
            "drag_proxy_baseline": float,
            "drag_proxy_winner": float}``.

        ``drag_reduction_pct`` is positive when the winner is better
        than the baseline. When ``baseline_resistance`` is zero we
        return zero across the board to avoid divide-by-zero.
    """
    baseline = float(baseline_resistance)
    winner = float(winner_resistance)
    annual_t = float(annual_fuel_burn_t)
    price = float(fuel_price_usd_per_t)

    if baseline == 0.0:
        return {
            "drag_reduction_pct": 0.0,
            "fuel_savings_t_per_year": 0.0,
            "fuel_savings_usd_per_year": 0.0,
            "drag_proxy_baseline": baseline,
            "drag_proxy_winner": winner,
        }

    # Positive when winner < baseline (good).
    drag_reduction_fraction = (baseline - winner) / baseline
    drag_reduction_pct = drag_reduction_fraction * 100.0
    # 1:1 industry rule of thumb: 1% drag drop ≈ 1% fuel drop.
    fuel_savings_t = drag_reduction_fraction * annual_t
    fuel_savings_usd = fuel_savings_t * price

    return {
        "drag_reduction_pct": drag_reduction_pct,
        "fuel_savings_t_per_year": fuel_savings_t,
        "fuel_savings_usd_per_year": fuel_savings_usd,
        "drag_proxy_baseline": baseline,
        "drag_proxy_winner": winner,
    }
