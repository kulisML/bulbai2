"""Tests for the cascade fidelity strategy.

Design reference: 2026-04-22-bulbopt-night-optimization-design.md §5.
The cascade runs NSGA-II with a cheap mid-fidelity surrogate for its
objective, then promotes top-K non-dominated candidates to a more
expensive gate (e.g. simpleFoam), then top-M from that to verification.
"""
from __future__ import annotations

import pytest

from bulbopt.optimization.parametric.kracht_space import (
    KrachtDesignSpace,
    KrachtVector,
)
from bulbopt.optimization.scheduler.budget_scheduler import BudgetScheduler
from bulbopt.optimization.strategies.cascade_strategy import (
    CascadeResult,
    CascadeStrategy,
    Gate,
)


def _mid_evaluate(vectors: list[KrachtVector]) -> list[list[float]]:
    """Mid-fidelity surrogate: minimize length, maximize breadth."""
    return [
        [float(v.values["length_ratio"]), -float(v.values["breadth_ratio"])]
        for v in vectors
    ]


def test_cascade_runs_search_then_promotes_top_k_to_high_fidelity() -> None:
    """End-to-end cascade: NSGA-II search yields a Pareto front; top-K by
    first objective go to the high-fidelity gate; that gate's results are
    recorded and exposed to the caller."""
    high_calls: list[KrachtVector] = []

    def high_evaluate(vectors: list[KrachtVector]) -> list[list[float]]:
        for v in vectors:
            high_calls.append(v)
        # High-fidelity is "correct" — returns real drag.
        return [[0.1 * float(v.values["length_ratio"]), 0.0] for v in vectors]

    space = KrachtDesignSpace()
    scheduler = BudgetScheduler(runtime_budget_hours=1.0, clock=lambda: 0.0)

    strategy = CascadeStrategy(
        space=space,
        scheduler=scheduler,
        population=10,
        generations=3,
        high_fidelity_budget=3,
        mid_gate=Gate(name="mid", evaluate=_mid_evaluate, estimated_seconds_per_eval=0.1),
        high_gate=Gate(name="high", evaluate=high_evaluate, estimated_seconds_per_eval=5.0),
        seed=42,
    )

    result = strategy.run()

    assert isinstance(result, CascadeResult)
    assert len(result.pareto_front.candidates) >= 1
    # Top-K promoted count never exceeds high_fidelity_budget.
    assert len(result.high_fidelity_results) <= 3
    assert len(high_calls) <= 3
    # Top-K promoted count never exceeds the Pareto-front size.
    assert len(result.high_fidelity_results) <= len(result.pareto_front.candidates)
    # At least one candidate must be promoted when budget allows.
    assert len(result.high_fidelity_results) >= 1
    assert result.budget_exhausted is False


def test_cascade_short_circuits_when_scheduler_becomes_critical() -> None:
    """When the scheduler reports critical, the cascade must skip the
    high-fidelity gate entirely (to leave time for the pipeline tail —
    HTML report, package, etc.)."""
    clock_value = [0.0]

    def fake_clock() -> float:
        return clock_value[0]

    scheduler = BudgetScheduler(
        runtime_budget_hours=1.0,
        critical_threshold=0.5,
        clock=fake_clock,
    )
    # Pre-burn most of the budget before the cascade even starts.
    scheduler.allocate(gate="prep", seconds=2000)  # 55% used → critical

    def unused_high(vectors: list[KrachtVector]) -> list[list[float]]:
        pytest.fail("high-fidelity gate must not be called when critical")

    space = KrachtDesignSpace()
    strategy = CascadeStrategy(
        space=space,
        scheduler=scheduler,
        population=10,
        generations=2,
        high_fidelity_budget=3,
        mid_gate=Gate(name="mid", evaluate=_mid_evaluate, estimated_seconds_per_eval=0.1),
        high_gate=Gate(name="high", evaluate=unused_high, estimated_seconds_per_eval=10.0),
        seed=3,
    )

    result = strategy.run()
    assert result.high_fidelity_results == []
    assert result.budget_exhausted is True


def test_cascade_passes_three_objectives_end_to_end() -> None:
    """L2: when n_objectives=3 the mid-gate must return triples and
    NSGA-II must preserve them through to the Pareto front."""
    def three_obj_evaluate(vectors: list[KrachtVector]) -> list[list[float]]:
        return [
            [
                float(v.values["length_ratio"]),
                -float(v.values["breadth_ratio"]),
                0.1 * float(v.values["height_ratio"]),  # mesh-quality-like proxy
            ]
            for v in vectors
        ]

    def three_obj_high(vectors: list[KrachtVector]) -> list[list[float]]:
        return [[0.1, 0.0, 0.0] for _ in vectors]

    space = KrachtDesignSpace()
    scheduler = BudgetScheduler(runtime_budget_hours=1.0, clock=lambda: 0.0)

    strategy = CascadeStrategy(
        space=space,
        scheduler=scheduler,
        population=10,
        generations=2,
        high_fidelity_budget=2,
        mid_gate=Gate(
            name="mid",
            evaluate=three_obj_evaluate,
            estimated_seconds_per_eval=0.01,
        ),
        high_gate=Gate(
            name="high",
            evaluate=three_obj_high,
            estimated_seconds_per_eval=0.05,
        ),
        seed=42,
        n_objectives=3,
    )

    result = strategy.run()
    assert len(result.pareto_front.candidates) >= 1
    for candidate in result.pareto_front.candidates:
        assert len(candidate.objectives) == 3
    for hf in result.high_fidelity_results:
        assert len(hf.objectives) == 3


def test_cascade_records_gate_timings_in_scheduler() -> None:
    space = KrachtDesignSpace()
    scheduler = BudgetScheduler(runtime_budget_hours=1.0, clock=lambda: 0.0)

    def high_evaluate(vectors: list[KrachtVector]) -> list[list[float]]:
        return [[0.0, 0.0] for _ in vectors]

    strategy = CascadeStrategy(
        space=space,
        scheduler=scheduler,
        population=10,
        generations=2,
        high_fidelity_budget=2,
        mid_gate=Gate(name="mid", evaluate=_mid_evaluate, estimated_seconds_per_eval=0.2),
        high_gate=Gate(name="high", evaluate=high_evaluate, estimated_seconds_per_eval=1.0),
        seed=1,
    )

    strategy.run()

    gate_timings = scheduler.gate_timings()
    assert "mid" in gate_timings
    assert "high" in gate_timings
    assert gate_timings["mid"] > 0.0
    assert gate_timings["high"] > 0.0


def test_cascade_skips_penalty_candidates_before_high_fidelity() -> None:
    """Penalty-only Pareto candidates should not consume high-fidelity budget."""
    high_calls: list[KrachtVector] = []

    def penalty_mid(vectors: list[KrachtVector]) -> list[list[float]]:
        return [[1e9, 1e9, 1e9] for _ in vectors]

    def high_evaluate(vectors: list[KrachtVector]) -> list[list[float]]:
        high_calls.extend(vectors)
        return [[0.0, 0.0, 0.0] for _ in vectors]

    scheduler = BudgetScheduler(runtime_budget_hours=1.0, clock=lambda: 0.0)
    strategy = CascadeStrategy(
        space=KrachtDesignSpace(),
        scheduler=scheduler,
        population=6,
        generations=2,
        high_fidelity_budget=3,
        mid_gate=Gate(name="mid", evaluate=penalty_mid, estimated_seconds_per_eval=0.01),
        high_gate=Gate(name="high", evaluate=high_evaluate, estimated_seconds_per_eval=10.0),
        seed=11,
        n_objectives=3,
    )

    result = strategy.run()

    assert result.high_fidelity_results == []
    assert high_calls == []
    assert "high" not in scheduler.gate_timings()
