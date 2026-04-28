"""Tests for the pymoo NSGA-II strategy adapter.

Design reference: 2026-04-22-bulbopt-night-optimization-design.md §6.
The strategy turns an ``evaluate_fn(kracht_vectors) -> objectives`` into a
Pareto front by running NSGA-II for a fixed population × generations
budget. Tests use small populations and toy analytic objectives so they
run fast and deterministically.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from bulbopt.optimization.parametric.kracht_space import (
    KRACHT_PARAMETER_NAMES,
    KrachtDesignSpace,
    KrachtVector,
)
from bulbopt.optimization.strategies.nsga2_strategy import (
    NSGA2Strategy,
    ParetoFront,
)


def _bi_objective_evaluate(vectors: list[KrachtVector]) -> list[list[float]]:
    """Simple bi-objective test function with known Pareto behaviour:

      f1 = length_ratio               (minimize - shorter is better)
      f2 = -breadth_ratio             (minimize - wider is better)

    Because these two metrics trade off inside the Kracht box, the true
    Pareto front is the bottom-left edge of the feasible region.
    """
    objectives: list[list[float]] = []
    for vector in vectors:
        v = vector.values
        f1 = float(v["length_ratio"])
        f2 = -float(v["breadth_ratio"])
        objectives.append([f1, f2])
    return objectives


def test_nsga2_strategy_runs_with_toy_problem_and_returns_pareto_front() -> None:
    space = KrachtDesignSpace()
    strategy = NSGA2Strategy(population=20, generations=5, seed=1)

    front = strategy.optimize(space=space, evaluate=_bi_objective_evaluate)

    assert isinstance(front, ParetoFront)
    assert len(front.candidates) >= 2, "Expected at least two non-dominated points"
    assert len(front.candidates) <= 20  # cannot exceed population size
    # Every candidate must be a valid Kracht vector.
    for candidate in front.candidates:
        assert space.validate(candidate.vector)
        assert len(candidate.objectives) == 2


def test_nsga2_strategy_respects_population_and_generations_budget() -> None:
    """Total evaluations must equal population × generations — otherwise
    the budget scheduler can't reason about cost."""
    space = KrachtDesignSpace()
    calls: dict[str, int] = {"evals": 0, "calls": 0}

    def counting_evaluate(vectors: list[KrachtVector]) -> list[list[float]]:
        calls["calls"] += 1
        calls["evals"] += len(vectors)
        return _bi_objective_evaluate(vectors)

    strategy = NSGA2Strategy(population=10, generations=3, seed=5)
    strategy.optimize(space=space, evaluate=counting_evaluate)

    assert calls["evals"] == 10 * 3


def test_nsga2_strategy_is_deterministic_under_fixed_seed() -> None:
    space = KrachtDesignSpace()
    strategy_a = NSGA2Strategy(population=10, generations=3, seed=42)
    strategy_b = NSGA2Strategy(population=10, generations=3, seed=42)

    front_a = strategy_a.optimize(space=space, evaluate=_bi_objective_evaluate)
    front_b = strategy_b.optimize(space=space, evaluate=_bi_objective_evaluate)

    # Same seed → same Pareto front.
    assert len(front_a.candidates) == len(front_b.candidates)
    a_objectives = sorted(tuple(c.objectives) for c in front_a.candidates)
    b_objectives = sorted(tuple(c.objectives) for c in front_b.candidates)
    for a, b in zip(a_objectives, b_objectives):
        assert a == pytest.approx(b, rel=1e-9, abs=1e-9)


def test_nsga2_strategy_pareto_points_are_non_dominated() -> None:
    """Classic NSGA-II invariant: no point in the returned front is
    dominated by another point in the same front."""
    space = KrachtDesignSpace()
    strategy = NSGA2Strategy(population=30, generations=8, seed=11)
    front = strategy.optimize(space=space, evaluate=_bi_objective_evaluate)

    pts = np.array([c.objectives for c in front.candidates])
    n = len(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Is j dominated by i?
            diff = pts[i] - pts[j]
            if np.all(diff <= 0) and np.any(diff < 0):
                pytest.fail(
                    f"Point {j} ({pts[j]}) is dominated by point {i} ({pts[i]})"
                )


def test_nsga2_strategy_fires_per_generation_callback() -> None:
    """For UI progress reporting, the strategy must emit one callback per
    generation with generation index and best objective so far."""
    space = KrachtDesignSpace()
    events: list[dict] = []

    def on_generation(event: dict) -> None:
        events.append(event)

    strategy = NSGA2Strategy(
        population=10,
        generations=4,
        seed=7,
        on_generation=on_generation,
    )
    strategy.optimize(space=space, evaluate=_bi_objective_evaluate)

    assert len(events) == 4
    for index, event in enumerate(events):
        assert event["generation"] == index
        assert "evaluations" in event
        assert "non_dominated_count" in event


def test_nsga2_strategy_invokes_generation_callback_per_generation() -> None:
    """Spec §10.2 — for per-generation snapshots the night flow needs a
    richer callback: each generation hands back the full population, the
    objectives matrix, and the current Pareto front so the use case can
    persist them under ``working/night_optimization/generations/``.

    The callback fires exactly once per generation with monotonically
    increasing ``generation_index`` starting at 0.
    """
    space = KrachtDesignSpace()
    events: list[dict] = []

    def on_generation_snapshot(
        generation_index: int,
        population: list[KrachtVector],
        objectives: list[list[float]],
        pareto: list,
    ) -> None:
        events.append(
            {
                "generation_index": int(generation_index),
                "n_individuals": len(population),
                "n_objectives_rows": len(objectives),
                "n_pareto": len(pareto),
            }
        )

    strategy = NSGA2Strategy(
        population=8,
        generations=3,
        seed=11,
        on_generation_snapshot=on_generation_snapshot,
    )
    strategy.optimize(space=space, evaluate=_bi_objective_evaluate)

    assert len(events) == 3
    indices = [event["generation_index"] for event in events]
    assert indices == [0, 1, 2]
    for event in events:
        assert event["n_individuals"] == 8
        assert event["n_objectives_rows"] == 8
        # Pareto front size cannot exceed the population, but must be >= 1
        # because at least one point is non-dominated.
        assert 1 <= event["n_pareto"] <= 8
