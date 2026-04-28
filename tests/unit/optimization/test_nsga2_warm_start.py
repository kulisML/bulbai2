"""Tests for NSGA-II warm-start seeding.

Design reference: 2026-04-23-bulbopt-mesh-quality-design.md §4 (L1).

Given a list of prior KrachtVectors, the strategy must include them in the
initial pymoo population rather than starting from a pure random
Latin-hypercube.
"""
from __future__ import annotations

import pytest

from bulbopt.optimization.parametric.kracht_space import (
    KRACHT_PARAMETER_NAMES,
    KrachtDesignSpace,
    KrachtVector,
)
from bulbopt.optimization.strategies.nsga2_strategy import NSGA2Strategy


def _trivial_evaluate(vectors: list[KrachtVector]) -> list[list[float]]:
    return [
        [float(v.values["length_ratio"]), -float(v.values["breadth_ratio"])]
        for v in vectors
    ]


def _middle_vector() -> KrachtVector:
    space = KrachtDesignSpace()
    values = {
        name: 0.5 * (lo + hi) for name, (lo, hi) in space.bounds.items()
    }
    return KrachtVector(values=values)


def test_nsga2_strategy_accepts_warm_start_vectors() -> None:
    """The warm-start vectors must appear in the initial generation that the
    evaluate fn sees. We capture every evaluated vector and check that the
    first pop contains our warm-start point."""
    space = KrachtDesignSpace()
    seed_vector = _middle_vector()
    first_gen_vectors: list[KrachtVector] = []

    def capturing_evaluate(vectors: list[KrachtVector]) -> list[list[float]]:
        # Capture only the first call (= initial generation).
        if not first_gen_vectors:
            first_gen_vectors.extend(vectors)
        return _trivial_evaluate(vectors)

    strategy = NSGA2Strategy(
        population=10,
        generations=2,
        seed=42,
        warm_start_vectors=[seed_vector],
    )
    strategy.optimize(space=space, evaluate=capturing_evaluate)

    # The first-generation pop must include a vector close to our seed.
    assert len(first_gen_vectors) == 10
    matches = [
        v
        for v in first_gen_vectors
        if all(
            abs(v.values[name] - seed_vector.values[name]) < 1e-9
            for name in KRACHT_PARAMETER_NAMES
        )
    ]
    assert len(matches) == 1, "warm-start vector must be in initial population"


def test_nsga2_strategy_warm_start_clamped_to_population() -> None:
    """If more warm-start vectors are passed than the population size, only
    the first ``population`` are used."""
    space = KrachtDesignSpace()
    seeds = space.sample(20, seed=0)  # 20 vectors but pop=6
    first_gen_vectors: list[KrachtVector] = []

    def capturing_evaluate(vectors: list[KrachtVector]) -> list[list[float]]:
        if not first_gen_vectors:
            first_gen_vectors.extend(vectors)
        return _trivial_evaluate(vectors)

    strategy = NSGA2Strategy(
        population=6,
        generations=1,
        seed=3,
        warm_start_vectors=seeds,
    )
    strategy.optimize(space=space, evaluate=capturing_evaluate)

    assert len(first_gen_vectors) == 6


def test_nsga2_strategy_without_warm_start_remains_random() -> None:
    """No regression: without warm-start the behaviour must match the pre-L1
    code path (random sampling)."""
    space = KrachtDesignSpace()
    strategy = NSGA2Strategy(population=10, generations=2, seed=7)
    front = strategy.optimize(space=space, evaluate=_trivial_evaluate)
    assert len(front.candidates) >= 1
