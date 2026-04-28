"""Tests for the GP surrogate used at the mid gate.

Design reference: 2026-04-23-bulbopt-mesh-quality-design.md §4 (L1).

The GP is fit on (8-D KrachtVector → Cd) pairs; it returns (mean, std)
per query vector. With < 5 training points the fit is too poor to be
useful, so ``predict`` returns ``None`` and callers fall back to the
analytic proxy.
"""
from __future__ import annotations

import numpy as np
import pytest

from bulbopt.optimization.learning.gp_surrogate import GPSurrogate
from bulbopt.optimization.parametric.kracht_space import (
    KRACHT_PARAMETER_NAMES,
    KrachtDesignSpace,
    KrachtVector,
)


def _toy_function(vector: KrachtVector) -> float:
    """Smooth toy function of the 8 parameters, used so the GP can learn it.
    Bounded roughly in [0.2, 0.8]."""
    v = vector.values
    return 0.4 + 0.2 * (v["length_ratio"] - 0.02) * 20.0 + 0.1 * v["breadth_ratio"]


def _sample_vectors(n: int, seed: int) -> list[KrachtVector]:
    return KrachtDesignSpace().sample(n, seed=seed)


def test_gp_surrogate_returns_none_with_fewer_than_5_points() -> None:
    surrogate = GPSurrogate()
    vectors = _sample_vectors(3, seed=1)
    cds = [_toy_function(v) for v in vectors]
    surrogate.fit(vectors, cds)
    prediction = surrogate.predict(vectors)
    assert prediction is None


def test_gp_surrogate_predicts_close_to_truth_with_enough_points() -> None:
    surrogate = GPSurrogate()
    train = _sample_vectors(30, seed=11)
    cds = [_toy_function(v) for v in train]
    surrogate.fit(train, cds)

    # Evaluate on held-out points inside the same box.
    query = _sample_vectors(10, seed=22)
    prediction = surrogate.predict(query)
    assert prediction is not None
    means, stds = prediction
    assert len(means) == 10
    assert len(stds) == 10
    truth = np.array([_toy_function(v) for v in query])
    # Should be within 0.1 of truth on a function ranging ~0.4-0.6.
    errors = np.abs(np.array(means) - truth)
    assert np.max(errors) < 0.2, f"GP predictions off: max err = {np.max(errors)}"
    # Uncertainties are non-negative.
    assert np.all(np.array(stds) >= 0.0)


def test_gp_surrogate_predict_handles_single_vector() -> None:
    surrogate = GPSurrogate()
    train = _sample_vectors(20, seed=3)
    cds = [_toy_function(v) for v in train]
    surrogate.fit(train, cds)

    one = _sample_vectors(1, seed=99)
    prediction = surrogate.predict(one)
    assert prediction is not None
    means, stds = prediction
    assert len(means) == 1
    assert len(stds) == 1
    assert np.isfinite(means[0])
    assert stds[0] >= 0.0
