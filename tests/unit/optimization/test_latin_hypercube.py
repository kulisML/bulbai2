"""Tests for Latin hypercube DOE sampler.

Design reference: 2026-04-26 audit — bootstrapping the GP surrogate
with a Latin hypercube design of experiments over the Kracht space.

The sampler stratifies each dimension into ``n`` equal slices and places
exactly one sample per slice; rows are then permuted independently per
dimension so that any 1-D projection sees full coverage.
"""
from __future__ import annotations

import pytest

from bulbopt.optimization.doe.latin_hypercube import latin_hypercube_sample
from bulbopt.optimization.parametric.kracht_space import (
    KRACHT_PARAMETER_NAMES,
    KrachtDesignSpace,
    KrachtVector,
)


def test_lhs_returns_n_vectors_with_deterministic_seed() -> None:
    """Same seed → identical batch; different seeds → different batches."""
    space = KrachtDesignSpace()

    batch_a = latin_hypercube_sample(10, space, seed=42)
    batch_b = latin_hypercube_sample(10, space, seed=42)

    assert len(batch_a) == 10
    assert all(isinstance(v, KrachtVector) for v in batch_a)
    # Determinism: bit-equal across two calls with the same seed.
    assert [v.values for v in batch_a] == [v.values for v in batch_b]

    # Reseeding produces a different batch (almost surely).
    batch_c = latin_hypercube_sample(10, space, seed=43)
    assert [v.values for v in batch_a] != [v.values for v in batch_c]


def test_lhs_respects_design_space_bounds() -> None:
    """Every drawn value must lie inside its declared ``[lo, hi]`` range."""
    space = KrachtDesignSpace()

    batch = latin_hypercube_sample(30, space, seed=7)

    assert len(batch) == 30
    for vector in batch:
        # Every Kracht parameter must be present.
        assert set(vector.values.keys()) == set(KRACHT_PARAMETER_NAMES)
        for name, value in vector.values.items():
            lo, hi = space.bounds[name]
            assert lo <= value <= hi, f"{name}={value} outside [{lo}, {hi}]"


def test_lhs_provides_full_dimension_coverage() -> None:
    """Stratification check: each 1-D projection should span >90 % of the bound range.

    With ``n=20`` strata per axis the LHS algorithm places exactly one
    sample per slice, so the projection of all samples on any single
    dimension covers the entire range up to one stratum width
    (``1/n = 5 %``). A coverage ratio of ``(max - min) / (hi - lo) > 0.9``
    therefore both validates that the strata exist *and* that they were
    actually used (not all collapsed at the slice midpoint).
    """
    space = KrachtDesignSpace()

    n = 20
    batch = latin_hypercube_sample(n, space, seed=0)

    for name in KRACHT_PARAMETER_NAMES:
        lo, hi = space.bounds[name]
        values = [v.values[name] for v in batch]
        coverage = (max(values) - min(values)) / (hi - lo)
        assert coverage > 0.9, (
            f"dimension {name!r} coverage={coverage:.3f}; "
            f"expected >0.90 with n={n} strata"
        )
