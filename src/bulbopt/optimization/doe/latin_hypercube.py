"""Latin hypercube design-of-experiments sampler over the Kracht space.

Design reference: 2026-04-26 audit — bootstrap a GP surrogate by running
real OpenFOAM on a small (~30) space-filling DOE and recording the
``(KrachtVector, Cd)`` pairs in the project history store.

The sampler is intentionally implemented with the standard library
``random`` module so the floats it produces are plain ``float``
instances (not numpy scalars) — that keeps the resulting
``KrachtVector`` JSON-serialisable without coercion, matching the
existing :meth:`KrachtDesignSpace.sample` convention.

Algorithm
---------
For each of the eight Kracht dimensions ``d`` we:

1. Divide ``[low_d, high_d]`` into ``n`` equal strata of width
   ``(high_d - low_d) / n``.
2. Draw exactly one sample per stratum (uniform offset inside the
   stratum, so the result is a *random* LHS, not a centre-of-stratum
   midpoint sample).
3. Apply an independent random permutation to the strata index across
   rows. Independence per dimension is what makes the design
   "hypercube": every 1-D projection still visits all ``n`` strata,
   while two different dimensions are decorrelated.

The final result is ``n`` :class:`KrachtVector` instances with
quasi-uniform coverage of the bounded design space. The function is
deterministic for a fixed ``(n, seed)`` pair because every random draw
is routed through a single :class:`random.Random` instance.
"""
from __future__ import annotations

import random
from typing import List

from bulbopt.optimization.parametric.kracht_space import (
    KRACHT_PARAMETER_NAMES,
    KrachtDesignSpace,
    KrachtVector,
)


def latin_hypercube_sample(
    n: int,
    design_space: KrachtDesignSpace,
    seed: int = 0,
) -> List[KrachtVector]:
    """Stratified-random space-filling sample over ``design_space``.

    Parameters
    ----------
    n:
        Number of samples to draw. Must be ``>= 1``.
    design_space:
        Kracht design space whose ``bounds`` define the ``[low_d, high_d]``
        per dimension.
    seed:
        Seed for the internal :class:`random.Random` instance — fixing it
        makes the call deterministic.

    Returns
    -------
    list[KrachtVector]
        ``n`` Kracht vectors. Every dimension visits all ``n`` strata
        exactly once, so the 1-D projection covers the full range.

    Notes
    -----
    The sampler does not enforce any of the coupled engineering
    constraints from :meth:`KrachtDesignSpace.constraint_violations` — it
    is the caller's responsibility to filter or down-weight invalid
    vectors. Keeping LHS bound-only matches the way NSGA-II's initial
    population is generated and lets the DOE driver collect data even at
    the edges of the search space.
    """
    if n < 1:
        raise ValueError(f"latin_hypercube_sample requires n >= 1; got {n}")

    rng = random.Random(seed)

    # 1) For every dimension, generate the n stratum-anchored offsets and
    #    permute them. The result is an n-element list per dimension.
    per_dim_values: dict[str, List[float]] = {}
    for name in KRACHT_PARAMETER_NAMES:
        lo, hi = design_space.bounds[name]
        width = (hi - lo) / float(n)
        # One sample per stratum (random offset within the stratum).
        stratum_values = [
            lo + (i + rng.random()) * width for i in range(n)
        ]
        # Independent random permutation per dimension is what makes the
        # design a Latin hypercube rather than a regular grid.
        rng.shuffle(stratum_values)
        per_dim_values[name] = stratum_values

    # 2) Re-assemble row-major: row i pulls index i from each dimension.
    samples: List[KrachtVector] = []
    for i in range(n):
        values = {name: float(per_dim_values[name][i]) for name in KRACHT_PARAMETER_NAMES}
        samples.append(KrachtVector(values=values))
    return samples
