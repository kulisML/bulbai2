"""pymoo NSGA-II adapter for the 8-D Kracht bulb parametric space.

Design reference: 2026-04-22-bulbopt-night-optimization-design.md §6.

NSGA-II is a multi-objective evolutionary algorithm that maintains a
population of candidate solutions, selects non-dominated individuals via
the fast-non-dominated-sort + crowding-distance heuristic, and evolves
toward the Pareto front. It is the textbook fit for our problem shape:
8 continuous vars, 2 objectives (drag + volume delta), small population,
modest generation count, black-box evaluator.

This module wraps ``pymoo.algorithms.moo.nsga2.NSGA2`` with a thin
adapter that:

* Translates our :class:`KrachtDesignSpace` bounds into pymoo's ``xl``
  and ``xu`` arrays (preserving KRACHT_PARAMETER_NAMES order).
* Lets the caller plug a simple ``evaluate(list[KrachtVector]) -> list[list[float]]``
  black box instead of subclassing ``pymoo.Problem``.
* Emits a per-generation callback carrying ``{generation, evaluations,
  non_dominated_count}`` so the UI/CLI can render progress.
* Returns a pure-Python :class:`ParetoFront` free of pymoo dependencies
  so downstream code (reporting, checkpoints) doesn't leak the algorithm
  library.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Mapping, Sequence

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize

from bulbopt.optimization.parametric.kracht_space import (
    KRACHT_PARAMETER_NAMES,
    KrachtDesignSpace,
    KrachtVector,
)


def _build_initial_sampling(
    space: KrachtDesignSpace,
    warm_start_vectors: Sequence[KrachtVector],
    population: int,
):
    """Build a pymoo sampling array seeded with ``warm_start_vectors``.

    The returned 2-D ``(population, n_var)`` array is passed as the
    ``sampling`` argument to pymoo's NSGA2 — pymoo accepts a raw ndarray
    and uses its rows as the initial population.

    Remaining rows (after the warm-start points) are filled with uniform
    random samples inside the KrachtDesignSpace bounds. If more
    warm-start vectors are supplied than the population size, the first
    ``population`` are used and the rest dropped.
    """
    rng = np.random.default_rng(0)
    xl = np.array([space.bounds[name][0] for name in KRACHT_PARAMETER_NAMES])
    xu = np.array([space.bounds[name][1] for name in KRACHT_PARAMETER_NAMES])
    rows: list[np.ndarray] = []
    for vector in warm_start_vectors[:population]:
        row = np.array(
            [float(vector.values[name]) for name in KRACHT_PARAMETER_NAMES],
            dtype=float,
        )
        # Clamp to bounds so pymoo doesn't reject the sample.
        row = np.minimum(np.maximum(row, xl), xu)
        rows.append(row)
    while len(rows) < population:
        rows.append(xl + rng.random(len(xl)) * (xu - xl))
    return np.asarray(rows, dtype=float)


EvaluateFn = Callable[[List[KrachtVector]], List[List[float]]]


@dataclass(slots=True, frozen=True)
class ParetoCandidate:
    """One non-dominated individual from the final population."""

    vector: KrachtVector
    objectives: List[float]


@dataclass(slots=True, frozen=True)
class ParetoFront:
    """Final output of :meth:`NSGA2Strategy.optimize`."""

    candidates: List[ParetoCandidate]


# Per-generation snapshot callback (spec 2026-04-22 §10.2). Default ``None``
# keeps the surface backward-compatible: tests that don't need snapshots
# (the bi-objective unit tests) skip the extra plumbing entirely.
GenerationSnapshotFn = Callable[
    [int, List[KrachtVector], List[List[float]], List[ParetoCandidate]],
    None,
]


class NSGA2Strategy:
    """NSGA-II driver for the Kracht space.

    Parameters
    ----------
    population:
        Number of individuals per generation. Also the upper bound on the
        Pareto front size because pymoo keeps the population non-dominated.
    generations:
        Number of generations (the "n_gen" termination criterion).
    seed:
        Seed fed to pymoo's RNG — makes runs reproducible.
    on_generation:
        Optional callback that fires once per generation with a dict
        ``{"generation": int, "evaluations": int, "non_dominated_count": int}``.
    """

    def __init__(
        self,
        population: int = 50,
        generations: int = 20,
        seed: int | None = None,
        on_generation: Callable[[Mapping[str, int]], None] | None = None,
        n_objectives: int = 2,
        warm_start_vectors: Sequence[KrachtVector] | None = None,
        on_generation_snapshot: GenerationSnapshotFn | None = None,
    ) -> None:
        if population < 2:
            raise ValueError("population must be >= 2")
        if generations < 1:
            raise ValueError("generations must be >= 1")
        if n_objectives < 1:
            raise ValueError("n_objectives must be >= 1")
        self.population = int(population)
        self.generations = int(generations)
        self.seed = seed
        self.on_generation = on_generation
        self.n_objectives = int(n_objectives)
        self.warm_start_vectors: List[KrachtVector] = (
            list(warm_start_vectors) if warm_start_vectors else []
        )
        # Per-generation snapshot callback (spec 2026-04-22 §10.2). Fires
        # once per generation with ``(generation_index, population,
        # objectives, pareto)`` so the use case can persist gen-NN
        # directories. Default ``None`` keeps the existing tests green.
        self.on_generation_snapshot = on_generation_snapshot

    def optimize(
        self,
        *,
        space: KrachtDesignSpace,
        evaluate: EvaluateFn,
    ) -> ParetoFront:
        problem = _KrachtProblem(
            space=space,
            evaluate=evaluate,
            n_objectives=self.n_objectives,
        )
        if self.warm_start_vectors:
            sampling = _build_initial_sampling(
                space=space,
                warm_start_vectors=self.warm_start_vectors,
                population=self.population,
            )
        else:
            sampling = FloatRandomSampling()
        algo = NSGA2(
            pop_size=self.population,
            sampling=sampling,
        )
        callback = _GenerationCallback(
            strategy=self,
            problem=problem,
        )
        result = minimize(
            problem,
            algo,
            termination=("n_gen", self.generations),
            seed=self.seed,
            verbose=False,
            callback=callback,
            save_history=False,
        )
        if result.X is None or result.F is None:
            return ParetoFront(candidates=[])

        candidates: List[ParetoCandidate] = []
        X = np.atleast_2d(result.X)
        F = np.atleast_2d(result.F)
        for row, objectives in zip(X, F):
            vector = space.from_array([float(v) for v in row])
            candidates.append(
                ParetoCandidate(
                    vector=vector,
                    objectives=[float(value) for value in objectives],
                )
            )
        return ParetoFront(candidates=candidates)


class _KrachtProblem(Problem):
    """Adapter: pymoo Problem backed by our ``evaluate`` callable."""

    def __init__(
        self,
        *,
        space: KrachtDesignSpace,
        evaluate: EvaluateFn,
        n_objectives: int,
    ) -> None:
        xl = np.array([space.bounds[name][0] for name in KRACHT_PARAMETER_NAMES])
        xu = np.array([space.bounds[name][1] for name in KRACHT_PARAMETER_NAMES])
        super().__init__(n_var=len(xl), n_obj=n_objectives, xl=xl, xu=xu)
        self._space = space
        self._evaluate_fn = evaluate

    def _evaluate(self, X, out, *args, **kwargs):  # pymoo Problem API
        vectors = [
            self._space.from_array([float(v) for v in row])
            for row in np.atleast_2d(X)
        ]
        objectives = self._evaluate_fn(vectors)
        out["F"] = np.asarray(objectives, dtype=float)


class _GenerationCallback(Callback):
    """Per-generation pymoo callback that fans out to user callbacks.

    Two independent fan-outs:

    * ``on_generation`` (legacy) — receives a flat ``{generation,
      evaluations, non_dominated_count}`` dict for UI progress bars.
    * ``on_generation_snapshot`` (spec 2026-04-22 §10.2) — receives the
      full population + objectives matrix + current Pareto front so the
      caller can write ``working/night_optimization/generations/gen-NN/``
      directories. We translate pymoo's raw arrays back into
      :class:`KrachtVector` and :class:`ParetoCandidate` here so callers
      never see pymoo internals.
    """

    def __init__(
        self,
        *,
        strategy: NSGA2Strategy,
        problem: "_KrachtProblem",
    ) -> None:
        super().__init__()
        self._strategy = strategy
        self._problem = problem
        self._generation_index = 0
        self._evaluations_so_far = 0

    def notify(self, algorithm) -> None:  # pymoo API
        pop = algorithm.pop
        if pop is None:
            return
        self._evaluations_so_far += len(pop)

        snapshot_cb = self._strategy.on_generation_snapshot
        legacy_cb = self._strategy.on_generation

        opt = algorithm.opt
        non_dominated_count = 0 if opt is None else len(opt)

        if snapshot_cb is not None:
            try:
                population_vectors, objectives_rows = self._extract_population(pop)
                pareto_candidates = self._extract_pareto(opt)
                snapshot_cb(
                    self._generation_index,
                    population_vectors,
                    objectives_rows,
                    pareto_candidates,
                )
            except Exception:
                # The snapshot writer is the caller's; never let a
                # serialisation/disk error abort the optimisation. The
                # caller is expected to log the failure itself.
                pass

        if legacy_cb is not None:
            legacy_cb(
                {
                    "generation": self._generation_index,
                    "evaluations": self._evaluations_so_far,
                    "non_dominated_count": non_dominated_count,
                }
            )
        self._generation_index += 1

    # ---- helpers ---------------------------------------------------------

    def _extract_population(
        self, pop
    ) -> tuple[List[KrachtVector], List[List[float]]]:
        X = np.atleast_2d(pop.get("X"))
        F_raw = pop.get("F")
        F = np.atleast_2d(F_raw) if F_raw is not None else np.zeros((len(X), 0))
        space = self._problem._space
        vectors = [
            space.from_array([float(v) for v in row]) for row in X
        ]
        objectives = [[float(value) for value in row] for row in F]
        return vectors, objectives

    def _extract_pareto(self, opt) -> List[ParetoCandidate]:
        if opt is None:
            return []
        X = np.atleast_2d(opt.get("X"))
        F_raw = opt.get("F")
        F = np.atleast_2d(F_raw) if F_raw is not None else np.zeros((len(X), 0))
        space = self._problem._space
        candidates: List[ParetoCandidate] = []
        for row, objectives in zip(X, F):
            candidates.append(
                ParetoCandidate(
                    vector=space.from_array([float(v) for v in row]),
                    objectives=[float(value) for value in objectives],
                )
            )
        return candidates
