"""Cascade fidelity strategy — composes NSGA-II search with top-K CFD.

Design reference: 2026-04-22-bulbopt-night-optimization-design.md §5, §6.

The cascade is a thin composition layer:

    Gate(mid)  — used as the NSGA-II objective evaluator. Fast surrogate or
                 analytic proxy; every GA individual is passed through this
                 gate. Cost ≈ population × generations × seconds_per_eval.
    Gate(high) — only the top K non-dominated candidates go through this
                 gate at the end. Cost ≈ K × seconds_per_eval.

The :class:`BudgetScheduler` is informed after each gate so runtime is
tracked per-gate; if the scheduler reports ``is_critical()`` before the
high-fidelity gate starts, the cascade skips it entirely — leaving time
for the pipeline tail (report, package, logs).

The design keeps *what* each gate does abstract (a callable) so we can
wire mid=potentialFoam / high=simpleFoam without touching this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Sequence

from bulbopt.execution.worker.parallel_worker import ParallelWorker
from bulbopt.optimization.parametric.kracht_space import (
    KrachtDesignSpace,
    KrachtVector,
)
from bulbopt.optimization.scheduler.budget_scheduler import (
    BudgetExhausted,
    BudgetScheduler,
)
from bulbopt.optimization.strategies.nsga2_strategy import (
    GenerationSnapshotFn,
    NSGA2Strategy,
    ParetoCandidate,
    ParetoFront,
)


GateEvaluate = Callable[[List[KrachtVector]], List[List[float]]]


@dataclass(slots=True, frozen=True)
class Gate:
    """One fidelity gate: a named evaluator with a cost estimate."""

    name: str
    evaluate: GateEvaluate
    estimated_seconds_per_eval: float


@dataclass(slots=True, frozen=True)
class HighFidelityResult:
    vector: KrachtVector
    objectives: List[float]


@dataclass(slots=True)
class CascadeResult:
    pareto_front: ParetoFront
    high_fidelity_results: List[HighFidelityResult] = field(default_factory=list)
    budget_exhausted: bool = False


class CascadeStrategy:
    """NSGA-II × cascade promotion orchestrator."""

    def __init__(
        self,
        *,
        space: KrachtDesignSpace,
        scheduler: BudgetScheduler,
        population: int,
        generations: int,
        high_fidelity_budget: int,
        mid_gate: Gate,
        high_gate: Gate,
        seed: int | None = None,
        n_objectives: int = 2,
        warm_start_vectors: Sequence[KrachtVector] | None = None,
        on_generation_snapshot: GenerationSnapshotFn | None = None,
        parallel_workers: int = 1,
    ) -> None:
        if high_fidelity_budget < 0:
            raise ValueError("high_fidelity_budget must be >= 0")
        self._space = space
        self._scheduler = scheduler
        self._population = int(population)
        self._generations = int(generations)
        self._high_fidelity_budget = int(high_fidelity_budget)
        self._mid_gate = mid_gate
        self._high_gate = high_gate
        self._seed = seed
        self._n_objectives = int(n_objectives)
        self._warm_start_vectors: List[KrachtVector] = (
            list(warm_start_vectors) if warm_start_vectors else []
        )
        # Spec 2026-04-22 §10.2: per-generation snapshot callback. Forwarded
        # straight to NSGA2Strategy so the use case can persist a
        # ``gen-NN/`` directory after each pymoo generation.
        self._on_generation_snapshot = on_generation_snapshot
        # Spec 2026-04-22 §8: optional fan-out of mid-gate evaluations
        # across worker processes. Default 1 keeps the existing test
        # corpus (282 sequential tests) bit-identical. >1 wraps the
        # mid-gate evaluator with ``ParallelWorker`` at run time so the
        # NSGA-II loop transparently parallelises every generation.
        self._parallel_workers = int(parallel_workers)

    # ---- main entry ------------------------------------------------------

    def run(self) -> CascadeResult:
        pareto_front = self._run_nsga2_with_mid_gate()
        if self._scheduler.is_critical() or self._high_fidelity_budget == 0:
            return CascadeResult(
                pareto_front=pareto_front,
                high_fidelity_results=[],
                budget_exhausted=self._scheduler.is_critical(),
            )
        high_results = self._run_high_fidelity_pass(pareto_front)
        return CascadeResult(
            pareto_front=pareto_front,
            high_fidelity_results=high_results,
            budget_exhausted=False,
        )

    # ---- stages ----------------------------------------------------------

    def _run_nsga2_with_mid_gate(self) -> ParetoFront:
        """Mid-gate evaluator also charges the scheduler per call."""

        # Spec §8: fan-out wrapper. When ``parallel_workers > 1`` the
        # mid-gate evaluator is dispatched across worker processes; the
        # scheduler call itself stays in the main process so the budget
        # bookkeeping is centralised. When ``parallel_workers <= 1`` the
        # ParallelWorker takes the direct-call path so behaviour is
        # bit-identical to the pre-§8 cascade.
        if self._parallel_workers > 1:
            parallel = ParallelWorker(max_workers=self._parallel_workers)

            def mid_evaluate(vectors: List[KrachtVector]) -> List[List[float]]:
                return parallel.evaluate(self._mid_gate.evaluate, vectors)
        else:
            def mid_evaluate(vectors: List[KrachtVector]) -> List[List[float]]:
                return self._mid_gate.evaluate(vectors)

        def accounted_evaluate(vectors: List[KrachtVector]) -> List[List[float]]:
            try:
                self._scheduler.allocate(
                    gate=self._mid_gate.name,
                    seconds=len(vectors) * self._mid_gate.estimated_seconds_per_eval,
                )
            except BudgetExhausted:
                # Return penalty fitness so NSGA-II can still converge
                # (large finite values, never NaN, so sort still works).
                return [[1e9] * self._n_objectives for _ in vectors]
            return mid_evaluate(vectors)

        strategy = NSGA2Strategy(
            population=self._population,
            generations=self._generations,
            seed=self._seed,
            n_objectives=self._n_objectives,
            warm_start_vectors=self._warm_start_vectors,
            on_generation_snapshot=self._on_generation_snapshot,
        )
        return strategy.optimize(space=self._space, evaluate=accounted_evaluate)

    def _run_high_fidelity_pass(
        self,
        pareto_front: ParetoFront,
    ) -> List[HighFidelityResult]:
        """Send top-K non-dominated candidates through the high-fidelity gate."""
        if not pareto_front.candidates:
            return []

        # Sort by first mid-gate objective ascending (minimize). If the two
        # objectives trade off perfectly, crowding picks the extremes —
        # that's fine for a first pass.
        ranked: List[ParetoCandidate] = sorted(
            (
                c
                for c in pareto_front.candidates
                if c.objectives and float(c.objectives[0]) < 1e8
            ),
            key=lambda c: c.objectives[0],
        )
        top = ranked[: self._high_fidelity_budget]
        if not top:
            return []

        vectors = [c.vector for c in top]
        try:
            self._scheduler.allocate(
                gate=self._high_gate.name,
                seconds=len(vectors) * self._high_gate.estimated_seconds_per_eval,
            )
        except BudgetExhausted:
            return []
        high_objectives = self._high_gate.evaluate(vectors)
        return [
            HighFidelityResult(
                vector=vec,
                objectives=[float(value) for value in row],
            )
            for vec, row in zip(vectors, high_objectives)
        ]
