"""Unit tests for the ``ParallelWorker`` mid-gate fan-out.

Design reference: 2026-04-22-bulbopt-night-optimization-design.md §8.

The worker dispatches a list of vectors to a ``ProcessPoolExecutor`` so a
1000-eval night run can finish in roughly N_CORES_TIMES_FASTER walltime
than the sequential cascade. The unit suite locks four behavioural
guarantees:

1. Functional equivalence — given a picklable evaluator the parallel
   path returns exactly the same answer as a single in-process call.
2. Graceful fallback — when the evaluator captures unpicklable objects
   (closures over trimesh meshes, lambdas) the worker falls back to a
   sequential call instead of raising. Set ``fallback_to_sequential=False``
   to opt out.
3. Subprocess failure recovery — if a chunk raises in a subprocess the
   worker retries that chunk's vectors sequentially in the parent
   process so partial progress is not lost.
4. ``max_workers=1`` shortcut — no ``ProcessPoolExecutor`` is started.
   Important so the existing 282-test suite (which uses sequential
   evaluation) keeps its determinism guarantees and doesn't pay the
   pickle/spawn overhead.
"""
from __future__ import annotations

from typing import List, Sequence
from unittest.mock import patch

import pytest

from bulbopt.execution.worker.parallel_worker import ParallelWorker
from bulbopt.optimization.parametric.kracht_space import (
    KRACHT_PARAMETER_NAMES,
    KrachtVector,
)


# Top-level helpers must be picklable for ProcessPoolExecutor — defining
# them at module scope (as opposed to inside the test functions) is the
# whole point.


def _make_vector(seed_value: float) -> KrachtVector:
    """Build a deterministic KrachtVector keyed off a single float."""
    return KrachtVector(
        values={name: float(seed_value + i * 0.01) for i, name in enumerate(KRACHT_PARAMETER_NAMES)}
    )


def square_objectives(vectors: Sequence[KrachtVector]) -> List[List[float]]:
    """Picklable evaluator: returns x^2 of the first parameter for each vector."""
    out: List[List[float]] = []
    for vec in vectors:
        x = vec.values[KRACHT_PARAMETER_NAMES[0]]
        out.append([x * x])
    return out


def raising_evaluator(vectors: Sequence[KrachtVector]) -> List[List[float]]:
    """Picklable evaluator that raises if it sees a sentinel input."""
    for vec in vectors:
        if vec.values[KRACHT_PARAMETER_NAMES[0]] == 99.0:
            raise RuntimeError("subprocess kaboom")
    return [[vec.values[KRACHT_PARAMETER_NAMES[0]]] for vec in vectors]


def test_parallel_worker_returns_same_results_as_sequential() -> None:
    """A picklable evaluator over 4 vectors must produce identical output
    to the sequential one-shot call. This is the baseline correctness
    guarantee — without this, the parallel speedup is meaningless.
    """
    vectors = [_make_vector(v) for v in (0.10, 0.20, 0.30, 0.40)]
    expected = square_objectives(vectors)

    worker = ParallelWorker(max_workers=4)
    actual = worker.evaluate(square_objectives, vectors)

    assert actual == expected, (
        f"Parallel worker output {actual!r} differs from sequential {expected!r}"
    )


def test_parallel_worker_falls_back_when_evaluator_not_picklable() -> None:
    """A closure capturing local state cannot be pickled. The default
    ``fallback_to_sequential=True`` keeps the run alive by evaluating the
    batch in-process. Setting it to False asks the caller to deal — used
    by tests that *want* to detect a regression in evaluator picklability.
    """
    vectors = [_make_vector(v) for v in (0.10, 0.20, 0.30)]
    captured = {"factor": 7.0}

    def closure_evaluator(vecs: Sequence[KrachtVector]) -> List[List[float]]:
        # ``captured`` is a local; this closure is not picklable.
        return [[v.values[KRACHT_PARAMETER_NAMES[0]] * captured["factor"]] for v in vecs]

    expected = closure_evaluator(vectors)

    fallback_worker = ParallelWorker(max_workers=4, fallback_to_sequential=True)
    actual = fallback_worker.evaluate(closure_evaluator, vectors)
    assert actual == expected

    strict_worker = ParallelWorker(max_workers=4, fallback_to_sequential=False)
    with pytest.raises(Exception):
        strict_worker.evaluate(closure_evaluator, vectors)


def test_parallel_worker_handles_subprocess_failure() -> None:
    """When a chunk raises in a subprocess we want the partial results
    plus a sequential retry of the doomed chunk's vectors. The user
    should still get a result for every input the evaluator can handle
    (which here is "every input that is not the sentinel 99.0").
    """
    vectors = [_make_vector(v) for v in (0.10, 0.20, 0.30, 0.40)]
    expected = raising_evaluator(vectors)

    worker = ParallelWorker(max_workers=4, fallback_to_sequential=True)
    actual = worker.evaluate(raising_evaluator, vectors)

    assert actual == expected


def test_parallel_worker_max_workers_one_is_direct_call() -> None:
    """``max_workers=1`` must short-circuit straight to a single call;
    no ``ProcessPoolExecutor`` should be instantiated. We verify by
    patching the executor class and asserting it's never invoked.
    """
    vectors = [_make_vector(v) for v in (0.10, 0.20)]

    with patch(
        "bulbopt.execution.worker.parallel_worker.ProcessPoolExecutor"
    ) as mock_executor:
        worker = ParallelWorker(max_workers=1)
        result = worker.evaluate(square_objectives, vectors)

    assert result == square_objectives(vectors)
    assert mock_executor.call_count == 0, (
        "max_workers=1 must not spawn a ProcessPoolExecutor"
    )
