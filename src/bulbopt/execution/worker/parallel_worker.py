"""Parallel mid-gate evaluator using ``ProcessPoolExecutor``.

Design reference: 2026-04-22-bulbopt-night-optimization-design.md §8.

The cascade currently calls ``mid_gate.evaluate(vectors)`` synchronously
on the main thread. For an 8-D, 50-individual, 20-generation NSGA-II run
the population-times-generations product is 1000 evaluations; at the
designed 5 s mid-gate cost a sequential pass burns roughly 83 minutes.
On a 4-core laptop a parallel fan-out should drop that to about 20 min,
which is the difference between "kick off before bed" and "the user
gives up and goes back to manual hull tweaking".

Why not just hand the evaluator to ``concurrent.futures`` blindly?

* Mid-gate evaluators capture trimesh meshes and FFD deformers in a
  closure (see ``_mid_gate_evaluator`` in ``run_night_optimization``).
  Closures over locally-scoped objects are not picklable, and
  ``ProcessPoolExecutor`` requires picklability to ship work to a
  child interpreter. We test for this up-front and fall back to a
  single in-process call instead of crashing.
* Subprocess workers occasionally die (Windows OOM, segfault inside a
  numpy native extension, ...). When that happens we don't want the
  whole night-run to abort — we want to mark the failed chunk for a
  sequential retry in the parent process and keep going.
* The serial ``max_workers=1`` shortcut is not just a perf hint: it's
  a determinism contract. The cascade must produce bit-identical
  outputs at ``parallel_workers=1`` so the existing test corpus is
  unaffected and so a deterministic seed still buys reproducibility.

This module is intentionally dependency-free: ``concurrent.futures``,
``pickle``, and ``sys`` are all stdlib. We deliberately don't pull in
``cloudpickle`` to widen the picklability set; if a user wants closure
support they can refactor their evaluator into a top-level callable
plus an explicit context object.
"""
from __future__ import annotations

import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, List, Sequence

from bulbopt.optimization.parametric.kracht_space import KrachtVector


Evaluator = Callable[[Sequence[KrachtVector]], List[List[float]]]


def _evenly_split(items: Sequence[KrachtVector], n_chunks: int) -> List[List[KrachtVector]]:
    """Split ``items`` into ``n_chunks`` near-equal-sized lists.

    Used to fan out the work across worker processes. We don't bother
    with anything fancier than "divmod" because the chunks are
    long-running compared to the dispatch overhead, so a slight
    imbalance at the tail is fine.
    """
    if n_chunks <= 0:
        raise ValueError("n_chunks must be >= 1")
    total = len(items)
    if total == 0:
        return []
    base, rem = divmod(total, n_chunks)
    chunks: List[List[KrachtVector]] = []
    start = 0
    for i in range(n_chunks):
        size = base + (1 if i < rem else 0)
        if size == 0:
            continue
        chunks.append(list(items[start : start + size]))
        start += size
    return chunks


def _is_picklable(obj: object) -> bool:
    """Cheap roundtrip pickle test.

    ``ProcessPoolExecutor`` will pickle the callable on submit. We do the
    same check up-front so we can choose the fallback path *before*
    spawning any subprocess (cheaper, and avoids the noisy traceback that
    a late ``PicklingError`` in the executor would emit).
    """
    try:
        pickle.dumps(obj)
        return True
    except Exception:
        return False


class ParallelWorker:
    """Run mid-gate evaluator over a list of vectors using ProcessPoolExecutor.

    Falls back to sequential evaluation when:
      - max_workers <= 1
      - the evaluator is not picklable (e.g. a closure capturing trimesh objects)
      - any subprocess raises (we keep the partial results, log the failure,
        and finish the rest sequentially in-process)
    """

    def __init__(
        self,
        max_workers: int = 4,
        fallback_to_sequential: bool = True,
    ) -> None:
        self._max_workers = int(max_workers)
        self._fallback_to_sequential = bool(fallback_to_sequential)

    def evaluate(
        self,
        evaluator: Evaluator,
        vectors: Sequence[KrachtVector],
    ) -> List[List[float]]:
        """Apply ``evaluator`` to ``vectors`` and return objective rows.

        The result has exactly ``len(vectors)`` rows in the same order
        as the input. This is part of the contract — NSGA-II indexes
        objectives back into its population by position.
        """
        if not vectors:
            return []

        # Path 1: ``max_workers <= 1`` — direct in-process call, no
        # ProcessPoolExecutor created. Important for the 282-test
        # determinism baseline.
        if self._max_workers <= 1:
            return list(evaluator(list(vectors)))

        # Path 2: evaluator not picklable. Fallback or raise.
        if not _is_picklable(evaluator):
            if not self._fallback_to_sequential:
                # Surface a real PicklingError so callers can detect this
                # case explicitly. We try one more pickle (without a
                # try/except) so the original exception chain is preserved.
                pickle.dumps(evaluator)
                # If pickle.dumps inexplicably succeeds the second time we
                # still want to refuse — a flaky pickle is worse than a
                # noisy raise.
                raise pickle.PicklingError(
                    "Evaluator is not reliably picklable; cannot use a process pool"
                )
            return list(evaluator(list(vectors)))

        # Path 3: real fan-out across workers.
        chunks = _evenly_split(vectors, self._max_workers)
        # Per-chunk slot for the eventual List[List[float]] result.
        chunk_results: List[List[List[float]] | None] = [None] * len(chunks)
        failed_indices: List[int] = []

        with ProcessPoolExecutor(max_workers=self._max_workers) as pool:
            futures = [
                pool.submit(evaluator, chunk) for chunk in chunks
            ]
            for idx, fut in enumerate(futures):
                try:
                    chunk_results[idx] = fut.result()
                except Exception as exc:  # broad on purpose: subprocess can die for any reason
                    if not self._fallback_to_sequential:
                        raise
                    print(
                        f"[ParallelWorker] chunk {idx} failed in subprocess: "
                        f"{type(exc).__name__}: {exc}; retrying sequentially",
                        file=sys.stderr,
                    )
                    failed_indices.append(idx)

        # Sequentially retry the chunks that died in their subprocess. We
        # do this after the pool exits so we're not racing against a
        # possibly-corrupt executor (e.g. one that lost a worker process).
        for idx in failed_indices:
            chunk_results[idx] = list(evaluator(chunks[idx]))

        # Stitch results back together in original vector order.
        flattened: List[List[float]] = []
        for rows in chunk_results:
            if rows is None:
                # Shouldn't happen — failed_indices were retried above —
                # but defend against it anyway so we never silently
                # truncate the population NSGA-II expects to receive.
                raise RuntimeError(
                    "ParallelWorker internal error: missing chunk result after retry"
                )
            flattened.extend(rows)
        return flattened
