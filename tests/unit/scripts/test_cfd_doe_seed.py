"""Tests for the parallel + resume flags in ``scripts/cfd_doe_seed.py``.

The DOE seed script is the bootstrap that feeds the GP surrogate. Two
operational concerns drove this test file:

1. **Resume idempotency.** A real DOE-30 sweep of OpenFOAM cases runs
   for ~2 hours; if it gets killed at sample 17/30 we must be able to
   restart and skip the 17 already-completed samples rather than
   re-burn 30 minutes of CPU. Use a stable seed: the LHS draw at
   ``--seed=0`` is bit-identical across runs, so "rows already in
   ``history.jsonl`` for the LHS-0 vectors" is a well-defined skip set.

2. **Parallel determinism.** ``--parallel-workers=N`` must produce the
   same set of ``(vector, cd)`` pairs as ``--parallel-workers=1`` —
   only the order of completion may differ. Without this guarantee
   reproducibility goes out the window the moment a user crank the
   workers up.

These tests deliberately mock the OpenFOAM adapter chain and the
baseline preparation step. The behavioural surface they exercise is the
script's CLI plumbing — the LHS sample iteration, the resume skip
logic, and the parallel fan-out / aggregation — not the OpenFOAM
solver.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


import cfd_doe_seed  # type: ignore  # noqa: E402

from bulbopt.optimization.doe.latin_hypercube import latin_hypercube_sample  # noqa: E402
from bulbopt.optimization.learning.history_store import HistoryStore  # noqa: E402
from bulbopt.optimization.parametric.kracht_space import (  # noqa: E402
    KRACHT_PARAMETER_NAMES,
    KrachtDesignSpace,
)


# ---------- shared helpers -------------------------------------------------


def _fake_baseline(case_dir: Path, source_stl: Path):
    """Return a placeholder (mesh, region) without doing real geometry repair.

    The script only needs the (mesh, region) pair to forward to the
    deformer and OpenFOAM builder — both of which are mocked out in
    these tests. Returning lightweight sentinels keeps the test fast.
    """
    case_dir.mkdir(parents=True, exist_ok=True)
    return ("mesh-sentinel", {"region": "stub"})


def _fake_run_factory(records: List[Dict[str, Any]]):
    """Build a picklable-shaped stand-in for ``_run_simple_foam_for_vector``.

    Each call appends an entry to ``records`` and returns a deterministic
    success manifest. Cd is derived from the first parameter so the test
    can later assert on a specific (vector, cd) mapping.

    The resulting callable is a closure — fine for the sequential path
    but it can't cross a process boundary. The parallel-workers path uses
    a top-level worker mock instead (see :func:`_top_level_run_one`).
    """
    def _impl(*, label, vector, **_kwargs):
        records.append({"label": label, "vector": dict(vector.values)})
        cd_value = float(vector.values[KRACHT_PARAMETER_NAMES[0]]) * 10.0 + 0.5
        return {
            "label": label,
            "vector": dict(vector.values),
            "cd": cd_value,
            "status": "succeeded",
            "reason": None,
        }
    return _impl


# ---------- test 1: resume skips completed samples -------------------------


def test_resume_skips_completed_samples(tmp_path, monkeypatch):
    """Pre-seeded history rows for LHS samples 0-4 must be skipped on rerun.

    With ``--n=10 --seed=0`` the script computes 10 LHS vectors. We
    pre-record samples 0-4 in the history store with the *same* seed so
    the script's resume check finds them, then assert only samples 5-9
    are passed through the run-sample callable.
    """
    # Faux env so the script's BULBOPT_OPENFOAM_BIN guard passes.
    monkeypatch.setenv("BULBOPT_OPENFOAM_BIN", str(tmp_path / "fake_of_bin"))

    history_path = tmp_path / "history.jsonl"
    project_root = tmp_path / "projects"
    project_root.mkdir()

    # Pre-populate history with the first 5 LHS-seed-0 samples on the
    # tightened bounds (the script's default --bounds value).
    space = KrachtDesignSpace.tightened()
    samples = latin_hypercube_sample(10, space, seed=0)
    pre_store = HistoryStore(path=history_path)
    for vec in samples[:5]:
        pre_store.record(vec, cd=0.99, backend="simple_foam")

    # Make the source STL discoverable; content does not matter because
    # _build_baseline is mocked out.
    fake_stl = tmp_path / "fake.stl"
    fake_stl.write_bytes(b"")

    records: List[Dict[str, Any]] = []
    fake_run = _fake_run_factory(records)

    with patch.object(cfd_doe_seed, "_build_baseline", _fake_baseline), \
         patch.object(cfd_doe_seed, "_run_simple_foam_for_vector", fake_run), \
         patch.object(cfd_doe_seed, "OpenFOAMAdapter") as mock_builder, \
         patch.object(cfd_doe_seed, "OpenFOAMRunnerAdapter") as mock_runner, \
         patch.object(cfd_doe_seed, "BulbFFDDeformer") as mock_deformer:
        mock_builder.return_value = object()
        mock_runner.return_value = object()
        mock_deformer.return_value = object()

        rc = cfd_doe_seed.main([
            "--n", "10",
            "--bounds", "tightened",
            "--case-name", "doe_resume",
            "--root", str(project_root),
            "--source-stl", str(fake_stl),
            "--history-path", str(history_path),
            "--seed", "0",
            "--skip-baseline",
            "--resume",
            "--parallel-workers", "1",
        ])

    assert rc == 0, "DOE seed script with --resume should exit 0"
    assert len(records) == 5, (
        f"Expected only samples 5-9 (5 calls) when 0-4 are already in "
        f"history; got {len(records)} calls: "
        f"{[r['label'] for r in records]}"
    )
    # Labels are 1-indexed in the script (sample_001 .. sample_010).
    skipped_labels = {f"sample_{i:03d}" for i in range(1, 6)}
    actual_labels = {r["label"] for r in records}
    assert actual_labels.isdisjoint(skipped_labels), (
        f"Skipped labels {skipped_labels} must not appear in run set "
        f"{actual_labels}"
    )
    expected_labels = {f"sample_{i:03d}" for i in range(6, 11)}
    assert actual_labels == expected_labels, (
        f"Run labels {actual_labels} != expected {expected_labels}"
    )


# ---------- test 2: parallel == sequential ---------------------------------


class _InProcessFuture:
    """Tiny ``Future`` look-alike that resolves immediately on creation."""

    def __init__(self, result_value, exc=None):
        self._result = result_value
        self._exc = exc

    def result(self):
        if self._exc is not None:
            raise self._exc
        return self._result


class _InProcessExecutor:
    """Drop-in for ``ProcessPoolExecutor`` that runs work inline.

    The script's parallel path imports ``ProcessPoolExecutor`` and
    ``as_completed`` at module scope. We patch both so the test
    exercises the chunking + aggregation code paths without paying for
    real subprocess spawn (or fighting Windows' fork-vs-spawn issues
    inside pytest).
    """

    def __init__(self, max_workers=None):
        self._max_workers = max_workers
        self._results: list[_InProcessFuture] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, *args, **kwargs):
        try:
            value = fn(*args, **kwargs)
            future = _InProcessFuture(value)
        except Exception as exc:  # pragma: no cover - defensive
            future = _InProcessFuture(None, exc=exc)
        self._results.append(future)
        return future


def _in_process_as_completed(futures):
    """Yield futures in submission order — close enough for the parallel test."""
    for future in list(futures):
        yield future


def test_parallel_workers_produce_same_set_as_sequential(tmp_path, monkeypatch):
    """With identical seeds, parallel-workers=2 must yield the same
    ``(parameters, cd)`` set as sequential mode (order may differ).

    We patch ``ProcessPoolExecutor`` and ``as_completed`` to in-process
    stubs so the test stays fast and avoids re-importing this module
    inside a child interpreter (which on Windows would force the
    pickle round-trip of the patched ``_run_simple_foam_for_vector``
    closure to fail).
    """
    monkeypatch.setenv("BULBOPT_OPENFOAM_BIN", str(tmp_path / "fake_of_bin"))

    fake_stl = tmp_path / "fake.stl"
    fake_stl.write_bytes(b"")

    project_root_a = tmp_path / "projects_seq"
    project_root_a.mkdir()
    project_root_b = tmp_path / "projects_par"
    project_root_b.mkdir()
    history_a = tmp_path / "history_seq.jsonl"
    history_b = tmp_path / "history_par.jsonl"

    def _run_with(parallel_workers: int, project_root: Path, history_path: Path) -> None:
        records: List[Dict[str, Any]] = []
        fake_run = _fake_run_factory(records)

        # The parallel path goes through _worker_run_chunk which
        # rebuilds adapters and reads a real STL. We bypass that by
        # patching the chunk runner with a thin in-process stand-in
        # that just calls _run_simple_foam_for_vector directly.
        def _fake_worker(payload):
            from bulbopt.optimization.parametric.kracht_space import KrachtVector
            out = []
            for item in payload["items"]:
                vec = KrachtVector(values={k: float(v) for k, v in item["vector"].items()})
                manifest = fake_run(
                    label=item["label"],
                    vector=vec,
                    baseline_mesh=None,
                    region=payload.get("region"),
                    deformer=None,
                    builder=None,
                    runner=None,
                    work_root=Path(payload["work_root"]),
                    timeout_seconds=int(payload["timeout"]),
                )
                out.append(manifest)
            return out

        with patch.object(cfd_doe_seed, "_build_baseline", _fake_baseline), \
             patch.object(cfd_doe_seed, "_run_simple_foam_for_vector", fake_run), \
             patch.object(cfd_doe_seed, "_worker_run_chunk", _fake_worker), \
             patch.object(cfd_doe_seed, "ProcessPoolExecutor", _InProcessExecutor), \
             patch.object(cfd_doe_seed, "as_completed", _in_process_as_completed), \
             patch.object(cfd_doe_seed, "OpenFOAMAdapter") as mock_builder, \
             patch.object(cfd_doe_seed, "OpenFOAMRunnerAdapter") as mock_runner, \
             patch.object(cfd_doe_seed, "BulbFFDDeformer") as mock_deformer:
            mock_builder.return_value = object()
            mock_runner.return_value = object()
            mock_deformer.return_value = object()

            rc = cfd_doe_seed.main([
                "--n", "4",
                "--bounds", "tightened",
                "--case-name", "doe_parallel",
                "--root", str(project_root),
                "--source-stl", str(fake_stl),
                "--history-path", str(history_path),
                "--seed", "11",
                "--skip-baseline",
                "--parallel-workers", str(parallel_workers),
            ])
            assert rc == 0

    _run_with(1, project_root_a, history_a)
    _run_with(2, project_root_b, history_b)

    seq_store = HistoryStore(path=history_a)
    par_store = HistoryStore(path=history_b)
    seq_rows = seq_store.load_all(backend="simple_foam")
    par_rows = par_store.load_all(backend="simple_foam")

    def _row_set(rows):
        # ``KrachtVector.values`` is a Mapping (frozen); freeze its
        # contents into a hashable tuple for set comparison.
        keyed = []
        for vec, cd in rows:
            params = tuple(round(float(vec.values[k]), 12) for k in KRACHT_PARAMETER_NAMES)
            keyed.append((params, round(float(cd), 12)))
        return set(keyed)

    assert len(seq_rows) == 4
    assert len(par_rows) == 4
    assert _row_set(seq_rows) == _row_set(par_rows), (
        "Parallel run should produce the same (vector, cd) set as "
        "sequential — only order may differ."
    )
