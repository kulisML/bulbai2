"""Integration tests: validity classifier prefilter wired into night-run.

Design reference: 2026-04-23 mesh-quality design §4 L4.

The prefilter is wired into :func:`run_night_optimization` as a wrapper
around the mid-gate evaluator. When the classifier predicts
``p(invalid) > reject_threshold`` the wrapper must:

1. Skip the mesh generation (deformer call) entirely.
2. Return a hard penalty objective row so NSGA-II demotes the candidate.
3. Record the predicted-invalid label in the history file anyway so the
   next run's training set grows.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest
import trimesh

from bulbopt.application.use_cases.run_night_optimization import (
    VALIDITY_HISTORY_FILENAME,
    _load_and_train_validity_classifier,
    _validity_history_path,
    _with_validity_prefilter,
)
from bulbopt.execution.logging.case_logger import CaseLogger
from bulbopt.optimization.learning.validity_classifier import ValidityClassifier
from bulbopt.optimization.parametric.ffd_deformer import BulbFFDDeformer
from bulbopt.optimization.parametric.kracht_space import (
    KRACHT_PARAMETER_NAMES,
    KrachtDesignSpace,
    KrachtVector,
)


class _StubClassifier(ValidityClassifier):
    """Deterministic classifier: the first N training samples is reported
    so the wrapper thinks the model is trained, and every prediction
    returns a hard-coded probability."""

    def __init__(self, *, probability: float, training_samples: int) -> None:
        super().__init__()
        self._probability = float(probability)
        self._n_training_samples = int(training_samples)
        self._trained_on_single_class = True
        self._model = float(probability)

    def predict_invalid_probability(self, vector):  # type: ignore[override]
        return float(self._probability)


def _seed_log(tmp_path: Path) -> CaseLogger:
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return CaseLogger(log_dir / "case.log")


def _baseline_region(mesh: trimesh.Trimesh) -> dict:
    extents = mesh.extents
    primary_axis = int(extents.argmax())
    axis_min = float(mesh.vertices[:, primary_axis].mean())
    axis_max = float(mesh.vertices[:, primary_axis].max())
    return {
        "axis_index": primary_axis,
        "axis_min": axis_min,
        "axis_max": axis_max,
    }


def test_prefilter_skips_deformer_and_returns_penalty(tmp_path: Path):
    mesh = trimesh.creation.box(extents=(4.0, 1.5, 1.0))
    region = _baseline_region(mesh)
    deformer = BulbFFDDeformer(post_smoothing_iterations=0)
    history_path = tmp_path / "validity_history.jsonl"
    case_logger = _seed_log(tmp_path)

    deformer_calls: List[KrachtVector] = []

    # 3 objectives: resistance_proxy, volume_delta, mesh_quality (L2).
    def _base_evaluator(vectors):
        for v in vectors:
            deformer_calls.append(v)
        return [[1.0, 0.1, 2.0] for _ in vectors]

    classifier = _StubClassifier(probability=0.95, training_samples=50)
    wrapped = _with_validity_prefilter(
        _base_evaluator,
        classifier=classifier,
        reject_threshold=0.7,
        history_path=history_path,
        baseline_mesh=mesh,
        region=region,
        deformer=deformer,
        case_logger=case_logger,
    )
    space = KrachtDesignSpace()
    samples = space.sample(5, seed=0)
    rows = wrapped(samples)

    # Every row is the penalty because the stub classifier votes reject.
    # Penalty width defaults to 3 (the real mid-gate's objective count
    # after L2 added mesh_quality). Spec 2026-04-23 §4 L4.
    assert len(rows) == 5
    for row in rows:
        assert row == pytest.approx([1e9, 1e9, 1e9])
    # Base evaluator was never called → deformer never ran on the mid gate.
    assert deformer_calls == []

    # History file contains 5 rows, all marked invalid.
    entries = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(entries) == 5
    assert all(entry["invalid"] == 1 for entry in entries)


def test_prefilter_falls_back_to_base_when_classifier_cold(tmp_path: Path):
    mesh = trimesh.creation.box(extents=(4.0, 1.5, 1.0))
    region = _baseline_region(mesh)
    deformer = BulbFFDDeformer(post_smoothing_iterations=0)
    history_path = tmp_path / "validity_history.jsonl"
    case_logger = _seed_log(tmp_path)

    invocations = {"count": 0}

    # 3 objectives: resistance, volume, mesh_quality (L2).
    def _base_evaluator(vectors):
        invocations["count"] += 1
        return [[1.23, 0.05, 0.8] for _ in vectors]

    # No fit() → classifier stays cold; predict_invalid_probability returns None.
    classifier = ValidityClassifier()
    wrapped = _with_validity_prefilter(
        _base_evaluator,
        classifier=classifier,
        reject_threshold=0.7,
        history_path=history_path,
        baseline_mesh=mesh,
        region=region,
        deformer=deformer,
        case_logger=case_logger,
    )
    space = KrachtDesignSpace()
    samples = space.sample(3, seed=0)
    rows = wrapped(samples)

    assert rows == [[1.23, 0.05, 0.8]] * 3
    assert invocations["count"] == 1
    # History records actual sanity-check labels for each sample.
    entries = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(entries) == 3
    for entry in entries:
        # Each entry has 8 Kracht values + 0/1 invalid flag.
        assert len(entry["vector"]) == len(KRACHT_PARAMETER_NAMES)
        assert entry["invalid"] in (0, 1)


def test_load_and_train_validity_classifier_uses_history(tmp_path: Path):
    history_path = tmp_path / "validity_history.jsonl"
    space = KrachtDesignSpace()
    samples = space.sample(25, seed=0)
    # Label samples: invalid when length_ratio exceeds the midpoint.
    threshold = 0.5 * sum(space.bounds["length_ratio"])
    with history_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            label = int(sample.values["length_ratio"] > threshold)
            row = {
                "vector": [float(sample.values[n]) for n in KRACHT_PARAMETER_NAMES],
                "invalid": label,
            }
            handle.write(json.dumps(row) + "\n")

    classifier = _load_and_train_validity_classifier(history_path)
    assert classifier.n_training_samples == 25
    probas = [
        classifier.predict_invalid_probability(sample) for sample in samples
    ]
    assert all(p is not None for p in probas)


def test_validity_history_path_is_stable(tmp_path: Path):
    """History path is reproducible; directory is created lazily on write."""
    project_root = tmp_path / "projects"
    path = _validity_history_path(project_root)
    assert path.name == VALIDITY_HISTORY_FILENAME
    # Parent directory must NOT exist yet — we avoid littering empty
    # project roots with a hidden folder on dry-run flows.
    assert not path.parent.exists()


def test_prefilter_does_not_reject_all_when_classifier_only_saw_invalid_history(
    tmp_path: Path,
):
    """Bug #4 (audit 2026-04-26): when 15 all-invalid history rows are
    pre-loaded, the freshly trained classifier must NOT freeze every
    prediction at 1.0 and force every fresh candidate into the penalty
    path. At least some of the 5 candidates must reach the base evaluator
    so NSGA-II keeps a real fitness landscape to climb.
    """
    history_path = tmp_path / "validity_history.jsonl"
    space = KrachtDesignSpace()
    seed_samples = space.sample(15, seed=42)
    with history_path.open("w", encoding="utf-8") as handle:
        for sample in seed_samples:
            row = {
                "vector": [
                    float(sample.values[name]) for name in KRACHT_PARAMETER_NAMES
                ],
                "invalid": 1,
            }
            handle.write(json.dumps(row) + "\n")

    classifier = _load_and_train_validity_classifier(history_path)
    # The classifier saw 15 all-invalid rows — Bug #4's freeze condition.

    mesh = trimesh.creation.box(extents=(4.0, 1.5, 1.0))
    region = _baseline_region(mesh)
    deformer = BulbFFDDeformer(post_smoothing_iterations=0)
    case_logger = _seed_log(tmp_path)

    base_calls: List[KrachtVector] = []

    def _base_evaluator(vectors):
        for v in vectors:
            base_calls.append(v)
        return [[0.5, 0.05, 1.2] for _ in vectors]

    wrapped = _with_validity_prefilter(
        _base_evaluator,
        classifier=classifier,
        reject_threshold=0.7,
        history_path=history_path,
        baseline_mesh=mesh,
        region=region,
        deformer=deformer,
        case_logger=case_logger,
    )

    fresh_samples = space.sample(5, seed=99)
    rows = wrapped(fresh_samples)

    assert len(rows) == 5
    # The freeze bug (Bug #4) would have produced 5 all-penalty rows AND
    # zero base-evaluator calls. After the fix, AT LEAST ONE candidate
    # must reach the base evaluator. We use ``len(base_calls) > 0`` as
    # the empirical proof the freeze is gone.
    assert len(base_calls) > 0, (
        "all 5 candidates were rejected — the all-invalid history "
        "froze the prefilter, NSGA-II would see a flat landscape"
    )

    # Of the 5 returned rows, at least one must NOT be a penalty row.
    non_penalty_rows = [r for r in rows if r != pytest.approx([1e9, 1e9, 1e9])]
    assert len(non_penalty_rows) > 0, (
        "all 5 rows were penalties — prefilter still freezes on "
        "all-invalid history"
    )
