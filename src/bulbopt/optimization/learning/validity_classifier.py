"""Logistic regression classifier that predicts mesh-invalid probability.

Design reference: ``docs/superpowers/specs/2026-04-23-bulbopt-mesh-quality-design.md`` §4 L4.

The classifier learns on the history of ``(KrachtVector, invalid_flag)`` pairs
accumulated across past night-runs (each HF evaluation records whether the
resulting mesh was watertight / non-self-intersecting / had positive volume).

The predictor is used as a cheap prefilter by the mid-gate: if
``p(invalid) > 0.7`` we skip the FFD + mesh work and return a hard penalty
so NSGA-II naturally learns to avoid that region of the Kracht box.

Two explicit design choices:

1. **No training below 10 samples.** Logistic regression on a tiny dataset
   is noisier than just trusting the HF adapter to catch invalid meshes
   downstream, and forbidding early predictions avoids a cold-start where
   random labels freeze the optimisation in one corner of the space.

2. **Always refit on ``fit``.** The classifier is small (8 features), so
   there's no reason to carry state between runs — we always rebuild from
   the full history. This makes the public API trivial to test.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from bulbopt.optimization.parametric.kracht_space import (
    KRACHT_PARAMETER_NAMES,
    KrachtVector,
)


# Minimum training samples before the classifier will emit a prediction.
# Below this threshold ``predict_invalid_probability`` returns ``None`` so
# callers know to fall back to the usual evaluation path.
MIN_TRAINING_SAMPLES: int = 10


@dataclass
class ValidityClassifier:
    """Wrap ``sklearn.linear_model.LogisticRegression`` with a KrachtVector API.

    ``fit`` accepts either a list of :class:`KrachtVector` (the primary form)
    or a 2-D numpy array already in declared KRACHT_PARAMETER_NAMES order.
    Labels are ``1`` for *invalid* meshes, ``0`` for *valid* meshes — the
    classifier's positive class is the thing the prefilter wants to avoid.
    """

    _model: object | None = None  # sklearn LogisticRegression | None
    _n_training_samples: int = 0
    _trained_on_single_class: bool = False

    def __init__(self) -> None:
        self._model = None
        self._n_training_samples = 0
        self._trained_on_single_class = False

    # ------------------------------------------------------------------ fit

    def fit(
        self,
        kracht_arrays: Sequence[KrachtVector] | Sequence[Sequence[float]] | np.ndarray,
        labels: Sequence[int] | Sequence[bool] | np.ndarray,
    ) -> "ValidityClassifier":
        """Fit the logistic regression on the supplied history.

        When fewer than :data:`MIN_TRAINING_SAMPLES` rows are supplied the
        classifier stays in "cold" state — :meth:`predict_invalid_probability`
        will return ``None`` for every input.

        When every supplied label has the same class (e.g. every past
        sample was valid or every past sample was invalid), sklearn
        refuses to fit logistic regression. Bug #4 (audit 2026-04-26)
        previously made this case freeze the predictor at the seen
        class's label, so an all-invalid bootstrap (e.g. a bad baseline
        mesh on the first night-run) would lock every prediction at
        ``1.0`` and reject every future candidate forever — flattening
        NSGA-II's fitness landscape.

        Fix: record the single-class shortcut and use a SAFE PROBABILITY
        in :meth:`predict_invalid_probability`:

        * all-valid (only class ``0`` seen) → return ``0.0`` (no
          rejection — the prefilter is a no-op until both classes appear).
        * all-invalid (only class ``1`` seen) → return ``None`` so the
          prefilter wrapper degrades gracefully and lets candidates
          through; subsequent runs add fresh labels and the freeze
          self-corrects as soon as a single valid candidate is seen.
        """
        features = _to_feature_matrix(kracht_arrays)
        y = np.asarray(list(labels), dtype=int).ravel()
        if features.shape[0] != y.shape[0]:
            raise ValueError(
                "kracht_arrays and labels must have the same length; "
                f"got {features.shape[0]} and {y.shape[0]}."
            )

        self._n_training_samples = int(features.shape[0])

        if self._n_training_samples < MIN_TRAINING_SAMPLES:
            # Stay cold — caller will see None and skip the prefilter.
            self._model = None
            self._trained_on_single_class = False
            return self

        unique_labels = np.unique(y)
        if unique_labels.size < 2:
            # sklearn.LogisticRegression requires at least two classes.
            # Record the single-class shortcut so predict_invalid_probability
            # can still answer with a SAFE probability (Bug #4 fix).
            self._trained_on_single_class = True
            self._model = float(unique_labels[0])
            return self

        # Deferred import so the module stays importable even when sklearn
        # isn't installed (matters for the unit-test discovery phase).
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=1000)
        model.fit(features, y)
        self._trained_on_single_class = False
        self._model = model
        return self

    # -------------------------------------------------------------- predict

    def predict_invalid_probability(
        self,
        vector: KrachtVector | Sequence[float] | np.ndarray,
    ) -> float | None:
        """Return ``p(invalid)`` for ``vector``.

        Returns ``None`` when the classifier has not yet been fit, or when
        it was fit on fewer than :data:`MIN_TRAINING_SAMPLES` rows. Callers
        that get ``None`` should fall back to running the full evaluation
        (the history is simply too sparse to trust the predictor).

        Bug #4 (audit 2026-04-26): when fit on a single class, the safe
        replies are:

        * only class ``0`` seen → ``0.0`` (no rejection).
        * only class ``1`` seen → ``None`` so the prefilter degrades to
          a no-op until at least one valid sample is observed.
        """
        if self._n_training_samples < MIN_TRAINING_SAMPLES:
            return None
        if self._model is None:
            return None

        row = _vector_to_row(vector)
        if self._trained_on_single_class:
            # Single-class shortcut. ``self._model`` is the float of the
            # only seen class — 0.0 or 1.0.
            seen_label = float(self._model)
            if seen_label <= 0.0:
                # Only valid samples seen → don't reject anything.
                return 0.0
            # Only invalid samples seen → refuse to predict so the wrapper
            # falls back to running the base evaluator. Returning 1.0
            # would freeze NSGA-II in a flat penalty landscape (Bug #4).
            return None

        # sklearn.LogisticRegression.classes_ is sorted ascending, so
        # invalid (=1) always ends up at index 1 when both classes exist.
        proba = self._model.predict_proba(row.reshape(1, -1))[0]
        classes = np.asarray(self._model.classes_)
        invalid_index = int(np.where(classes == 1)[0][0])
        return float(proba[invalid_index])

    # ------------------------------------------------------------- helpers

    @property
    def n_training_samples(self) -> int:
        return int(self._n_training_samples)


# ----------------------------------------------------------------- internals


def _to_feature_matrix(
    kracht_arrays: Sequence[KrachtVector] | Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Normalise the ``fit`` input into an ``(N, 8)`` float matrix."""
    if isinstance(kracht_arrays, np.ndarray):
        return np.asarray(kracht_arrays, dtype=float).reshape(-1, len(KRACHT_PARAMETER_NAMES))

    rows: List[List[float]] = []
    for entry in kracht_arrays:
        rows.append(_vector_to_row(entry).tolist())
    if not rows:
        return np.zeros((0, len(KRACHT_PARAMETER_NAMES)), dtype=float)
    return np.asarray(rows, dtype=float)


def _vector_to_row(
    vector: KrachtVector | Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return a single ``(8,)`` float row in KRACHT_PARAMETER_NAMES order."""
    if isinstance(vector, KrachtVector):
        return np.asarray(
            [float(vector.values[name]) for name in KRACHT_PARAMETER_NAMES],
            dtype=float,
        )
    arr = np.asarray(list(vector) if not isinstance(vector, np.ndarray) else vector, dtype=float)
    if arr.shape != (len(KRACHT_PARAMETER_NAMES),):
        raise ValueError(
            f"Expected a {len(KRACHT_PARAMETER_NAMES)}-element vector; got shape {arr.shape}."
        )
    return arr


__all__ = ["ValidityClassifier", "MIN_TRAINING_SAMPLES"]
