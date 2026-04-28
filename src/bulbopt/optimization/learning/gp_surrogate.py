"""Gaussian-Process surrogate for Cd prediction in the 8-D Kracht space.

Design reference: 2026-04-23-bulbopt-mesh-quality-design.md §4 (L1).

Wraps ``sklearn.gaussian_process.GaussianProcessRegressor`` with an
RBF + white-noise kernel. Inputs are 8-D KrachtVectors, the output is the
scalar Cd (or any high-fidelity drag metric). With fewer than 5 training
points the GP isn't meaningful, so ``predict`` returns ``None`` as a
signal to the caller to fall back to the analytic proxy.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from bulbopt.optimization.parametric.kracht_space import (
    KRACHT_PARAMETER_NAMES,
    KrachtVector,
)


MIN_TRAIN_POINTS = 5


def _vectors_to_matrix(vectors: Sequence[KrachtVector]) -> np.ndarray:
    return np.asarray(
        [[float(v.values[name]) for name in KRACHT_PARAMETER_NAMES] for v in vectors],
        dtype=float,
    )


class GPSurrogate:
    """Thin GP wrapper.

    The model is intentionally lazy — ``fit`` replaces any existing model
    so you can refit on a growing history.
    """

    def __init__(self) -> None:
        self._model = None
        self._n_train = 0

    def fit(self, vectors: Sequence[KrachtVector], cds: Sequence[float]) -> None:
        """Fit the GP on the given (vector, cd) pairs.

        If ``len(vectors) < MIN_TRAIN_POINTS`` the surrogate is kept
        "untrained" (``predict`` will return ``None``). Sklearn is a
        relatively heavy import so we defer it to first call.
        """
        self._n_train = len(vectors)
        if self._n_train < MIN_TRAIN_POINTS:
            self._model = None
            return

        # sklearn imports — deferred so tests that don't need the GP don't
        # pay the import cost.
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel

        X = _vectors_to_matrix(vectors)
        y = np.asarray(list(cds), dtype=float)

        # Length-scale bounds are deliberately loose so the optimizer can
        # adapt to whatever range the Kracht parameters were sampled from.
        kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)) + WhiteKernel(
            noise_level=1e-3, noise_level_bounds=(1e-6, 1e0)
        )
        self._model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=0,
        )
        self._model.fit(X, y)

    def predict(
        self, vectors: Sequence[KrachtVector]
    ) -> Optional[Tuple[List[float], List[float]]]:
        """Return (means, stds) per query vector, or None if untrained."""
        if self._model is None:
            return None
        X = _vectors_to_matrix(vectors)
        means, stds = self._model.predict(X, return_std=True)
        return ([float(m) for m in means], [float(s) for s in stds])
