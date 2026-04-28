"""Append-only JSONL history store for (KrachtVector, Cd) pairs.

Design reference: 2026-04-23-bulbopt-mesh-quality-design.md §4 (L1).

Every completed high-fidelity evaluation writes one row
``{"parameters": {...}, "cd": float}`` to
``~/.bulbopt/history.jsonl`` (overridable). Subsequent night-runs load
this history to:

* Warm-start the NSGA-II initial population with the lowest-Cd points.
* Train a Gaussian Process that can replace the analytic mid-gate
  evaluator once enough history is accumulated.

The store deliberately uses plain JSONL so it's editable / inspectable /
trimmable with standard tools; no database needed.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from bulbopt.optimization.parametric.kracht_space import (
    KRACHT_PARAMETER_NAMES,
    KrachtVector,
)


def default_history_path() -> Path:
    """Default path: ``~/.bulbopt/history.jsonl``.

    Uses ``Path.home()`` which respects ``HOME`` / ``USERPROFILE`` on
    both POSIX and Windows.
    """
    return Path.home() / ".bulbopt" / "history.jsonl"


@dataclass(slots=True)
class HistoryStore:
    """Append-only history of (vector, Cd) pairs.

    Parameters
    ----------
    path:
        File path for the JSONL store. Defaults to
        ``~/.bulbopt/history.jsonl`` — callers (e.g. ProjectRepository)
        may override with a test path or a per-project location.
    """

    path: Path

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_history_path()

    # ---- writing ---------------------------------------------------------

    def record(
        self,
        vector: KrachtVector,
        cd: float,
        *,
        backend: str | None = None,
    ) -> None:
        """Append one row to the JSONL store. Creates parents lazily.

        ``backend`` tags which Cd computation produced this row. Used to
        keep the GP surrogate from training on mixed-scale Cd (Bug #5,
        audit 2026-04-26): real OpenFOAM Cd lives in ~0.3-1.5 while the
        analytic proxy lives in ~0.01-0.5. Common values:

        * ``"simple_foam"`` — real OpenFOAM ``simpleFoam`` Cd.
        * ``"surrogate"`` / ``"proxy"`` — geometric ``(beam*draft)/axial``
          fallback when OpenFOAM is unavailable.
        * ``"external"`` — caller-supplied evaluator.
        * ``None`` — unknown / legacy. Persisted as ``"unknown"`` so
          old-shape JSONL rows reload without losing information.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "parameters": {
                name: float(vector.values[name]) for name in KRACHT_PARAMETER_NAMES
            },
            "cd": float(cd),
            "backend": str(backend) if backend is not None else "unknown",
        }
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row) + "\n")

    # ---- reading ---------------------------------------------------------

    def load_all(
        self,
        *,
        backend: str | None = None,
    ) -> List[Tuple[KrachtVector, float]]:
        """Return the history as ``[(KrachtVector, cd), ...]``.

        Missing file returns an empty list. Malformed rows are silently
        skipped so a corrupt edit doesn't brick the optimizer.

        ``backend``: when supplied, only rows tagged with that backend
        are returned. Legacy rows without a ``backend`` field are tagged
        ``"unknown"`` on read; pass ``backend="unknown"`` to retrieve
        them, or omit ``backend`` to ignore the filter entirely.
        """
        if not self.path.exists():
            return []
        rows: List[Tuple[KrachtVector, float]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                parameters = payload["parameters"]
                cd = float(payload["cd"])
            except (KeyError, ValueError, json.JSONDecodeError):
                continue
            if not all(name in parameters for name in KRACHT_PARAMETER_NAMES):
                continue
            row_backend = payload.get("backend")
            row_backend = (
                str(row_backend) if row_backend is not None else "unknown"
            )
            if backend is not None and row_backend != backend:
                continue
            vec = KrachtVector(
                values={
                    name: float(parameters[name]) for name in KRACHT_PARAMETER_NAMES
                }
            )
            rows.append((vec, cd))
        return rows

    def top_k(self, n: int, *, backend: str | None = None) -> List[KrachtVector]:
        """Return the ``n`` KrachtVectors with the lowest Cd.

        If history has fewer than ``n`` rows, returns all of them.
        Order is ascending Cd. Pass ``backend`` to filter the source set
        (e.g. ``backend="simple_foam"`` to warm-start only from real CFD).
        """
        rows = self.load_all(backend=backend)
        rows.sort(key=lambda pair: pair[1])
        return [vec for vec, _cd in rows[: int(max(n, 0))]]
