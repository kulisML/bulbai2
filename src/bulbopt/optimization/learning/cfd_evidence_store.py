"""Append-only evidence store for high-fidelity CFD evaluations."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

from bulbopt.optimization.parametric.kracht_space import (
    KRACHT_PARAMETER_NAMES,
    KrachtVector,
)


@dataclass(slots=True)
class CFDEvidenceStore:
    """Persist machine-readable CFD evidence as JSONL.

    The store intentionally stays filesystem-native so engineers can inspect,
    diff, archive, or feed the rows into ML tooling without a database.
    """

    path: Path

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def append_many(self, rows: Iterable[dict]) -> None:
        rows = list(rows)
        if not rows:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    def write_all(self, rows: Iterable[dict]) -> None:
        rows = list(rows)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    def load_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        loaded: list[dict] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                loaded.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return loaded

    def top_k_safe_warm_start(
        self,
        n: int,
        *,
        hull_fingerprint: str | None = None,
        settings_hash: str | None = None,
    ) -> list[KrachtVector]:
        """Return verified improving candidates ordered by lowest final Cd."""
        rows = []
        for row in self.load_all():
            if self._warm_start_rejection_reason(
                row,
                hull_fingerprint=hull_fingerprint,
                settings_hash=settings_hash,
            ):
                continue
            vector = self._vector_from_row(row)
            if vector is None:
                continue
            rows.append((vector, float(row["final_cd"])))

        rows.sort(key=lambda pair: pair[1])
        return [vector for vector, _cd in rows[: int(max(n, 0))]]

    def surrogate_training_pairs(
        self,
        *,
        hull_fingerprint: str | None = None,
        settings_hash: str | None = None,
    ) -> list[Tuple[KrachtVector, float]]:
        """Return finite, engineering-valid CFD rows for surrogate training."""
        rows: list[Tuple[KrachtVector, float]] = []
        for row in self.load_all():
            if self._training_rejection_reason(
                row,
                hull_fingerprint=hull_fingerprint,
                settings_hash=settings_hash,
            ):
                continue
            try:
                cd = float(row["final_cd"])
            except (TypeError, ValueError):
                continue
            vector = self._vector_from_row(row)
            if vector is None:
                continue
            rows.append((vector, cd))
        return rows

    def warm_start_eligibility_summary(
        self,
        *,
        hull_fingerprint: str | None = None,
        settings_hash: str | None = None,
    ) -> dict:
        summary = {
            "candidate_rows": 0,
            "eligible": 0,
            "hull_mismatch": 0,
            "settings_mismatch": 0,
            "not_engineering_valid": 0,
            "geometry_high_risk": 0,
            "manufacturability_warning": 0,
            "not_improving": 0,
            "missing_final_cd": 0,
            "missing_parameters": 0,
        }
        for row in self.load_all():
            if row.get("record_type") != "candidate":
                continue
            summary["candidate_rows"] += 1
            reason = self._warm_start_rejection_reason(
                row,
                hull_fingerprint=hull_fingerprint,
                settings_hash=settings_hash,
            )
            if reason is None:
                summary["eligible"] += 1
            elif reason in summary:
                summary[reason] += 1
        return summary

    def best_candidate_rows(
        self,
        n: int,
        *,
        hull_fingerprint: str | None = None,
        settings_hash: str | None = None,
    ) -> list[dict]:
        """Return compatible, improving candidate evidence rows by lowest Cd."""
        rows: list[dict] = []
        for row in self.load_all():
            if self._warm_start_rejection_reason(
                row,
                hull_fingerprint=hull_fingerprint,
                settings_hash=settings_hash,
            ):
                continue
            rows.append(dict(row))
        rows.sort(key=lambda row: float(row["final_cd"]))
        return rows[: int(max(n, 0))]

    def _warm_start_rejection_reason(
        self,
        row: dict,
        *,
        hull_fingerprint: str | None,
        settings_hash: str | None,
    ) -> str | None:
        reason = self._base_candidate_rejection_reason(
            row,
            hull_fingerprint=hull_fingerprint,
            settings_hash=settings_hash,
        )
        if reason is not None:
            return reason
        improvement = row.get("improvement_percent")
        try:
            if improvement is None or float(improvement) <= 0.0:
                return "not_improving"
        except (TypeError, ValueError):
            return "not_improving"
        return None

    def _training_rejection_reason(
        self,
        row: dict,
        *,
        hull_fingerprint: str | None,
        settings_hash: str | None,
    ) -> str | None:
        return self._base_candidate_rejection_reason(
            row,
            hull_fingerprint=hull_fingerprint,
            settings_hash=settings_hash,
        )

    def _base_candidate_rejection_reason(
        self,
        row: dict,
        *,
        hull_fingerprint: str | None,
        settings_hash: str | None,
    ) -> str | None:
        if row.get("record_type") != "candidate":
            return "not_candidate"
        if hull_fingerprint and row.get("hull_fingerprint") != hull_fingerprint:
            return "hull_mismatch"
        if settings_hash and row.get("settings_hash") != settings_hash:
            return "settings_mismatch"
        if row.get("engineering_valid") is not True:
            return "not_engineering_valid"
        final_cd = row.get("final_cd")
        if final_cd is None:
            return "missing_final_cd"
        try:
            if float(final_cd) >= 1e8:
                return "missing_final_cd"
        except (TypeError, ValueError):
            return "missing_final_cd"
        if self._vector_from_row(row) is None:
            return "missing_parameters"
        geometry_reason = self._geometry_rejection_reason(row)
        if geometry_reason is not None:
            return geometry_reason
        return None

    def _geometry_rejection_reason(self, row: dict) -> str | None:
        geometry = row.get("geometry")
        if not isinstance(geometry, dict):
            return None

        if geometry.get("constraint_violations"):
            return "geometry_high_risk"
        stl_report = geometry.get("stl_report")
        if isinstance(stl_report, dict) and stl_report.get("checks_passed") is False:
            return "geometry_high_risk"
        if geometry.get("geometry_risk") == "high":
            return "geometry_high_risk"
        if geometry.get("manufacturability_risk") == "warning":
            return "manufacturability_warning"
        if geometry.get("parameter_warnings"):
            return "manufacturability_warning"
        return None

    def _vector_from_row(self, row: dict) -> KrachtVector | None:
        parameters = row.get("parameters") or {}
        if not all(name in parameters for name in KRACHT_PARAMETER_NAMES):
            return None
        try:
            return KrachtVector(
                values={
                    name: float(parameters[name])
                    for name in KRACHT_PARAMETER_NAMES
                }
            )
        except (TypeError, ValueError):
            return None
