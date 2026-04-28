"""Static reference-Cd database for sanity-checking surrogate predictions.

Why this exists
---------------

Multi-Fr CFD (see ``simple_foam_gate.SimpleFoamHighFidelityGate.evaluate``)
returns weighted-aggregate Cd values. Surrogates (``gp_surrogate``)
extrapolate from those. When the GP predicts something obviously wrong —
e.g. Cd=0.5 for a Wigley parabolic hull at Fr=0.20, where published
results put it near 0.0035 — the operator wants to know the prediction
is two orders of magnitude off scale.

This module provides a hand-maintained list of canonical
(hull, Fr, Cd_published) triples plus a small lookup helper. The lookup
intentionally returns ``None`` when no canonical Fr is within tolerance,
so callers can choose to skip the cross-check rather than mis-anchor it.

Important caveat
----------------

The ``cd_published`` values in this database are the **frictional /
pressure drag coefficients** reported in the cited literature at canonical
Reynolds numbers. They do **not** directly correspond to BulbOpt's Cd
metric (which is computed by OpenFOAM ``forceCoeffs`` against the
project's chosen ``magUInf`` / ``Aref`` / ``rhoInf`` triple). Treat the
database as an order-of-magnitude sanity check, not as ground truth.

Sources
-------

* Larsson, L. and Raven, H.C. (2010), *Ship Resistance and Flow*,
  Principles of Naval Architecture series, SNAME — Wigley parabolic
  hull benchmark data.
* ITTC quality manuals / KRISO benchmark — Series 60 Cb=0.60 reference
  resistance curves.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ReferenceCdEntry:
    """One published (hull, Fr, Cd) data point."""

    hull_id: str
    description: str
    froude_number: float
    cd_published: float
    source: str  # ITTC, Larsson-Raven, etc.


REFERENCE_CD_DATABASE: list[ReferenceCdEntry] = [
    # Wigley parabolic hull, Larsson 2010 (Ship Resistance and Flow).
    ReferenceCdEntry(
        "wigley_l100", "Wigley L=100m parabolic", 0.20, 0.0035, "Larsson-Raven 2010"
    ),
    ReferenceCdEntry(
        "wigley_l100", "Wigley L=100m parabolic", 0.30, 0.0048, "Larsson-Raven 2010"
    ),
    ReferenceCdEntry(
        "wigley_l100", "Wigley L=100m parabolic", 0.35, 0.0062, "Larsson-Raven 2010"
    ),
    # Series 60 Cb=0.60 — KRISO / ITTC benchmark.
    ReferenceCdEntry(
        "series60_cb60", "Series 60 Cb=0.60", 0.18, 0.0034, "ITTC benchmark"
    ),
    ReferenceCdEntry(
        "series60_cb60", "Series 60 Cb=0.60", 0.25, 0.0041, "ITTC benchmark"
    ),
    ReferenceCdEntry(
        "series60_cb60", "Series 60 Cb=0.60", 0.30, 0.0049, "ITTC benchmark"
    ),
]


def lookup_reference_cd(
    hull_id: str, fr: float, tolerance: float = 0.02
) -> ReferenceCdEntry | None:
    """Return the closest reference entry within Fr-tolerance, else None.

    Parameters
    ----------
    hull_id:
        Canonical hull identifier (must match an entry in
        ``REFERENCE_CD_DATABASE``).
    fr:
        Target Froude number.
    tolerance:
        Maximum |Fr - entry.froude_number| accepted as a match. Default
        0.02 — half a typical Fr-grid spacing.

    Returns
    -------
    The closest matching entry (smallest |Fr| difference) or ``None`` if
    no entry for ``hull_id`` lies within the tolerance window.
    """
    best: ReferenceCdEntry | None = None
    best_distance = float("inf")
    for entry in REFERENCE_CD_DATABASE:
        if entry.hull_id != hull_id:
            continue
        distance = abs(entry.froude_number - fr)
        if distance > tolerance:
            continue
        if distance < best_distance:
            best = entry
            best_distance = distance
    return best
