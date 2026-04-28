"""Tests for the static reference-Cd database.

The database stores published Cd values for canonical hulls (Wigley,
Series 60, ...) at canonical Froude numbers. ``lookup_reference_cd``
returns the closest entry whose Fr is within ``tolerance`` of the
requested Fr — used to sanity-check GP predictions for orders-of-magnitude
errors.
"""
from __future__ import annotations

from bulbopt.optimization.learning.reference_cd_database import (
    REFERENCE_CD_DATABASE,
    ReferenceCdEntry,
    lookup_reference_cd,
)


def test_reference_cd_database_contains_wigley_and_series60() -> None:
    """Sanity-check the database is populated with the canonical hulls."""
    hull_ids = {entry.hull_id for entry in REFERENCE_CD_DATABASE}
    assert "wigley_l100" in hull_ids
    assert "series60_cb60" in hull_ids


def test_reference_cd_lookup_finds_within_tolerance() -> None:
    """A request at Fr=0.21 with tolerance 0.02 must snap to the
    Fr=0.20 Wigley entry; a request at Fr=0.50 must miss every entry."""
    near = lookup_reference_cd("wigley_l100", 0.21, tolerance=0.02)
    assert near is not None
    assert isinstance(near, ReferenceCdEntry)
    assert near.hull_id == "wigley_l100"
    assert near.froude_number == 0.20
    assert near.cd_published == 0.0035

    far = lookup_reference_cd("wigley_l100", 0.50, tolerance=0.02)
    assert far is None


def test_reference_cd_lookup_unknown_hull_returns_none() -> None:
    """A hull_id that's not in the database returns None even if Fr matches."""
    assert lookup_reference_cd("nonexistent_hull", 0.20, tolerance=0.05) is None


def test_reference_cd_lookup_picks_closest_entry_when_multiple_in_tolerance() -> None:
    """If several entries fall within the tolerance window, the lookup
    returns the closest one in Froude space."""
    # Wigley has Fr=0.20, 0.30, 0.35 entries. With tolerance 0.06 around
    # 0.32 both 0.30 and 0.35 are eligible; 0.30 is closer.
    closest = lookup_reference_cd("wigley_l100", 0.32, tolerance=0.06)
    assert closest is not None
    assert closest.froude_number == 0.30
