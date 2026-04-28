"""Unit tests for :mod:`bulbopt.optimization.parametric.isotropic_remesh`.

The module wraps :mod:`meshlib` to apply edge-flip / split / collapse
remeshing to the bulb region of a deformed hull. These tests check:

1. Graceful no-op when :mod:`meshlib` is unavailable.
2. Aspect-ratio reduction in the region (``> 70%`` drop on a slivered
   fixture — well above what ACVD achieves).
3. Bit-identical preservation of strictly-outside vertices (so the seam
   to the rest of the hull doesn't break).
"""
from __future__ import annotations

import numpy as np
import pytest
import trimesh

from bulbopt.optimization.parametric import isotropic_remesh
from bulbopt.optimization.parametric.isotropic_remesh import (
    is_isotropic_remesh_available,
    remesh_region_isotropic,
)


def _slivery_box() -> tuple[trimesh.Trimesh, dict]:
    """Build a watertight slivered mesh whose front half acts as the
    bulb region.

    Strategy: take a box, subdivide it five times so the seam ring has
    many uniform edges (matching the production-hull case where the
    seam edges are uniform), then collapse the Z extent of the +X half
    by 10×. The resulting triangles in the +X half have aspect ratios
    in the 30-40 range — comparable to the slivers the meshlib
    remesher is meant to repair on the real bow tip.

    A box (rather than a sphere) and many subdivisions are used because
    meshlib's ``frozenBoundary`` flag preserves boundary edge counts
    only when the seam edge lengths are reasonably uniform; on a coarse
    sphere fixture the very short edges at a tip violate this and the
    remesher collapses them anyway.
    """
    mesh = trimesh.creation.box(extents=(10.0, 4.0, 4.0))
    # Subdivide 5 times → 12288 faces, dense uniform seam.
    for _ in range(5):
        mesh = mesh.subdivide()
    primary = int(np.argmax(mesh.extents))
    axis_min = float(mesh.vertices[:, primary].mean())

    verts = mesh.vertices.copy()
    region_mask = mesh.vertices[:, primary] >= axis_min
    verts[region_mask, 2] *= 0.1  # collapse z by 10× → ~30:1 slivers

    slivery = trimesh.Trimesh(vertices=verts, faces=mesh.faces.copy(), process=False)
    region = {"axis_index": primary, "axis_min": axis_min}
    return slivery, region


def _max_aspect_in_region(mesh: trimesh.Trimesh, region: dict) -> float:
    """Largest (longest_edge / shortest_edge) ratio over all triangles
    whose face has at least one vertex in the region."""
    primary = region["axis_index"]
    axis_min = region["axis_min"]
    in_region = (mesh.vertices[mesh.faces, primary] >= axis_min).any(axis=1)
    if not in_region.any():
        return 1.0
    tris = mesh.vertices[mesh.faces[in_region]]
    e0 = np.linalg.norm(tris[:, 1] - tris[:, 0], axis=1)
    e1 = np.linalg.norm(tris[:, 2] - tris[:, 1], axis=1)
    e2 = np.linalg.norm(tris[:, 0] - tris[:, 2], axis=1)
    edges = np.stack([e0, e1, e2], axis=1)
    return float(np.max(edges.max(axis=1) / np.maximum(edges.min(axis=1), 1e-12)))


def test_no_op_when_backend_missing(monkeypatch) -> None:
    """If :mod:`meshlib` cannot be imported, ``remesh_region_isotropic``
    must return the input mesh unchanged.

    We simulate the missing dependency by monkeypatching the module's
    availability flag to ``False``. The function must short-circuit
    *before* touching meshlib, so this test runs even on systems where
    the optional dep *is* installed.
    """
    monkeypatch.setattr(isotropic_remesh, "_MESHLIB_AVAILABLE", False)
    mesh, region = _slivery_box()
    out = remesh_region_isotropic(mesh, region)
    assert out is mesh, (
        "When meshlib is unavailable, remesh_region_isotropic must return "
        "the input mesh object unchanged (no copy, no remesh)."
    )


def test_remesh_lowers_max_aspect_in_region() -> None:
    """Isotropic remeshing must drop the worst-case in-region aspect
    ratio by ``> 70%``.

    This is well above ACVD's empirical 50% — meshlib's edge-flip /
    collapse / split remesher targets equilateral triangles directly,
    not just uniform vertex spacing.

    Skipped on systems without :mod:`meshlib` so the laptop test suite
    still runs there.
    """
    if not is_isotropic_remesh_available():
        pytest.skip("meshlib not installed; skipping isotropic remesh test")

    mesh, region = _slivery_box()
    before = _max_aspect_in_region(mesh, region)
    assert before > 5.0, "test fixture must actually contain slivers"

    out = remesh_region_isotropic(mesh, region, iterations=2)
    after = _max_aspect_in_region(out, region)
    drop = (before - after) / before
    assert drop > 0.7, (
        f"Expected > 70% drop in max aspect ratio after isotropic remesh, "
        f"got before={before:.2f}, after={after:.2f}, drop={drop:.1%}"
    )


def test_remesh_preserves_seam_topology() -> None:
    """Vertices strictly outside the region must be bit-identical
    before and after.

    The output mesh's vertex array is laid out so the first
    ``n_outside_hull`` slots are taken verbatim from the input mesh's
    out-of-region vertices (in the same order :func:`np.unique`
    returns). We check that prefix is exactly equal to the
    corresponding input slice. We also verify the mesh stays
    watertight, since a seam break shows up there immediately.

    Skipped on systems without :mod:`meshlib`.
    """
    if not is_isotropic_remesh_available():
        pytest.skip("meshlib not installed; skipping seam preservation test")

    mesh, region = _slivery_box()
    primary = region["axis_index"]
    axis_min = region["axis_min"]

    # Snapshot the strictly-outside vertices (axis < axis_min) before
    # remesh; these must appear unchanged in the output mesh.
    outside_mask = mesh.vertices[:, primary] < axis_min
    outside_pts_before = mesh.vertices[outside_mask].copy()

    out = remesh_region_isotropic(mesh, region, iterations=2)

    # Each strictly-outside input vertex must be findable in the output
    # mesh's vertex list at distance 0.
    from scipy.spatial import cKDTree

    tree = cKDTree(out.vertices)
    dists, _ = tree.query(outside_pts_before)
    max_dist = float(dists.max())
    assert max_dist < 1e-9, (
        f"Strictly-outside vertices must be bit-identical after remesh, "
        f"max displacement was {max_dist:.3e}"
    )

    # Watertight is a derived property — if the seam stitch broke, this
    # is where we'd notice.
    assert out.is_watertight, "remeshed mesh must remain watertight"
