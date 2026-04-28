"""Unit tests for :func:`subdivide_region` (design §4 L5)."""
from __future__ import annotations

import numpy as np
import pytest
import trimesh

from bulbopt.optimization.parametric.adaptive_subdivision import subdivide_region


def _box_region(extents=(4.0, 1.5, 1.0)) -> tuple[trimesh.Trimesh, dict]:
    mesh = trimesh.creation.box(extents=extents)
    primary = int(np.argmax(mesh.extents))
    axis_min = float(mesh.vertices[:, primary].mean())  # half of the box
    axis_max = float(mesh.vertices[:, primary].max())
    region = {"axis_index": primary, "axis_min": axis_min, "axis_max": axis_max}
    return mesh, region


def test_coarse_box_gets_subdivided_to_meet_threshold():
    mesh, region = _box_region()
    out = subdivide_region(mesh, region, min_triangles=50, max_iterations=3)
    centroids = np.asarray(out.triangles_center, dtype=float)
    primary = region["axis_index"]
    region_face_count = int((centroids[:, primary] >= region["axis_min"]).sum())
    assert region_face_count >= 50
    # Output remains watertight so snappyHexMesh can still consume the STL.
    assert out.is_watertight


def test_coarse_box_respects_max_iterations_cap():
    mesh, region = _box_region()
    # Ask for an impossible threshold so the cap is the binding constraint.
    out = subdivide_region(mesh, region, min_triangles=100_000, max_iterations=2)
    centroids = np.asarray(out.triangles_center, dtype=float)
    primary = region["axis_index"]
    region_face_count = int((centroids[:, primary] >= region["axis_min"]).sum())
    # 6 region faces, 4x subdivision per step, 2 steps → 6 * 16 = 96
    # (trimesh's 1-to-4 subdivision pattern). The cap stops the loop.
    assert region_face_count <= 200  # well below 100_000
    assert out.is_watertight


def test_dense_box_is_not_resubdivided():
    mesh, region = _box_region()
    # Pre-densify once globally so the region already has plenty of faces.
    dense = mesh.subdivide().subdivide()
    primary = region["axis_index"]
    centroids = np.asarray(dense.triangles_center, dtype=float)
    initial_region_count = int((centroids[:, primary] >= region["axis_min"]).sum())
    assert initial_region_count >= 24  # 6 * 4

    out = subdivide_region(
        dense,
        region,
        min_triangles=initial_region_count - 1,  # already above threshold
        max_iterations=3,
    )
    # Face count is unchanged.
    assert len(out.faces) == len(dense.faces)
    assert out.is_watertight


def test_output_is_always_a_new_mesh():
    mesh, region = _box_region()
    out = subdivide_region(mesh, region, min_triangles=1, max_iterations=3)
    assert out is not mesh


def test_invalid_arguments_raise():
    mesh, region = _box_region()
    with pytest.raises(ValueError):
        subdivide_region(mesh, region, min_triangles=0, max_iterations=1)
    with pytest.raises(ValueError):
        subdivide_region(mesh, region, min_triangles=10, max_iterations=-1)


def test_subdivision_increases_region_face_count_monotonically():
    mesh, region = _box_region()
    primary = region["axis_index"]
    counts = []
    current = mesh
    for max_iter in (0, 1, 2, 3):
        result = subdivide_region(
            current, region, min_triangles=10_000, max_iterations=max_iter
        )
        centroids = np.asarray(result.triangles_center, dtype=float)
        counts.append(int((centroids[:, primary] >= region["axis_min"]).sum()))
    # Strictly non-decreasing.
    assert counts == sorted(counts)
    # 0 iterations → unchanged
    assert counts[0] == 6  # default box baseline


def test_region_with_no_faces_is_passthrough():
    mesh = trimesh.creation.box(extents=(4.0, 1.5, 1.0))
    primary = int(np.argmax(mesh.extents))
    # axis_min beyond max → no face centroid qualifies.
    bad_region = {
        "axis_index": primary,
        "axis_min": float(mesh.vertices[:, primary].max()) + 10.0,
    }
    out = subdivide_region(mesh, bad_region, min_triangles=100, max_iterations=3)
    assert len(out.faces) == len(mesh.faces)
    assert out.is_watertight


def test_subdivision_dedup_tolerance_scales_with_edge_length():
    """Bug #7: the midpoint-dedup uses ``dists[min_idx] <= 1e-18`` (squared
    distance, so 1e-9 m linear). On a 100 m-class hull the analytic
    midpoint can differ from trimesh's by ~1e-7 m due to internal float
    arithmetic — the check rejects valid midpoints, leaves T-junctions,
    output mesh fails ``is_watertight``.

    After the fix, the tolerance scales with the mesh's edge length so
    coarse meshes at any scale (small bow STLs through 300 m container
    ships) remain watertight after ``subdivide_region``.
    """
    # Small mesh (10 m extent) and large mesh (10000 m extent).
    small = trimesh.creation.box(extents=(10.0, 4.0, 3.0))
    large = trimesh.creation.box(extents=(10000.0, 4000.0, 3000.0))

    for mesh, label in ((small, "10 m"), (large, "10000 m")):
        primary = int(np.argmax(mesh.extents))
        axis_max = float(mesh.vertices[:, primary].max())
        axis_min = float(mesh.vertices[:, primary].mean())
        region = {
            "axis_index": primary,
            "axis_min": axis_min,
            "axis_max": axis_max,
        }
        out = subdivide_region(mesh, region, min_triangles=200, max_iterations=3)
        assert out.is_watertight, (
            f"{label} mesh lost watertightness after subdivision — Bug #7 "
            f"(midpoint-dedup tolerance not scale-aware)"
        )

        # Repeated subdivision must remain idempotent: once the threshold
        # is met, the function returns the input unchanged.
        out2 = subdivide_region(out, region, min_triangles=1, max_iterations=3)
        assert len(out2.faces) == len(out.faces), (
            f"{label} subdivision is not idempotent under repeated calls"
        )
        assert out2.is_watertight
