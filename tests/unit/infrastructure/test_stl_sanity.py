"""Tests for the STL sanity-check adapter.

Design reference: 2026-04-23-bulbopt-mesh-quality-design.md §4 (L6).

``validate_stl`` returns a dict
    {watertight, winding_consistent, volume, vertex_count, face_count,
     checks_passed}
Every top-candidate STL gets its own ``stl_valid.json`` and any failure
surfaces a warning in ``night_report.html``.
"""
from __future__ import annotations

import numpy as np
import pytest
import trimesh

from bulbopt.infrastructure.adapters.stl_sanity import validate_stl


def test_validate_stl_watertight_mesh_all_checks_pass() -> None:
    mesh = trimesh.creation.box(extents=(2.0, 1.0, 1.0))
    report = validate_stl(mesh)
    assert report["watertight"] is True
    assert report["winding_consistent"] is True
    assert report["volume"] > 0
    assert report["vertex_count"] > 0
    assert report["face_count"] > 0
    assert report["degenerate_face_count"] == 0
    assert report["high_aspect_face_count"] == 0
    assert report["geometry_risk"] == "low"
    assert report["failure_reasons"] == []
    assert report["checks_passed"] is True


def test_validate_stl_broken_mesh_fails_checks() -> None:
    mesh = trimesh.creation.box(extents=(2.0, 1.0, 1.0))
    broken = trimesh.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=mesh.faces[: len(mesh.faces) // 2].copy(),
        process=False,
    )
    report = validate_stl(broken)
    assert report["watertight"] is False
    # With half the faces removed, volume is ill-defined (not a closed
    # surface), checks_passed must be False so the report can warn.
    assert report["checks_passed"] is False


def test_validate_stl_reports_integer_counts() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    report = validate_stl(mesh)
    assert isinstance(report["vertex_count"], int)
    assert isinstance(report["face_count"], int)
    assert report["vertex_count"] == len(mesh.vertices)
    assert report["face_count"] == len(mesh.faces)


def test_validate_stl_volume_zero_marks_failure() -> None:
    """A degenerate (flat) mesh with zero volume should not pass."""
    # Two triangles forming a flat quad in the xy plane — zero volume.
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    flat = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    report = validate_stl(flat)
    assert report["checks_passed"] is False
    assert report["volume"] == pytest.approx(0.0, abs=1e-9)


def test_validate_stl_reports_degenerate_triangle_metrics() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    report = validate_stl(mesh)

    assert report["degenerate_face_count"] == 1
    # Audit 2026-04-26: degenerate triangles are a SOFT warning, not a
    # hard failure — a single degenerate triangle in an otherwise sound
    # baseline shouldn't reject the candidate downstream.
    assert "degenerate_faces" in report["warnings"]
    # The 1-triangle "mesh" is also non-watertight, so geometry_risk is
    # "high" because of the hard failure_reasons, not the soft warning.
    assert report["geometry_risk"] == "high"
    # And it fails checks because it's not watertight.
    assert report["checks_passed"] is False


def test_validate_stl_reports_high_aspect_triangle_metrics() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [0.0, 0.01, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    report = validate_stl(mesh)

    assert report["max_triangle_aspect_ratio"] > 1000.0
    assert report["high_aspect_face_count"] == 1
    # Audit 2026-04-26: high-aspect triangles are a SOFT warning, not a
    # hard failure (the baseline STL routinely contains 100+ such tris
    # inherited from the input triangulation; rejecting on them would
    # block every night-run candidate).
    assert "high_aspect_triangles" in report["warnings"]
    assert "high_aspect_triangles" not in report["failure_reasons"]


def test_validate_stl_passes_when_only_soft_warnings() -> None:
    """Audit 2026-04-26 regression: a watertight, winding-consistent
    mesh with positive volume but a few inherited high-aspect triangles
    must PASS the gate. Otherwise every night-run candidate downstream
    of a real ship hull (which always has slivers in the keel/transom
    triangulation) gets rejected."""
    # Start from a watertight icosphere then deliberately stretch ONE
    # vertex along Z to create a single high-aspect triangle without
    # breaking watertightness or volume.
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    assert mesh.is_watertight
    vertices = mesh.vertices.copy()
    # Move one vertex slightly tangent so we get one elongated face;
    # don't touch its connectivity so watertightness survives.
    vertices[0, 2] += 50.0
    sheared = trimesh.Trimesh(
        vertices=vertices,
        faces=mesh.faces,
        process=False,
    )
    if not bool(sheared.is_watertight):
        pytest.skip("icosphere shear unexpectedly broke watertightness")

    report = validate_stl(sheared)
    assert report["watertight"] is True
    assert report["winding_consistent"] is True
    # Volume must be positive after the shear.
    assert report["volume"] > 0.0
    # Aspect ratio must show the elongated triangle as a warning.
    assert report["max_triangle_aspect_ratio"] > 5.0
    # The gate MUST still pass — high-aspect / degenerate triangles are
    # warnings, not blockers.
    assert report["checks_passed"] is True
    # And those signals live in report["warnings"], NOT failure_reasons.
    assert "high_aspect_triangles" not in report.get("failure_reasons", [])
    assert "degenerate_faces" not in report.get("failure_reasons", [])
