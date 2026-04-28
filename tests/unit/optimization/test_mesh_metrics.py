"""Tests for mesh-quality metric used as the 3rd GA objective.

Design reference: 2026-04-23-bulbopt-mesh-quality-design.md §4 (L2).

``compute_mesh_quality(deformed)`` returns a scalar (lower is better):
    max(dihedral_angle_deviation, symmetry_error, watertight_penalty)
* ``watertight_penalty == 100`` if ``mesh.is_watertight`` is False else 0.
  Broken meshes therefore get dominated by NSGA-II (effectively killed).
"""
from __future__ import annotations

import numpy as np
import pytest
import trimesh

from bulbopt.optimization.quality.mesh_metrics import compute_mesh_quality


def test_watertight_box_mesh_has_low_quality_score() -> None:
    mesh = trimesh.creation.box(extents=(2.0, 1.0, 1.0))
    score = compute_mesh_quality(mesh)
    # Axis-aligned box has perfectly flat faces — dihedral deviation is
    # driven only by the 90-degree corners, and the mesh is symmetric
    # about y=0 if centred.
    assert score < 100.0


def test_broken_mesh_triggers_watertight_penalty() -> None:
    """Remove some faces so watertightness fails. The returned score must
    be >= 100 (watertight penalty) so NSGA-II treats the candidate as
    dominated."""
    mesh = trimesh.creation.box(extents=(2.0, 1.0, 1.0))
    # Drop half the faces to break watertightness.
    broken = trimesh.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=mesh.faces[: len(mesh.faces) // 2].copy(),
        process=False,
    )
    assert broken.is_watertight is False
    score = compute_mesh_quality(broken)
    assert score >= 100.0


def test_quality_score_is_nonnegative_and_finite() -> None:
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    score = compute_mesh_quality(mesh)
    assert np.isfinite(score)
    assert score >= 0.0


def test_compute_mesh_quality_respects_beam_axis_kwarg() -> None:
    """The default ``argmin(extents)`` heuristic picks the wrong beam axis
    on a real ship hull (audit 2026-04-26): on docs/base_hull.stl the Y
    extent is slightly smaller than Z, so the metric mirrors around Y
    (the asymmetric draft axis) instead of Z (the symmetric beam axis).
    When ``beam_axis`` is supplied explicitly, the metric must use it.

    Use a smooth icosphere (low dihedral deviation so the symmetry
    component dominates ``max(...)``) and squash its lower half so Y is
    teardrop-asymmetric while Z stays a perfect mirror."""
    base = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    verts = np.asarray(base.vertices, dtype=float).copy()
    verts[:, 0] *= 4.0           # primary = X
    # Z stays at unit radius (symmetric ±1, beam axis).
    # Squash the bottom (-Y) so the Y distribution is teardrop-shaped:
    # vertices with Y < 0 are pulled toward the centerline.
    verts[verts[:, 1] < 0, 1] *= 0.3
    mesh = trimesh.Trimesh(vertices=verts, faces=base.faces, process=False)

    score_z = compute_mesh_quality(mesh, beam_axis=2)
    score_y = compute_mesh_quality(mesh, beam_axis=1)

    assert score_z < score_y, (
        f"beam_axis kwarg ignored: Z-quality {score_z:.6f} should be smaller "
        f"than Y-quality {score_y:.6f} on a Z-symmetric, Y-teardrop hull"
    )


# Audit 2026-04-26 — Add #3: bulb/aft self-intersection penalty. The
# existing components (dihedral, symmetry, watertight) cannot detect
# the case where the FFD has pushed the bulb sub-mesh through the rest
# of the hull. The penalty counts the fraction of aft-region vertices
# that fall *inside* the bulb sub-mesh's signed-distance field.


def _two_blob_mesh(
    *,
    bulb_origin: tuple[float, float, float],
    bulb_radius: float = 0.5,
    aft_origin: tuple[float, float, float] = (-1.5, 0.0, 0.0),
    aft_radius: float = 0.5,
    subdivisions: int = 2,
) -> trimesh.Trimesh:
    """Two disjoint icospheres concatenated into a single (non-water-
    tight as a *whole*, but each blob is watertight) mesh."""
    bulb = trimesh.creation.icosphere(subdivisions=subdivisions, radius=bulb_radius)
    bulb.apply_translation(np.asarray(bulb_origin, dtype=float))
    aft = trimesh.creation.icosphere(subdivisions=subdivisions, radius=aft_radius)
    aft.apply_translation(np.asarray(aft_origin, dtype=float))
    return trimesh.util.concatenate([bulb, aft])


def test_intersection_penalty_zero_when_bulb_does_not_intersect_aft() -> None:
    """A healthy candidate: bulb sphere at (1.5, 0, 0), aft sphere at
    (-1.5, 0, 0). With ``axis_min`` placed between them, every aft-
    region vertex is well outside the bulb's signed-distance field, so
    the intersection penalty is zero and the overall quality stays low.
    """
    mesh = _two_blob_mesh(bulb_origin=(1.5, 0.0, 0.0))
    region = {
        "axis_index": 0,
        "axis_min": 1.0,    # forward of x=1 lives the bulb
        "axis_max": 2.0,
        "blend_width": 0.2,
    }
    score = compute_mesh_quality(mesh, region=region)
    assert np.isfinite(score)
    assert score < 100.0, f"healthy candidate must not trigger watertight penalty (score={score})"


def test_intersection_penalty_positive_when_bulb_passes_through_aft() -> None:
    """A pathological candidate: the bulb sphere is enlarged so that it
    overlaps the aft sphere. Many aft-region vertices fall inside the
    bulb's signed-distance field, so the intersection penalty raises
    the overall quality vs. the disjoint-blob baseline."""
    healthy = _two_blob_mesh(bulb_origin=(1.5, 0.0, 0.0), bulb_radius=0.5)
    pathological = _two_blob_mesh(bulb_origin=(0.0, 0.0, 0.0), bulb_radius=2.0)

    region = {
        "axis_index": 0,
        "axis_min": -0.5,
        "axis_max": 2.0,
        "blend_width": 0.2,
    }
    score_clean = compute_mesh_quality(healthy, region=region)
    score_bad = compute_mesh_quality(pathological, region=region)
    assert np.isfinite(score_clean)
    assert np.isfinite(score_bad)
    assert score_bad > score_clean + 0.05, (
        f"intersecting candidate should be penalised; clean={score_clean:.4f} "
        f"bad={score_bad:.4f}"
    )


def test_intersection_penalty_skipped_when_bulb_not_watertight() -> None:
    """When the bulb sub-mesh isn't watertight, ``signed_distance`` is
    meaningless and the penalty must be skipped (no crash, finite
    score)."""
    bulb = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    bulb.apply_translation(np.array([1.5, 0.0, 0.0]))
    aft = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    aft.apply_translation(np.array([-1.5, 0.0, 0.0]))
    mesh = trimesh.util.concatenate([bulb, aft])
    # Drop half the bulb's faces (the *first* concatenated half) so the
    # bulb sub-mesh is no longer watertight.
    half_bulb = len(bulb.faces) // 2
    broken_faces = np.concatenate([mesh.faces[half_bulb:len(bulb.faces)], mesh.faces[len(bulb.faces):]], axis=0)
    broken = trimesh.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=broken_faces.copy(),
        process=False,
    )
    region = {
        "axis_index": 0,
        "axis_min": 1.0,
        "axis_max": 2.0,
        "blend_width": 0.2,
    }
    score = compute_mesh_quality(broken, region=region)
    assert np.isfinite(score), f"score should be finite on non-watertight bulb sub-mesh, got {score}"
