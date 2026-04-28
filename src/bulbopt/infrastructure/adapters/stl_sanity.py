"""STL sanity validator — companion check alongside every top-candidate
STL written to ``outputs/top_candidates/candidate-XXX/``.

Design reference: 2026-04-23-bulbopt-mesh-quality-design.md §4 (L6).

``validate_stl(mesh) -> dict`` returns six fields:

    watertight          bool
    winding_consistent  bool
    volume              float    (zero for broken/flat meshes)
    vertex_count        int
    face_count          int
    checks_passed       bool     (True iff all the other invariants hold)

The use case writes this dict as ``stl_valid.json`` next to each STL,
and the night-run HTML report consumes the failure list to surface a
warning block.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import trimesh


def validate_stl(mesh: trimesh.Trimesh) -> Dict[str, Any]:
    """Inspect a Trimesh and return the sanity-check summary.

    All numerical fields are plain Python floats / ints so the result
    can be JSON-serialised without ``default=str`` hacks.
    """
    try:
        watertight = bool(mesh.is_watertight)
    except Exception:
        watertight = False

    try:
        winding_consistent = bool(mesh.is_winding_consistent)
    except Exception:
        winding_consistent = False

    try:
        if mesh.is_volume:
            volume = float(abs(mesh.volume))
        else:
            volume = 0.0
    except Exception:
        volume = 0.0

    vertex_count = int(len(mesh.vertices))
    face_count = int(len(mesh.faces))
    triangle_metrics = _triangle_quality_metrics(mesh)
    # Hard failures — must fail the gate.
    failure_reasons: list[str] = []
    # Soft warnings — surfaced in the report but do NOT block the gate.
    # Audit 2026-04-26 finding: hull triangulations inherited from the
    # baseline STL routinely contain a small fraction of high-aspect
    # / near-degenerate triangles (the canonical docs/base_hull.stl has
    # 125 high-aspect tris with max aspect 9853 — pre-existing in the
    # input, not introduced by the deformer). Treating those as hard
    # failures rejects EVERY night-run candidate downstream of an
    # inherited mesh, regardless of how good the deformation is.
    warnings: list[str] = []
    if not watertight:
        failure_reasons.append("not_watertight")
    if not winding_consistent:
        failure_reasons.append("winding_inconsistent")
    if volume <= 0.0:
        failure_reasons.append("non_positive_volume")
    if vertex_count <= 0:
        failure_reasons.append("no_vertices")
    if face_count <= 0:
        failure_reasons.append("no_faces")
    if triangle_metrics["degenerate_face_count"] > 0:
        warnings.append("degenerate_faces")
    if triangle_metrics["high_aspect_face_count"] > 0:
        warnings.append("high_aspect_triangles")

    geometry_risk = "low"
    if failure_reasons:
        geometry_risk = "high"
    elif warnings:
        geometry_risk = "medium"
    elif triangle_metrics["max_triangle_aspect_ratio"] > 20.0:
        geometry_risk = "medium"

    checks_passed = bool(
        watertight
        and winding_consistent
        and volume > 0.0
        and vertex_count > 0
        and face_count > 0
    )

    return {
        "watertight": watertight,
        "winding_consistent": winding_consistent,
        "volume": volume,
        "vertex_count": vertex_count,
        "face_count": face_count,
        "min_face_area": triangle_metrics["min_face_area"],
        "max_triangle_aspect_ratio": triangle_metrics["max_triangle_aspect_ratio"],
        "degenerate_face_count": triangle_metrics["degenerate_face_count"],
        "high_aspect_face_count": triangle_metrics["high_aspect_face_count"],
        "geometry_risk": geometry_risk,
        "failure_reasons": failure_reasons,
        "warnings": warnings,
        "checks_passed": checks_passed,
    }


def _triangle_quality_metrics(mesh: trimesh.Trimesh) -> Dict[str, Any]:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    if len(vertices) == 0 or len(faces) == 0:
        return {
            "min_face_area": 0.0,
            "max_triangle_aspect_ratio": 0.0,
            "degenerate_face_count": 0,
            "high_aspect_face_count": 0,
        }

    try:
        triangles = vertices[faces]
    except Exception:
        return {
            "min_face_area": 0.0,
            "max_triangle_aspect_ratio": float("inf"),
            "degenerate_face_count": int(len(faces)),
            "high_aspect_face_count": int(len(faces)),
        }

    edges = np.stack(
        [
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 1],
            triangles[:, 0] - triangles[:, 2],
        ],
        axis=1,
    )
    edge_lengths = np.linalg.norm(edges, axis=2)
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    areas = 0.5 * np.linalg.norm(cross, axis=1)

    area_eps = 1e-12
    edge_eps = 1e-12
    degenerate = (areas <= area_eps) | (edge_lengths.min(axis=1) <= edge_eps)
    with np.errstate(divide="ignore", invalid="ignore"):
        aspect = edge_lengths.max(axis=1) / np.maximum(edge_lengths.min(axis=1), edge_eps)
    aspect = np.where(np.isfinite(aspect), aspect, float("inf"))
    high_aspect = aspect > 1000.0

    return {
        "min_face_area": float(areas.min()) if len(areas) else 0.0,
        "max_triangle_aspect_ratio": float(aspect.max()) if len(aspect) else 0.0,
        "degenerate_face_count": int(np.count_nonzero(degenerate)),
        "high_aspect_face_count": int(np.count_nonzero(high_aspect)),
    }
