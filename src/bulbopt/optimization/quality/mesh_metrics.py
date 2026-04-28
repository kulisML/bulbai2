"""Mesh-quality scalar metric used as the 3rd NSGA-II objective.

Design reference: 2026-04-23-bulbopt-mesh-quality-design.md §4 (L2).

The GA minimises a composite

    mesh_quality = max(
        dihedral_angle_deviation,   # high-frequency facet roughness
        symmetry_error,             # deviation from y -> -y symmetry
        watertight_penalty,         # 100 if broken, 0 if watertight
    ) + intersection_penalty        # bulb sub-mesh poking into aft hull

so the single worst classical indicator governs, and the bulb/aft
self-intersection adds an additive cost on top so the GA can still
distinguish "clean watertight bulb" from "watertight bulb that pokes
through the aft hull" — the latter would otherwise score the same.

Audit 2026-04-26 (Add #3): the ``intersection_penalty`` term is only
active when the caller supplies a ``region`` describing the bulb's
axis. Without ``region`` the function falls back to the legacy
``max(...)`` composite so existing callers stay bit-identical.

The individual components are intentionally simple so the metric is
cheap enough to run inside the mid-gate on every GA individual.
"""
from __future__ import annotations

import numpy as np
import trimesh


WATERTIGHT_PENALTY = 100.0

# Audit 2026-04-26 (Add #3): default fraction of the bulb-region length
# used to define the "aft" half-space when the region dict doesn't carry
# an explicit ``blend_width``. Mirrors the ``BLEND_WIDTH_FRACTION`` used
# by ``BulbFFDDeformer`` so the intersection check operates on the same
# blend boundary the deformer actually used.
_DEFAULT_BLEND_WIDTH_FRACTION = 0.10


def _dihedral_deviation(mesh: trimesh.Trimesh) -> float:
    """RMS deviation of face-adjacency dihedral angles from the local
    mean, normalised into [0, 1].

    A perfectly smooth (curvature-constant) surface has near-zero
    deviation. Facetted or noisy regions have high deviation. We use
    ``mesh.face_adjacency_angles`` which trimesh computes once per mesh.
    """
    try:
        angles = mesh.face_adjacency_angles
    except Exception:
        return 0.0
    if angles is None or len(angles) == 0:
        return 0.0
    a = np.asarray(angles, dtype=float)
    mean = float(np.mean(a))
    deviation = float(np.sqrt(np.mean((a - mean) ** 2)))
    # Scale by pi so an angle spread of a radian maps to ~0.32.
    return deviation / float(np.pi)


def _symmetry_error(
    mesh: trimesh.Trimesh, beam_axis: int | None = None
) -> float:
    """Vertex-to-mirror RMS distance, normalised by bounding-box diagonal.

    Mirrors the hull about its centreline on the beam axis and measures
    how close each vertex is to its mirrored counterpart.

    When ``beam_axis`` is ``None`` (default) the function falls back to
    the legacy ``argmin(extents)`` heuristic — fine for synthetic test
    meshes where all non-primary axes are centered on zero. Production
    callers should pass ``beam_axis=region["beam_axis"]`` so the metric
    measures symmetry around the truly symmetric axis of the hull
    (audit 2026-04-26).

    Returns a non-negative float; 0 means perfectly symmetric.
    """
    vertices = np.asarray(mesh.vertices, dtype=float)
    if vertices.size == 0:
        return 0.0
    extents = mesh.extents.astype(float)
    if np.any(extents <= 0):
        return 0.0

    if beam_axis is None:
        beam_axis = int(np.argmin(extents))
    else:
        beam_axis = int(beam_axis)
    centre = 0.5 * (vertices[:, beam_axis].max() + vertices[:, beam_axis].min())

    mirrored = vertices.copy()
    mirrored[:, beam_axis] = 2.0 * centre - vertices[:, beam_axis]

    # For each mirrored vertex, find the closest original vertex.
    try:
        from scipy.spatial import cKDTree  # type: ignore
    except Exception:
        # Fallback: vectorised pairwise distance (O(N^2)). For typical
        # mid-gate meshes (<= 5k vertices) this is still acceptable.
        diffs = vertices[:, None, :] - mirrored[None, :, :]
        d = np.linalg.norm(diffs, axis=2)
        rms = float(np.sqrt(np.mean(d.min(axis=1) ** 2)))
    else:
        tree = cKDTree(vertices)
        distances, _ = tree.query(mirrored, k=1)
        rms = float(np.sqrt(np.mean(np.asarray(distances, dtype=float) ** 2)))

    diagonal = float(np.linalg.norm(extents))
    if diagonal <= 0:
        return 0.0
    return rms / diagonal


def _watertight_penalty(mesh: trimesh.Trimesh) -> float:
    try:
        if bool(mesh.is_watertight):
            return 0.0
    except Exception:
        pass
    return WATERTIGHT_PENALTY


def _build_bulb_submesh(
    mesh: trimesh.Trimesh,
    bulb_mask: np.ndarray,
) -> trimesh.Trimesh | None:
    """Audit 2026-04-26 (Add #3) — build the bulb sub-mesh used for
    the signed-distance / point-in-volume query.

    We use a *face-based* identifier rather than the strict vertex
    half-space: the bulb sub-mesh is every face that has at least one
    vertex in the bulb half-space. This is necessary because in a
    pathologically FFD-deformed mesh the bulb's geometry can extend
    its triangulation past the axis_min boundary — vertices that
    "originally" belonged to the bulb may now sit at axis < axis_min,
    and we need to capture those triangles in the bulb shape so the
    intersection check can see them.

    Returns the sub-mesh, or ``None`` when the construction fails (no
    qualifying faces, or trimesh raises).
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.size == 0:
        return None
    face_in_bulb = bulb_mask[faces].any(axis=1)
    if not face_in_bulb.any():
        return None
    try:
        sub = mesh.submesh([np.nonzero(face_in_bulb)[0]], append=True)
    except Exception:
        return None
    if sub is None:
        return None
    if not isinstance(sub, trimesh.Trimesh):
        try:
            sub = trimesh.util.concatenate(sub)  # type: ignore[arg-type]
        except Exception:
            return None
    if len(sub.vertices) == 0 or len(sub.faces) == 0:
        return None
    return sub


def _intersection_penalty(
    mesh: trimesh.Trimesh, region: dict | None
) -> float:
    """Audit 2026-04-26 (Add #3) — fraction of aft-region vertices that
    lie *inside* the bulb sub-mesh's volume.

    Splits the input mesh by the primary axis into a forward "bulb"
    half-space and a sufficiently-aft "aft" half-space, separated by a
    blend gap so vertices in the smoothstep-blended seam don't
    contaminate either side.

    The bulb's volume is approximated by the *convex hull* of the
    bulb sub-mesh's vertices. The bulb sub-mesh itself is built from
    every face that has at least one vertex in the bulb half-space —
    this captures triangles whose tongue extends into the aft region
    when the FFD has pushed the bulb past axis_min, which is exactly
    the failure mode we want to detect.

    We use the convex hull (rather than the raw bulb sub-mesh) because:

    * The face-filtered sub-mesh has an open seam where it meets the
      rest of the hull, so it's typically not watertight and
      ``trimesh.proximity.signed_distance`` is undefined.
    * Point-in-hull tests via ``scipy.spatial.Delaunay.find_simplex``
      are robust and have no dependency on the optional ``rtree``
      package that trimesh's ``contains`` / ``signed_distance`` need.
    * The hull *over*-approximates a concave bulb, so the penalty
      slightly over-counts intersections — the conservative direction
      for a quality objective (we'd rather flag a near-miss than miss
      a real intersection).

    Per the audit contract, the check is skipped (returns 0) when no
    closed bulb representation is available. The convex hull is closed
    by construction, so the only "skip" path is when the hull itself
    fails to build — typically too few or coplanar vertices, or a
    scipy QhullError on degenerate inputs.

    Returns a value in ``[0, 1]``. Falls through to ``0.0`` when:
    * ``region`` is ``None`` or missing ``axis_index`` / ``axis_min``;
    * either side is empty (degenerate split);
    * the bulb sub-mesh can't be constructed;
    * fewer than 4 non-coplanar bulb vertices (no hull possible);
    * scipy's Delaunay/QHull raise (defensive — the metric must stay
      finite for NSGA-II).
    """
    if region is None:
        return 0.0
    if "axis_index" not in region or "axis_min" not in region:
        return 0.0

    primary_axis = int(region["axis_index"])
    axis_min = float(region["axis_min"])

    vertices = np.asarray(mesh.vertices, dtype=float)
    if vertices.size == 0:
        return 0.0

    # Width of the smoothstep blend zone — vertices inside it are
    # excluded from the aft half-space because they're partially
    # deformed and naturally sit close to the bulb surface. Mirrors
    # the deformer's BLEND_WIDTH_FRACTION (0.10 of the region length).
    if "blend_width" in region and region["blend_width"] is not None:
        blend_width = float(region["blend_width"])
    else:
        axis_max = float(region.get("axis_max", float(vertices[:, primary_axis].max())))
        blend_width = max(
            _DEFAULT_BLEND_WIDTH_FRACTION * (axis_max - axis_min), 0.0
        )

    bulb_mask = vertices[:, primary_axis] >= axis_min
    aft_mask = vertices[:, primary_axis] < (axis_min - blend_width)

    if not bulb_mask.any() or not aft_mask.any():
        return 0.0

    sub = _build_bulb_submesh(mesh, bulb_mask)
    if sub is None:
        return 0.0

    sub_vertices = np.asarray(sub.vertices, dtype=float)
    if len(sub_vertices) < 4:
        return 0.0

    # Audit contract: skip when no closed bulb representation is
    # available. With ``signed_distance`` this would be the watertight
    # check; with the ``Delaunay``-based convex-hull point-in-volume
    # test the closed surface is the hull itself, which is always
    # closed by construction. We still inherit the audit's skip path
    # via ``Delaunay``'s ``QhullError`` on degenerate inputs.
    try:
        from scipy.spatial import Delaunay, QhullError
    except ImportError:  # pragma: no cover — scipy is a hard dep
        return 0.0

    try:
        delaunay = Delaunay(sub_vertices)
    except (QhullError, ValueError):
        return 0.0
    except Exception:  # noqa: BLE001 — defensive against scipy edge cases
        return 0.0

    aft_points = vertices[aft_mask]
    aft_count = int(len(aft_points))
    if aft_count == 0:
        return 0.0

    try:
        # ``find_simplex`` returns a non-negative simplex index when
        # the point is inside the hull, -1 outside. Vectorised over
        # the input array.
        inside_arr = np.asarray(delaunay.find_simplex(aft_points)) >= 0
    except Exception:  # noqa: BLE001 — defensive against scipy edge cases
        return 0.0

    inside = int(np.count_nonzero(inside_arr))
    fraction = inside / float(aft_count)
    if not np.isfinite(fraction):
        return 0.0
    return float(min(max(fraction, 0.0), 1.0))


def compute_mesh_quality(
    deformed: trimesh.Trimesh,
    beam_axis: int | None = None,
    region: dict | None = None,
) -> float:
    """Return a composite quality scalar for the deformed mesh.

    Lower is better. Non-negative and finite. A broken (non-watertight)
    mesh returns ``>= WATERTIGHT_PENALTY`` so NSGA-II dominates it.

    Pass ``beam_axis`` (typically ``region["beam_axis"]`` from the
    geometry analysis) so the symmetry sub-metric mirrors around the
    truly symmetric axis of the hull. The default ``None`` keeps the
    legacy ``argmin(extents)`` heuristic for callers that don't have a
    beam axis at hand (e.g. unit tests on icospheres / boxes).

    Audit 2026-04-26 (Add #3): when ``region`` is supplied (with
    ``axis_index`` + ``axis_min``), the function adds a self-
    intersection penalty in ``[0, 1]`` measuring the fraction of aft-
    region vertices that fall inside the bulb sub-mesh's volume
    (approximated by the convex hull of the bulb sub-mesh's vertices,
    so the test is robust to small surface holes). The penalty is
    additive so a clean candidate sees no behavioural change but a
    candidate whose FFD has driven the bulb through the aft hull gets
    clearly ranked worse by NSGA-II. Default ``region=None`` keeps the
    legacy ``max(...)`` composite.
    """
    classical = max(
        _dihedral_deviation(deformed),
        _symmetry_error(deformed, beam_axis=beam_axis),
        _watertight_penalty(deformed),
    )
    intersection = _intersection_penalty(deformed, region)
    return float(classical + intersection)
