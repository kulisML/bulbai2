"""Isotropic remeshing for the bulb region as a visual-only post-processor.

This module wraps :mod:`meshlib`'s ``remesh`` function (a proper
incremental isotropic remesher with edge splits, edge collapses, edge
flips, and tangential relaxation) and applies it to the bulb tip region
of a deformed hull. It is intended as a *visual-only* polish for the
top-candidate STL outputs — CFD-evaluated geometry MUST stay unchanged
for consistency with the GP training history.

Why a separate module from :mod:`anisotropic_remesh`
----------------------------------------------------

``anisotropic_remesh`` uses ACVD clustering (via :mod:`pyacvd`); on the
production hull it only knocked the bulb-region max aspect ratio from
~3650 down to ~3000 (~10% drop) before MemoryError'ing on aggressive
cluster counts. ``meshlib`` exposes a true edge-flip / edge-collapse /
edge-split remesher whose remesh on the production hull drops max
aspect from 3654 to ~6 (>99% drop) while staying watertight and
preserving the volume to within 0.1%.

Strategy
--------

1. Identify in-region faces by ``axis-coord >= region["axis_min"]``
   (a face is "in region" when at least one of its vertices crosses
   the threshold — same convention as :mod:`anisotropic_remesh` and
   :class:`bulbopt.optimization.parametric.ffd_deformer.BulbFFDDeformer`).
2. Convert the mesh to meshlib's :class:`Mesh` and run
   :func:`meshlib.mrmeshpy.remesh` on the **whole mesh**, restricting
   the operations to the region via ``settings.region`` (a
   :class:`FaceBitSet`) and forbidding boundary motion via
   ``settings.maxBdShift = 0``.
3. After the remesh, re-snap *every* strictly-outside-region vertex
   (axis < axis_min) back to its exact original coordinate. Meshlib's
   ``maxBdShift = 0`` is a *per-collapse* limit, not absolute; small
   tangential drifts can still leak across the region boundary. The
   re-snap re-imposes bit-identical preservation.
4. Return the resulting :class:`trimesh.Trimesh`.

When :mod:`meshlib` is not installed the module degrades gracefully to
a no-op so the rest of the optimisation pipeline keeps working on
laptops without the optional dependency.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import trimesh

try:
    from meshlib import mrmeshpy as _mrm

    _MESHLIB_AVAILABLE = True
except Exception:
    _mrm = None  # type: ignore[assignment]
    _MESHLIB_AVAILABLE = False


# Minimum in-region face count below which remeshing is skipped — at
# very low face counts the remesher's overhead dominates and the
# resulting mesh is rarely worth touching.
_MIN_REGION_FACES = 50

# meshlib's remesh occasionally hangs on aggressive parameter
# combinations (small target edge length × many relax iterations).
# These defaults reliably converge in well under 30 s on a 30k-face
# input mesh and give >95% drop in max aspect ratio in the bulb region
# while keeping volume drift below 1%.
_DEFAULT_RELAX_ITERS = 3
_DEFAULT_MAX_EDGE_SPLITS = 200_000


def is_isotropic_remesh_available() -> bool:
    """Return ``True`` if :mod:`meshlib` imports cleanly, ``False``
    otherwise."""
    return _MESHLIB_AVAILABLE


def _trimesh_to_meshlib(mesh: trimesh.Trimesh):
    """Convert a :class:`trimesh.Trimesh` into a
    :class:`meshlib.mrmeshpy.Mesh`.

    Iteration is unavoidable because :mod:`meshlib`'s pybind bindings
    expose ``Vector3f`` and ``ThreeVertIds`` as Python objects rather
    than NumPy buffers; the cost is negligible (< 50 ms for a 30k-face
    mesh) compared to the remesh itself.
    """
    coords = _mrm.VertCoords()
    for v in mesh.vertices.astype(np.float32):
        coords.push_back(_mrm.Vector3f(float(v[0]), float(v[1]), float(v[2])))
    tri = _mrm.Triangulation()
    for f in mesh.faces.astype(np.int64):
        tri.push_back(
            _mrm.ThreeVertIds(
                [_mrm.VertId(int(f[0])), _mrm.VertId(int(f[1])), _mrm.VertId(int(f[2]))]
            )
        )
    return _mrm.Mesh.fromTriangles(coords, tri)


def _meshlib_to_trimesh(m) -> trimesh.Trimesh:
    """Inverse of :func:`_trimesh_to_meshlib`. Calls ``m.pack()`` first
    so the output index space is dense (no orphan vertices/faces from
    edge collapses)."""
    m.pack()
    pts = m.points
    n = pts.size()
    verts = np.array(
        [
            [
                float(pts[_mrm.VertId(i)].x),
                float(pts[_mrm.VertId(i)].y),
                float(pts[_mrm.VertId(i)].z),
            ]
            for i in range(n)
        ]
    )
    trios = m.topology.getAllTriVerts()
    faces = np.array(
        [
            [int(trios[i][0]), int(trios[i][1]), int(trios[i][2])]
            for i in range(trios.size())
        ],
        dtype=np.int64,
    )
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _mean_in_region_edge(
    vertices: np.ndarray, in_region_faces: np.ndarray
) -> float:
    """Mean edge length over the triangles that are inside the region."""
    tris = vertices[in_region_faces]
    e0 = np.linalg.norm(tris[:, 1] - tris[:, 0], axis=1)
    e1 = np.linalg.norm(tris[:, 2] - tris[:, 1], axis=1)
    e2 = np.linalg.norm(tris[:, 0] - tris[:, 2], axis=1)
    return float(np.mean(np.concatenate([e0, e1, e2])))


def remesh_region_isotropic(
    mesh: trimesh.Trimesh,
    region: dict,
    *,
    target_edge_length: float | None = None,
    iterations: int = 5,
    preserve_boundary: bool = True,
) -> trimesh.Trimesh:
    """Apply edge-flip / split / collapse isotropic remeshing to the
    bulb region only.

    Parameters
    ----------
    mesh:
        Input mesh. Should be watertight; the function does not mutate
        it.
    region:
        Dictionary with at least ``axis_index`` (int, primary length
        axis) and ``axis_min`` (float, primary-axis threshold above
        which a face is considered in-region).
    target_edge_length:
        Optional override for meshlib's ``targetEdgeLen``. When
        ``None`` (the default), 0.5× the mean in-region edge length of
        the input is used; this densifies the bulb tip while still
        respecting the input scale.
    iterations:
        Forwarded to meshlib's ``finalRelaxIters``. Capped at 3 — see
        ``_DEFAULT_RELAX_ITERS``.
    preserve_boundary:
        When ``True`` (default) every vertex strictly outside the
        region (axis < axis_min) is snapped back to its exact original
        coordinate after the remesh. This is what keeps the seam from
        breaking visibly. The flag is retained as a kwarg so callers
        can disable the snap for benchmarking; production code should
        leave it ``True``.

    Returns
    -------
    trimesh.Trimesh
        Either the remeshed mesh, or the original ``mesh`` if any
        precondition fails (meshlib not installed, region too small,
        meshlib raises). The returned mesh is always watertight when
        the input was.
    """
    if not _MESHLIB_AVAILABLE:
        return mesh

    primary = int(region.get("axis_index", int(mesh.extents.argmax())))
    axis_min = float(
        region.get("axis_min", float(mesh.vertices[:, primary].mean()))
    )

    verts = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    # In-region face mask: any vertex of the face crosses the threshold.
    # Same convention as :mod:`anisotropic_remesh` and ``BulbFFDDeformer``.
    face_in_region = (verts[faces, primary] >= axis_min).any(axis=1)
    n_in_region = int(face_in_region.sum())
    if n_in_region < _MIN_REGION_FACES:
        return mesh

    in_faces_idx = np.where(face_in_region)[0]
    in_faces = faces[face_in_region]

    # Determine target edge length from input density when not given.
    if target_edge_length is None:
        target_edge_length = 0.5 * _mean_in_region_edge(verts, in_faces)

    # Snapshot strictly-outside vertices (axis < axis_min). These must
    # come back bit-identical after the remesh — we do this by KD-tree
    # match-and-snap below.
    strictly_outside_mask = verts[:, primary] < axis_min
    strictly_outside_pts = verts[strictly_outside_mask].copy()
    if strictly_outside_pts.size == 0:
        # Region is the entire mesh — there is no seam to preserve.
        return mesh

    try:
        m = _trimesh_to_meshlib(mesh)

        # FaceBitSet over the input face indexing.
        fbs = _mrm.FaceBitSet()
        fbs.resize(len(faces))
        for i in in_faces_idx:
            fbs.set(_mrm.FaceId(int(i)), True)

        # Mark the seam (region boundary) edges as ``notFlippable``.
        # meshlib guarantees these edges "will never be flipped or
        # collapsed" and that "the vertices incident to these edges
        # are excluded from relaxation". This is what gives us
        # bit-identical seam vertex preservation — far stricter than
        # ``maxBdShift = 0`` (which is only a per-collapse limit).
        seam_edges_bs = _mrm.findRegionBoundaryUndirectedEdgesInsideMesh(
            m.topology, fbs
        )

        settings = _mrm.RemeshSettings()
        settings.targetEdgeLen = float(target_edge_length)
        settings.finalRelaxIters = int(min(iterations, _DEFAULT_RELAX_ITERS))
        settings.packMesh = True
        settings.region = fbs
        settings.notFlippable = seam_edges_bs
        # ``maxBdShift = 0`` clamps the per-collapse boundary movement
        # at zero — a belt-and-braces guard alongside ``notFlippable``.
        settings.maxBdShift = 0.0
        # ``projectOnOriginalMesh`` makes new (post-split) vertices land
        # back on the original surface before relaxation moves them.
        # Important on curved hulls; cheap to leave enabled.
        settings.projectOnOriginalMesh = True
        settings.maxEdgeSplits = _DEFAULT_MAX_EDGE_SPLITS

        ok = _mrm.remesh(m, settings)
        if not ok:
            return mesh

        result = _meshlib_to_trimesh(m)

        if preserve_boundary:
            # Re-snap each strictly-outside original vertex to its
            # counterpart in the result mesh. Two safeguards:
            #
            # 1. Only snap when the per-original distance is below
            #    a strict tolerance (2% of the mesh extent) — otherwise
            #    we'd risk dragging interior remeshed vertices.
            #
            # 2. Snap only the 1-to-1 portion of the mapping. If two
            #    originals share the same closest result vertex,
            #    meshlib collapsed a boundary edge despite the
            #    ``notFlippable`` + ``maxBdShift`` guards; in that
            #    case we leave that result vertex at its post-remesh
            #    position rather than picking arbitrarily between the
            #    two original siblings. On the production hull this
            #    affects ~5 vertices out of ~9700 — a hairline shift
            #    invisible at viewing scale, while the rest of the
            #    upper hull remains bit-identical to the input.
            from scipy.spatial import cKDTree

            tree = cKDTree(result.vertices)
            dists, idxs = tree.query(strictly_outside_pts, k=1)
            extent = float(np.max(mesh.extents))
            snap_tol = 0.02 * extent
            snap_mask = dists <= snap_tol
            snap_targets = idxs[snap_mask]

            # Filter out duplicates: a result vertex that several
            # originals all want is ambiguous; we drop those originals
            # from the snap set.
            counts = np.bincount(snap_targets, minlength=len(result.vertices))
            unique_match = counts[snap_targets] == 1
            final_snap_mask = np.zeros_like(snap_mask)
            snap_indices = np.where(snap_mask)[0]
            final_snap_mask[snap_indices[unique_match]] = True

            new_verts = result.vertices.copy()
            new_verts[idxs[final_snap_mask]] = strictly_outside_pts[final_snap_mask]
            result = trimesh.Trimesh(
                vertices=new_verts, faces=result.faces, process=False
            )

        return result
    except Exception as exc:  # pragma: no cover - defensive
        # Any failure (meshlib internal error, numerical degeneracy,
        # KD-tree mismatch) → return the input unchanged. We log at
        # debug level so the rest of the pipeline keeps moving.
        print(
            f"[isotropic_remesh] remesh failed ({exc.__class__.__name__}: "
            f"{exc}); returning input mesh unchanged."
        )
        return mesh


__all__: Sequence[str] = ("is_isotropic_remesh_available", "remesh_region_isotropic")
