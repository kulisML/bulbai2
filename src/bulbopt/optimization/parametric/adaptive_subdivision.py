"""Adaptive subdivision of the bulb region before FFD.

Design reference: ``docs/superpowers/specs/2026-04-23-bulbopt-mesh-quality-design.md`` §4 L5.

Problem
-------

FFD moves vertices; it does not create them. If the baseline bulb region
has too few triangles (a common case when the STL author pre-decimated
the bow), every deformed candidate looks polygonal. Global subdivision
over-meshes the hull midsection, which slows both the CFD solver and
the downstream post-smoothing pass.

Solution
--------

:func:`subdivide_region` counts the triangles inside the FFD region and,
if they are below ``min_triangles`` (default 500), performs targeted
subdivision:

1. Select the region faces by ``region["axis_index"]`` + ``region["axis_min"]``.
2. Call :func:`trimesh.remesh.subdivide` on exactly those faces.
3. Repair the T-junctions that appear on the boundary between
   subdivided and unsubdivided regions by splitting each neighbour
   triangle at the midpoint introduced by step 2 (watertightness is
   required for the OpenFOAM surfaceFeatureEdges pass).
4. Repeat up to ``max_iterations`` times while the region face count
   is still below the threshold.

The function is idempotent once the region has enough triangles: coarse
baselines get densified, already-dense baselines are returned unchanged
(minus a copy on the way out).
"""
from __future__ import annotations

from typing import Dict, Set, Tuple

import numpy as np
import trimesh
from trimesh.remesh import subdivide as _subdivide


def subdivide_region(
    mesh: trimesh.Trimesh,
    region: Dict,
    min_triangles: int = 500,
    max_iterations: int = 3,
) -> trimesh.Trimesh:
    """Return ``mesh`` with the bulb region densified to ``min_triangles`` tris.

    The returned mesh is a fresh :class:`trimesh.Trimesh`, always; the
    input is never mutated. When the region already meets the threshold
    the function returns ``mesh.copy()`` unchanged.

    Parameters
    ----------
    mesh:
        Baseline hull mesh. Must be watertight — the T-junction repair
        assumes adjacent triangles share exactly one edge.
    region:
        Same dict the deformer consumes (``axis_index`` and ``axis_min``
        identify the bulb faces by triangle-centroid selection).
    min_triangles:
        Stop iterating once the region's face count reaches this value.
    max_iterations:
        Hard cap on subdivision passes; prevents runaway on very coarse
        baselines.
    """
    if min_triangles <= 0:
        raise ValueError("min_triangles must be > 0")
    if max_iterations < 0:
        raise ValueError("max_iterations must be >= 0")

    current = mesh.copy()
    for _ in range(int(max_iterations)):
        region_mask = _region_face_mask(current, region)
        if int(region_mask.sum()) >= int(min_triangles):
            return current
        if not region_mask.any():
            return current
        current = _subdivide_with_tjunction_repair(current, region_mask)

    return current


# ---- helpers --------------------------------------------------------------


def _region_face_mask(mesh: trimesh.Trimesh, region: Dict) -> np.ndarray:
    """Return a boolean mask picking faces whose centroid is aft of the
    bulb axis_min threshold.

    The deformer itself uses a vertex test (``axis >= axis_min``); for
    subdivision we use the triangle centroid so we never drag a
    unsubdivided triangle's edge into the subdivided region.
    """
    primary_axis = int(region.get("axis_index", int(mesh.extents.argmax())))
    axis_min = float(region.get("axis_min", mesh.vertices[:, primary_axis].mean()))
    centroids = np.asarray(mesh.triangles_center, dtype=float)
    return centroids[:, primary_axis] >= axis_min


def _subdivide_with_tjunction_repair(
    mesh: trimesh.Trimesh,
    region_mask: np.ndarray,
) -> trimesh.Trimesh:
    """Subdivide region faces and split each neighbour face across the
    new midpoint so the mesh stays watertight.
    """
    vertices = np.asarray(mesh.vertices, dtype=float).copy()
    faces = np.asarray(mesh.faces, dtype=np.int64).copy()
    n_original_vertices = len(vertices)

    region_indices = np.where(region_mask)[0]
    region_faces = faces[region_mask]
    non_region_faces = faces[~region_mask]

    if len(region_indices) == 0:
        return mesh.copy()

    new_vertices, new_faces = _subdivide(
        vertices=vertices, faces=faces, face_index=region_indices
    )
    new_vertices = np.asarray(new_vertices, dtype=float)
    new_faces = np.asarray(new_faces, dtype=np.int64)
    n_non_region = len(non_region_faces)
    subdivided_region_faces = new_faces[n_non_region:]

    # Bug #7 (audit 2026-04-26): the midpoint-dedup tolerance must scale
    # with the mesh's edge length, not be a fixed 1e-18 squared distance.
    # On a 100 m hull trimesh's internal subdivide can place the midpoint
    # ~1e-7 m away from the analytic ``0.5 * (va + vb)`` due to float
    # accumulation; a fixed 1e-9 m tolerance rejects valid midpoints,
    # leaves T-junctions, and breaks watertightness. Use a relative
    # tolerance derived from the region's mean edge length so the dedup
    # works robustly on 1 m boats through 300 m container ships.
    region_edge_lengths = np.linalg.norm(
        vertices[region_faces[:, [1, 2, 0]]] - vertices[region_faces],
        axis=2,
    )
    mean_edge_len = float(region_edge_lengths.mean()) if region_edge_lengths.size else 0.0
    # 1e-6 of the edge length is far tighter than any legitimate spatial
    # separation between distinct midpoints (their nearest possible pair
    # is ~half an edge apart) but loose enough to absorb float ULP noise
    # at any hull scale. Floor with 1e-18 to preserve the historical
    # behaviour on degenerate or fully-collapsed inputs.
    tol_squared = max((1e-6 * mean_edge_len) ** 2, 1e-18)

    # Map each unique region edge → its new midpoint vertex index.
    edge_to_midpoint: Dict[Tuple[int, int], int] = {}
    for face in region_faces:
        for i in range(3):
            a = int(face[i])
            b = int(face[(i + 1) % 3])
            key = (a, b) if a <= b else (b, a)
            if key in edge_to_midpoint:
                continue
            mid_point = 0.5 * (vertices[a] + vertices[b])
            # Newly added vertices live in new_vertices[n_original_vertices:];
            # vectorised search is cheaper than an np.allclose loop.
            tail = new_vertices[n_original_vertices:]
            if len(tail) == 0:
                continue
            diffs = tail - mid_point
            dists = np.einsum("ij,ij->i", diffs, diffs)
            min_idx = int(np.argmin(dists))
            if dists[min_idx] <= tol_squared:
                edge_to_midpoint[key] = n_original_vertices + min_idx

    # Boundary edges: present in both region and non-region faces.
    region_edge_set: Set[Tuple[int, int]] = set()
    for face in region_faces:
        for i in range(3):
            a = int(face[i])
            b = int(face[(i + 1) % 3])
            region_edge_set.add((a, b) if a <= b else (b, a))

    boundary_edges: Set[Tuple[int, int]] = set()
    for face in non_region_faces:
        for i in range(3):
            a = int(face[i])
            b = int(face[(i + 1) % 3])
            key = (a, b) if a <= b else (b, a)
            if key in region_edge_set:
                boundary_edges.add(key)

    # For each non-region face, split along every edge that is also a
    # boundary edge. A face may have 1, 2, or 3 boundary edges (corner
    # triangles at the region perimeter can have two; a fully surrounded
    # face has three) — we handle all three cases so no T-junction
    # survives. Winding is preserved by always listing the midpoint
    # between the two original vertex indices.
    patched_faces = []
    for face in non_region_faces:
        ab = int(face[0])
        bc = int(face[1])
        ca = int(face[2])
        # Per-edge midpoint lookups (None if the edge isn't boundary).
        m_ab = _lookup_midpoint(ab, bc, boundary_edges, edge_to_midpoint)
        m_bc = _lookup_midpoint(bc, ca, boundary_edges, edge_to_midpoint)
        m_ca = _lookup_midpoint(ca, ab, boundary_edges, edge_to_midpoint)
        mids = [m_ab, m_bc, m_ca]
        n_mids = sum(1 for m in mids if m is not None)
        if n_mids == 0:
            patched_faces.append([ab, bc, ca])
        elif n_mids == 3:
            # 4-triangle split matching the subdivided-region pattern.
            patched_faces.append([ab, m_ab, m_ca])
            patched_faces.append([m_ab, bc, m_bc])
            patched_faces.append([m_ca, m_bc, ca])
            patched_faces.append([m_ab, m_bc, m_ca])
        elif n_mids == 2:
            # Two boundary edges share a vertex — split into 3 triangles.
            if m_ab is not None and m_bc is not None:
                # Shared vertex is bc.
                patched_faces.append([ab, m_ab, ca])
                patched_faces.append([m_ab, bc, m_bc])
                patched_faces.append([m_ab, m_bc, ca])
            elif m_bc is not None and m_ca is not None:
                patched_faces.append([bc, m_bc, ab])
                patched_faces.append([m_bc, ca, m_ca])
                patched_faces.append([m_bc, m_ca, ab])
            else:  # m_ab is not None and m_ca is not None
                patched_faces.append([ca, m_ca, bc])
                patched_faces.append([m_ca, ab, m_ab])
                patched_faces.append([m_ca, m_ab, bc])
        else:  # n_mids == 1
            if m_ab is not None:
                patched_faces.append([ab, m_ab, ca])
                patched_faces.append([m_ab, bc, ca])
            elif m_bc is not None:
                patched_faces.append([bc, m_bc, ab])
                patched_faces.append([m_bc, ca, ab])
            else:  # m_ca is not None
                patched_faces.append([ca, m_ca, bc])
                patched_faces.append([m_ca, ab, bc])

    final_faces = np.asarray(
        patched_faces + subdivided_region_faces.tolist(),
        dtype=np.int64,
    )

    return trimesh.Trimesh(vertices=new_vertices, faces=final_faces, process=False)


def _lookup_midpoint(
    a: int,
    b: int,
    boundary_edges: Set[Tuple[int, int]],
    edge_to_midpoint: Dict[Tuple[int, int], int],
) -> int | None:
    """Return the midpoint vertex index for edge ``(a, b)`` or ``None``.

    Edges are stored in ``boundary_edges`` and ``edge_to_midpoint`` with
    the canonical form ``(min(a, b), max(a, b))``; this helper keeps
    callers from having to remember that convention.
    """
    key = (a, b) if a <= b else (b, a)
    if key in boundary_edges and key in edge_to_midpoint:
        return int(edge_to_midpoint[key])
    return None


__all__ = ["subdivide_region"]
