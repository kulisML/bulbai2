"""Parametric Wigley hull mesh fixture (audit C, item Add #7).

A Wigley hull is the canonical analytical reference shape used by the ITTC
naval-hydrodynamics community for benchmarking. Its surface is given in
closed form by

    y(x, z) = (B/2) * (1 - (2x/L)**2) * (1 - (z/T)**2)

with ``L`` the waterline length, ``B`` the beam, ``T`` the draft,
``x in [-L/2, +L/2]`` and ``z in [-T, 0]``. The hull surface is the locus
``±y(x, z)`` — mirror-symmetric about the y=0 plane — which gives us
asymmetric draft (deck above, keel below), a sharp keel edge at z=-T,
and pointed bow/transom edges at x=±L/2. This makes it the smallest
realistic test shape that exercises the FFD/quality stack on properties
real ship hulls have but the unit-test corpus does not.

Fixture contract
----------------

* :func:`make_wigley_hull` returns a ``trimesh.Trimesh`` with
  ``is_watertight`` True (verified by the integration tests).
* Surface vertices for both the +Y and -Y half-hulls share boundary
  vertices on the keel (z=-T, y=0), bow (x=+L/2, y=0), and transom
  (x=-L/2, y=0) — so closing the deck cap at z=0 produces a closed
  manifold mesh with no internal seams.
* The deck cap fills the rectangle at z=0 between the +Y and -Y
  waterline traces, oriented with outward normal +Z.
* The builder is fully deterministic — no random state, no thread-local
  caches.

Default mesh size
-----------------

``n_x=40, n_z=20`` produces ~1500 vertices and ~3000 faces — small enough
to build in <0.4 s on a laptop, large enough to exercise the FFD's
adaptive subdivision and Taubin smoothing on a real-shape topology.
"""
from __future__ import annotations

import numpy as np
import trimesh
import trimesh.repair


def make_wigley_hull(
    L: float = 100.0,
    B: float = 10.0,
    T: float = 6.25,
    n_x: int = 40,
    n_z: int = 20,
) -> trimesh.Trimesh:
    """Build a closed Wigley hull mesh with deck cap.

    Parameters
    ----------
    L:
        Waterline length (positive, in metres).
    B:
        Maximum beam (positive, in metres).
    T:
        Draft (positive, in metres). Hull extends from z=-T (keel) to
        z=0 (waterline / deck).
    n_x:
        Number of stations along the longitudinal axis. Must be ≥ 3.
    n_z:
        Number of waterline traces from keel to deck. Must be ≥ 3.

    Returns
    -------
    trimesh.Trimesh
        A watertight, winding-consistent closed mesh. ``mesh.extents``
        is approximately ``(L, B, T)`` — beam may fall slightly short
        of B because the discrete grid doesn't sample the maximum-
        beam waterline exactly when n_x or n_z is small.

    Notes
    -----
    Vertices on the boundary (keel, bow, transom) where ``y = 0`` are
    shared between the +Y and -Y half-surfaces. This keeps the welded-
    graph adjacency proper across the centreline (relevant for Taubin
    smoothing inside the FFD deformer) and makes the deck cap a single
    quad per longitudinal cell.
    """
    if n_x < 3 or n_z < 3:
        raise ValueError("n_x and n_z must each be at least 3")
    if L <= 0 or B <= 0 or T <= 0:
        raise ValueError("L, B, T must all be positive")

    # ---- Sample the surface grid -----------------------------------------
    xs = np.linspace(-L / 2.0, +L / 2.0, n_x)
    zs = np.linspace(-T, 0.0, n_z)
    X, Z = np.meshgrid(xs, zs, indexing="ij")  # shape (n_x, n_z)
    # Wigley half-beam: zero at all four boundaries (x=±L/2 and z=-T).
    Y_half = (B / 2.0) * (1.0 - (2.0 * X / L) ** 2) * (1.0 - (Z / T) ** 2)
    # Numerical noise can make boundary samples slightly negative.
    Y_half = np.clip(Y_half, 0.0, None)

    EPS = 1e-9

    # ---- Vertex bookkeeping ----------------------------------------------
    #
    # ``plus_idx[i, j]`` and ``minus_idx[i, j]`` are the vertex indices for
    # the +Y and -Y surface samples at grid cell (i, j). On the boundary
    # (Y_half[i, j] < EPS) both indices alias the *same* vertex on the
    # centreline, so closing the keel/bow/transom seam doesn't require any
    # extra triangles. ``is_shared`` marks those positions for later
    # degenerate-triangle filtering.
    plus_idx = np.full((n_x, n_z), -1, dtype=np.int64)
    minus_idx = np.full((n_x, n_z), -1, dtype=np.int64)
    is_shared = np.zeros((n_x, n_z), dtype=bool)
    verts: list[tuple[float, float, float]] = []

    for i in range(n_x):
        for j in range(n_z):
            y = float(Y_half[i, j])
            x = float(xs[i])
            z = float(zs[j])
            if y < EPS:
                idx = len(verts)
                verts.append((x, 0.0, z))
                plus_idx[i, j] = idx
                minus_idx[i, j] = idx
                is_shared[i, j] = True
            else:
                idx_p = len(verts)
                verts.append((x, +y, z))
                plus_idx[i, j] = idx_p
                idx_m = len(verts)
                verts.append((x, -y, z))
                minus_idx[i, j] = idx_m

    # ---- Triangulate the two surface grids -------------------------------
    faces: list[tuple[int, int, int]] = []

    def _add_tri(
        side_idx: np.ndarray,
        i0: int,
        j0: int,
        i1: int,
        j1: int,
        i2: int,
        j2: int,
        reverse: bool = False,
    ) -> None:
        """Append one triangle from ``side_idx`` skipping degenerates.

        A triangle whose three corners are *all* on the boundary
        (``is_shared``) would be duplicated by the opposite-side surface
        — both the +Y and -Y grids would emit the same tri because their
        indices alias on shared boundary vertices. Skipping such tris
        keeps the mesh manifold (otherwise we'd get one edge appearing
        in 4 faces, which fails ``is_watertight`` even though the mesh
        is closed).
        """
        if is_shared[i0, j0] and is_shared[i1, j1] and is_shared[i2, j2]:
            return
        v0 = int(side_idx[i0, j0])
        v1 = int(side_idx[i1, j1])
        v2 = int(side_idx[i2, j2])
        if v0 == v1 or v0 == v2 or v1 == v2:
            return
        if reverse:
            faces.append((v0, v2, v1))
        else:
            faces.append((v0, v1, v2))

    # +Y surface: outward normal +Y. CCW order viewed from +Y is
    # (i, j) → (i, j+1) → (i+1, j+1) → (i+1, j).
    for i in range(n_x - 1):
        for j in range(n_z - 1):
            _add_tri(plus_idx, i, j, i, j + 1, i + 1, j + 1)
            _add_tri(plus_idx, i, j, i + 1, j + 1, i + 1, j)

    # -Y surface: outward normal -Y. Reverse winding order so the right-
    # hand rule yields a -Y normal.
    for i in range(n_x - 1):
        for j in range(n_z - 1):
            _add_tri(minus_idx, i, j, i + 1, j + 1, i, j + 1)
            _add_tri(minus_idx, i, j, i + 1, j, i + 1, j + 1)

    # Deck cap at z = 0 (j = n_z - 1). For each cell along x, build a
    # quad between the +Y and -Y waterline traces. Outward normal +Z.
    j_top = n_z - 1
    for i in range(n_x - 1):
        p0 = int(plus_idx[i, j_top])
        p1 = int(plus_idx[i + 1, j_top])
        m0 = int(minus_idx[i, j_top])
        m1 = int(minus_idx[i + 1, j_top])
        # Quad vertices in CCW order from +Z: (p0, p1, m1, m0). Skip
        # degenerate halves at the bow/transom where p0==m0 or p1==m1.
        if not (p0 == p1 or p0 == m1 or p1 == m1):
            faces.append((p0, p1, m1))
        if not (p0 == m1 or p0 == m0 or m1 == m0):
            faces.append((p0, m1, m0))

    vertices = np.asarray(verts, dtype=float)
    faces_arr = np.asarray(faces, dtype=np.int64)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces_arr, process=False)
    # The triangulation above is consistent under the right-hand rule per
    # surface, but the deck cap and -Y surface windings rely on
    # documentation rather than numeric verification. ``fix_winding`` flips
    # any face whose normal disagrees with the majority of its neighbours
    # so the whole mesh ends up consistently outward-pointing, and
    # ``fix_normals`` finalises by computing per-face normals from
    # vertex order.
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)
    return mesh
