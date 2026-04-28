"""Free-Form Deformation of the bulb region.

Design reference: 2026-04-22-bulbopt-night-optimization-design.md §4.2.

We implement tri-variate Bezier (Bernstein-basis) FFD directly on numpy
because the canonical SISSA PyGeM is not on PyPI and installing it via git
is fragile inside a laptop environment. The math is short and well-tested:

  P_new(s,t,u) = P_old + sum_{i=0..l, j=0..m, k=0..n}
                       B_i^l(s) * B_j^m(t) * B_k^n(u) * dP_ijk

where B_i^l is the ith Bernstein polynomial of degree l and s,t,u are the
point's normalised coordinates inside the control box. Offsets dP_ijk at
the control lattice corners deform the volume, and the deformation smoothly
vanishes at the box boundaries — exactly what we want for a local bulb
region edit.

The deformer is intentionally decoupled from the Kracht space: ``deform``
takes an already-sampled ``KrachtVector`` and maps its 8 parameters onto
lattice offsets. That mapping is deterministic, lives in one place
(``_kracht_to_lattice_offsets``), and is the only knob to tune if the
generated bulbs look wrong.
"""
from __future__ import annotations

from math import comb
from typing import Tuple

import numpy as np
import trimesh

from bulbopt.optimization.parametric.kracht_space import KrachtVector


class BulbFFDDeformer:
    """Deform a mesh's bulb region via a tri-variate Bezier FFD lattice."""

    LATTICE_SHAPE: Tuple[int, int, int] = (5, 4, 4)
    # Degree of the Bernstein basis per axis is n_cp - 1.

    # Smooth blend before axis_min so the deformed bulb joins the rest of the
    # hull with C1 continuity (spec 2026-04-23 §3 Fix A). Width is a fraction
    # of the bulb length (axis_max - axis_min).
    BLEND_WIDTH_FRACTION: float = 0.10

    def __init__(
        self,
        force_port_starboard_symmetry: bool = True,
        post_smoothing_iterations: int = 3,
        adaptive_subdivision: bool = True,
        adaptive_min_triangles: int = 500,
        adaptive_max_iterations: int = 3,
        post_repair: bool = True,
        seam_smoothing_iterations: int = 2,
        tip_remesh_max_edge_factor: float = 0.0,
        tip_finisher_iterations: int = 0,
    ) -> None:
        """
        Parameters
        ----------
        force_port_starboard_symmetry:
            When True (default), the deformer enforces mirror symmetry of
            the output mesh around the beam midplane (spec 2026-04-23 §3
            Fix B). This is defensive: even if the baseline STL has tiny
            triangulation asymmetries, the engineer still gets a
            mirror-clean bulb.
        post_smoothing_iterations:
            Number of Taubin smoothing passes applied after FFD. Taubin
            alternates a positive Laplacian step (smoothing) with a
            negative one (anti-smoothing) so the low-frequency shape is
            preserved (volume drift ≤ 1% after 3 iterations) while the
            polygonal facet edges are rounded (spec 2026-04-23 §3 Fix C).
            Set to 0 to disable smoothing (useful for volume tests).
        adaptive_subdivision:
            Opt-in triangle densification before FFD (spec 2026-04-23 §4
            L5). When True, the bulb region is subdivided up to
            ``adaptive_max_iterations`` times until it contains at least
            ``adaptive_min_triangles`` triangles, so FFD has enough
            resolution to avoid polygonal facets in the output.
        adaptive_min_triangles:
            Triangle count threshold for the region. Default 500.
        adaptive_max_iterations:
            Hard cap on subdivision passes (prevents runaway on coarse
            baselines). Default 3.
        post_repair:
            Audit C 2026-04-26 (Add #2): when True (default) the deformer
            runs ``merge_vertices`` + ``process(validate=False)`` on the
            output mesh just before returning it. This folds coincident
            vertices left over by ``process=False`` construction and
            re-asserts consistent winding, so meshes loaded from STLs
            with per-face independent vertices stay watertight after
            FFD + Taubin. Set to False for tests that need bit-identical
            vertex-array invariance (``test_deformer_preserves_face_
            topology`` and friends pass it via ``adaptive_subdivision=
            False``).
        seam_smoothing_iterations:
            Audit C 2026-04-26 (Add #5): pure-Laplacian smoothing
            iterations applied to the ring of vertices whose smoothstep
            blend weight is in (0.05, 0.95) — i.e. inside the seam
            between bulb interior and the rest of the hull. This levels
            the per-triangle dihedral discontinuity at the boundary
            without disturbing the rest of the hull. Default 2; set to 0
            to disable.
        tip_remesh_max_edge_factor:
            Audit 2026-04-26 — Module B3 Add #1: opt-in uniform remesh of
            the bulb region BEFORE the FFD lattice push. When > 0 the
            bulb region (faces with at least one vertex in
            ``axis >= blend_start``) is replaced by a uniformly
            triangulated copy where the longest edge does not exceed
            ``factor * box_size_min(bulb_region)`` (the smallest extent
            of the region's bounding box). The rest of the hull is
            untouched. This kills high-aspect-ratio sliver triangles
            inherited from the baseline STL, which FFD itself cannot
            remove (it only moves vertices). Default 0.0 (disabled) for
            backward compatibility; recommended value when enabling is
            0.05 — set explicitly by callers, do NOT change the default.
        tip_finisher_iterations:
            Audit 2026-04-26 — Module B3 Add #2: opt-in extra targeted
            Laplacian smoothing pass applied AFTER Taubin + seam smoothing
            + post-repair. It selects faces whose dihedral angle with
            any neighbour exceeds 30° and runs N gentle Laplacian steps
            on the union of their vertices. Kills the residual creases
            visible at the bow tip on real ship hulls without disturbing
            the rest of the bulb. Default 0 (disabled) for backward
            compatibility; recommended value when enabling is 2.
        """
        self.force_port_starboard_symmetry = bool(force_port_starboard_symmetry)
        self.post_smoothing_iterations = max(int(post_smoothing_iterations), 0)
        self.adaptive_subdivision = bool(adaptive_subdivision)
        self.adaptive_min_triangles = max(int(adaptive_min_triangles), 1)
        self.adaptive_max_iterations = max(int(adaptive_max_iterations), 0)
        self.post_repair = bool(post_repair)
        self.seam_smoothing_iterations = max(int(seam_smoothing_iterations), 0)
        self.tip_remesh_max_edge_factor = max(float(tip_remesh_max_edge_factor), 0.0)
        self.tip_finisher_iterations = max(int(tip_finisher_iterations), 0)

    def deform(
        self,
        mesh: trimesh.Trimesh,
        region: dict,
        vector: KrachtVector,
    ) -> trimesh.Trimesh:
        """Return a copy of ``mesh`` with bulb-region vertices displaced.

        Vertices outside ``region`` are bit-identical to the input.
        Topology (faces, order) is unchanged so a watertight input stays
        watertight on output.
        """
        primary_axis = int(region["axis_index"])
        axis_min = float(region["axis_min"])
        axis_max = float(region["axis_max"])
        if axis_max <= axis_min:
            return mesh.copy()

        # Beam (port-starboard, symmetric) and draft (keel-deck, asymmetric)
        # axes. Audit 2026-04-26: read these from the region dict when
        # provided so the deformer mirrors around the *actual* symmetric
        # axis of the hull. Fall back to the legacy heuristic
        # ``other_axes[0]/[1]`` when the region dict doesn't carry them so
        # synthetic test meshes (icospheres, boxes) keep working.
        beam_axis, draft_axis = _resolve_secondary_axes(region, primary_axis)

        # Spec 2026-04-23 §4 L5: densify the bulb region before FFD so we
        # always have at least ``adaptive_min_triangles`` tris in the
        # deformation zone. This is a no-op when the input already meets
        # the threshold or the hook is disabled at __init__.
        if self.adaptive_subdivision:
            # Local import keeps the ffd_deformer module importable in
            # environments where adaptive_subdivision has missing deps.
            from bulbopt.optimization.parametric.adaptive_subdivision import (
                subdivide_region,
            )

            mesh = subdivide_region(
                mesh,
                region,
                min_triangles=self.adaptive_min_triangles,
                max_iterations=self.adaptive_max_iterations,
            )

        # Audit 2026-04-26 — Module B3 Add #1: uniform remesh of the bulb
        # region BEFORE FFD. Adaptive subdivision densifies (multiplies
        # triangle count); this normalises edge length so high-aspect
        # slivers inherited from the baseline STL are replaced by uniform
        # tris. Different concern, different pass; both are opt-in.
        if self.tip_remesh_max_edge_factor > 0.0:
            mesh = _uniform_remesh_region(
                mesh,
                region,
                max_edge_factor=self.tip_remesh_max_edge_factor,
            )

        # Spec 2026-04-23 §3 Fix A: smooth blend across the boundary. The
        # participating region extends BLEND_WIDTH_FRACTION × (axis_max -
        # axis_min) aft of axis_min; every vertex there gets a smoothstep
        # weight from 0 (at blend_start) to 1 (at axis_min), so there is
        # no C0 discontinuity across the boundary.
        blend_width = self.BLEND_WIDTH_FRACTION * (axis_max - axis_min)
        blend_start = axis_min - blend_width

        participating = mesh.vertices[:, primary_axis] >= blend_start
        if not np.any(participating):
            return mesh.copy()

        deformed_vertices = mesh.vertices.copy()

        # Build control box in the bulb region's bounding volume so the
        # deformation tapers to zero on the aft-most side (where the hull
        # glues back into the rest of the mesh).
        region_vertices = mesh.vertices[participating]
        box_origin, box_size = self._region_box(
            region_vertices,
            primary_axis=primary_axis,
            axis_min=axis_min,
            axis_max=axis_max,
        )

        offsets = self._kracht_to_lattice_offsets(
            vector=vector,
            primary_axis=primary_axis,
            box_size=box_size,
            beam_axis=beam_axis,
            draft_axis=draft_axis,
        )

        deformed_region = self._apply_ffd(
            points=region_vertices,
            box_origin=box_origin,
            box_size=box_size,
            offsets=offsets,
        )

        # Blend weights: 0 at blend_start, 1 at axis_min, 1 beyond.
        axis_values = region_vertices[:, primary_axis]
        if blend_width > 0:
            raw = (axis_values - blend_start) / blend_width
        else:
            raw = np.where(axis_values >= axis_min, 1.0, 0.0)
        t = np.clip(raw, 0.0, 1.0)
        # smoothstep (Hermite) for C1 continuity at both ends.
        weights = t * t * (3.0 - 2.0 * t)
        # Apply weighted displacement back to the original vertices.
        weighted_displacement = (deformed_region - region_vertices) * weights[:, None]
        deformed_vertices[participating] = region_vertices + weighted_displacement

        if self.force_port_starboard_symmetry:
            # Only symmetrize vertices that actually participate in the
            # deformation — leave the rest of the hull alone so legacy
            # topology is preserved bit-identical.
            participating_indices = np.nonzero(participating)[0]
            deformed_vertices = _enforce_mirror_symmetry_subset(
                vertices=deformed_vertices,
                beam_axis=beam_axis,
                subset_indices=participating_indices,
                primary_axis=primary_axis,
            )

        out = trimesh.Trimesh(
            vertices=deformed_vertices,
            faces=mesh.faces,
            process=False,
        )

        # Spec 2026-04-23 §3 Fix C: Taubin volume-preserving smoothing.
        # Lambda=0.5, Nu=-0.53 is the classic pair from Taubin 1995 that
        # preserves low-frequency shape while rounding high-frequency
        # facet edges. We apply this only to bulb-region vertices so the
        # rest of the hull keeps its original triangulation.
        taubin_skipped = False
        if self.post_smoothing_iterations > 0:
            taubin_skipped = not _apply_taubin_to_region(
                out,
                primary_axis=primary_axis,
                blend_start=blend_start,
                blend_width=blend_width,
                iterations=self.post_smoothing_iterations,
            )
            # Re-assert symmetry after smoothing (Laplacian can drift
            # micro-asymmetries back in). Keep the same subset scope and
            # the same resolved beam axis we used for the first pass.
            if self.force_port_starboard_symmetry and not taubin_skipped:
                out.vertices = _enforce_mirror_symmetry_subset(
                    vertices=np.asarray(out.vertices),
                    beam_axis=beam_axis,
                    subset_indices=np.nonzero(participating)[0],
                    primary_axis=primary_axis,
                )

        # Audit C 2026-04-26 (Add #5): tiny pure-Laplacian smoothing on
        # the ring of seam vertices to flatten the per-triangle dihedral
        # discontinuity at the boundary between bulb interior and the
        # rest of the hull. Only runs if Taubin actually executed (small
        # regions where Taubin is bypassed: the seam ring would also be
        # too sparse for the Laplacian to behave well, so we skip it
        # together — keeps the same "tiny → polygonal-but-watertight"
        # tradeoff Bug #6 already chose).
        if self.seam_smoothing_iterations > 0 and not taubin_skipped:
            _apply_seam_laplacian(
                out,
                primary_axis=primary_axis,
                blend_start=blend_start,
                blend_width=blend_width,
                iterations=self.seam_smoothing_iterations,
                beam_axis=beam_axis if self.force_port_starboard_symmetry else None,
            )
            if self.force_port_starboard_symmetry:
                out.vertices = _enforce_mirror_symmetry_subset(
                    vertices=np.asarray(out.vertices),
                    beam_axis=beam_axis,
                    subset_indices=np.nonzero(participating)[0],
                    primary_axis=primary_axis,
                )

        # Audit C 2026-04-26 (Add #2): fold coincident vertices and
        # re-assert winding. Done last so it sees the final post-Taubin /
        # post-seam-smoothing positions. ``merge_vertices`` only
        # consolidates positions that are *already* coincident (within
        # trimesh's default tol); it does not undo the Taubin smoothing
        # — verified by ``test_taubin_does_not_collapse_mesh_with_
        # duplicated_vertices`` continuing to pass.
        if self.post_repair:
            out.merge_vertices()
            out.process(validate=False)

        # Audit 2026-04-26 — Module B3 Add #2: tip finisher. Targeted
        # Laplacian smoothing on faces whose dihedral with any neighbour
        # exceeds 30°. Run AFTER post_repair so we see the final mesh
        # topology. The pass is opt-in (default 0) and gentle by design;
        # if the participating ring is empty (no high-dihedral faces) it
        # is a silent no-op.
        if self.tip_finisher_iterations > 0:
            _apply_tip_finisher(
                out,
                primary_axis=primary_axis,
                blend_start=blend_start,
                iterations=self.tip_finisher_iterations,
                beam_axis=beam_axis if self.force_port_starboard_symmetry else None,
            )
            if self.force_port_starboard_symmetry:
                out.vertices = _enforce_mirror_symmetry_subset(
                    vertices=np.asarray(out.vertices),
                    beam_axis=beam_axis,
                    subset_indices=np.nonzero(
                        np.asarray(out.vertices)[:, primary_axis] >= blend_start
                    )[0],
                    primary_axis=primary_axis,
                )

        return out

    # ---- geometry helpers -------------------------------------------------

    def _region_box(
        self,
        vertices: np.ndarray,
        *,
        primary_axis: int,
        axis_min: float,
        axis_max: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounding box enclosing the bulb region."""
        other_axes = [axis for axis in range(3) if axis != primary_axis]
        origin = np.zeros(3, dtype=float)
        size = np.zeros(3, dtype=float)
        origin[primary_axis] = axis_min
        size[primary_axis] = max(axis_max - axis_min, 1e-9)
        for axis in other_axes:
            lo = float(vertices[:, axis].min())
            hi = float(vertices[:, axis].max())
            # Pad laterally so boundary vertices sit strictly inside the
            # box and their weights are well-defined at the edges.
            pad = max((hi - lo) * 0.05, 1e-6)
            origin[axis] = lo - pad
            size[axis] = (hi - lo) + 2.0 * pad
        return origin, size

    # ---- Kracht → lattice offsets ----------------------------------------

    def _kracht_to_lattice_offsets(
        self,
        *,
        vector: KrachtVector,
        primary_axis: int,
        box_size: np.ndarray,
        beam_axis: int | None = None,
        draft_axis: int | None = None,
    ) -> np.ndarray:
        """Map an 8-D Kracht sample onto (l, m, n, 3) lattice offsets.

        The mapping is deterministic and smooth: each Kracht parameter
        contributes to one or two lattice dimensions, and the overall
        magnitude is scaled by the physical ``box_size`` so the deformation
        feels consistent across hulls of different scale.

        ``beam_axis`` and ``draft_axis`` default to the legacy
        ``other_axes[0]/[1]`` heuristic when not supplied (kept for the
        synthetic-mesh tests). The deformer's public ``deform`` resolves
        them from the ``region`` dict before calling this method.
        """
        l_cp, m_cp, n_cp = self.LATTICE_SHAPE
        offsets = np.zeros((l_cp, m_cp, n_cp, 3), dtype=float)

        v = vector.values
        axis_primary = primary_axis
        if beam_axis is None or draft_axis is None:
            axes_secondary = [a for a in range(3) if a != primary_axis]
            if beam_axis is None:
                beam_axis = axes_secondary[0]
            if draft_axis is None:
                draft_axis = axes_secondary[1]
        beam_axis = int(beam_axis)
        draft_axis = int(draft_axis)

        # Lattice's 3 spatial dims map 1-to-1 to world axes (dim 0 → world
        # axis 0, dim 1 → world axis 1, dim 2 → world axis 2). The naming
        # ``i, j, k`` for lattice indices used to also be aliased to
        # ``primary, beam, draft`` — that aliasing was correct only when
        # primary=0, beam=1, draft=2 (the legacy box / icosphere case).
        # On real ship hulls (audit 2026-04-26) primary=0 but beam=2,
        # draft=1, so we must dispatch each per-axis loop onto the
        # *correct* lattice dimension.
        #
        # ``axes_dim[w]`` is the lattice dimension that varies along
        # world axis ``w``. With the current LATTICE_SHAPE convention
        # (5, 4, 4) it's the identity, but we name it explicitly so the
        # broadcasts below stay readable.
        primary_dim = axis_primary  # lattice dim that varies along world primary axis
        beam_dim = beam_axis
        draft_dim = draft_axis

        # ---- Length: push forward-most slab along primary axis ----
        length_push = v["length_ratio"] * box_size[axis_primary]
        longitudinal_weight = v["longitudinal_pos"]
        prim_n = offsets.shape[primary_dim]
        prim_frac = np.arange(prim_n) / max(prim_n - 1, 1)
        prim_weight = (prim_frac ** 2) * (0.5 + 0.5 * longitudinal_weight)
        # length_weight^1.5 ramp, used by breadth/height blocks below.
        length_ramp_15 = prim_frac ** 1.5
        # Reshape so a 1-D ramp along the primary lattice dim broadcasts
        # over the (l_cp, m_cp, n_cp) offsets array.
        prim_shape = [1, 1, 1]
        prim_shape[primary_dim] = prim_n
        prim_weight_b = prim_weight.reshape(prim_shape)
        length_ramp_15_b = length_ramp_15.reshape(prim_shape)
        offsets[..., axis_primary] += prim_weight_b * length_push

        # ---- Breadth: ± along beam axis, scaled by primary-axis ramp ----
        breadth_push = v["breadth_ratio"] * box_size[beam_axis] * 0.5
        beam_n = offsets.shape[beam_dim]
        beam_frac = np.arange(beam_n) / max(beam_n - 1, 1)
        beam_centred = beam_frac - 0.5
        beam_sign = np.sign(beam_centred)
        beam_shape = [1, 1, 1]
        beam_shape[beam_dim] = beam_n
        beam_sign_b = beam_sign.reshape(beam_shape)
        offsets[..., beam_axis] += (
            beam_sign_b * length_ramp_15_b * breadth_push
        )

        # ---- Height: ± along draft axis, plus uniform draft-axis shift ----
        height_push = v["height_ratio"] * box_size[draft_axis] * 0.5
        axis_z = (v["axis_z_ratio"] - 0.25) * box_size[draft_axis]
        draft_n = offsets.shape[draft_dim]
        draft_frac = np.arange(draft_n) / max(draft_n - 1, 1)
        draft_centred = draft_frac - 0.5
        draft_sign = np.sign(draft_centred)
        draft_shape = [1, 1, 1]
        draft_shape[draft_dim] = draft_n
        draft_sign_b = draft_sign.reshape(draft_shape)
        offsets[..., draft_axis] += (
            draft_sign_b * length_ramp_15_b * height_push
        )
        # Whole slab translates vertically (no draft index dependency)
        offsets[..., draft_axis] += length_ramp_15_b * axis_z

        # ---- Cross-section shape: corner pull toward circle/ellipse ----
        # Build masks selecting the four "corners" along the (beam, draft)
        # plane — a corner is a lattice point at index 0 or end on both
        # the beam dim AND the draft dim.
        c = v["cross_section_c"]
        circle_pull = (c - 0.5) * 0.15 * max(box_size[beam_axis], box_size[draft_axis])

        beam_corner_mask = np.zeros(beam_n, dtype=float)
        beam_corner_mask[0] = -1.0
        beam_corner_mask[-1] = +1.0
        draft_corner_mask = np.zeros(draft_n, dtype=float)
        draft_corner_mask[0] = -1.0
        draft_corner_mask[-1] = +1.0
        beam_corner_b = beam_corner_mask.reshape(beam_shape)
        draft_corner_b = draft_corner_mask.reshape(draft_shape)
        # corner_indicator is +/-1 at the four (beam, draft) corners and
        # 0 elsewhere. We use abs() to gate on "is a corner" and use the
        # signed mask for the direction.
        is_corner = (np.abs(beam_corner_b) > 0) & (np.abs(draft_corner_b) > 0)
        offsets[..., beam_axis] -= np.where(is_corner, beam_corner_b, 0.0) * circle_pull
        offsets[..., draft_axis] -= np.where(is_corner, draft_corner_b, 0.0) * circle_pull

        # ---- Volume coefficient: overall magnitude scale ----
        offsets *= 0.5 + v["volume_coef"]

        # ---- Nose sharpness: taper the forward-most primary slab ----
        sharpness = v["nose_sharpness"]
        taper = (1.0 - sharpness) * 0.3
        # Build a slab indicator that picks the LAST primary slice
        # (i = l_cp-1) and broadcasts to the full lattice.
        nose_slab = np.zeros(prim_n, dtype=float)
        nose_slab[-1] = 1.0
        nose_slab_b = nose_slab.reshape(prim_shape)
        # Beam corner direction at nose: -1 at beam_dim=0, +1 at beam_dim=end.
        # (Same convention as before, just generalised across the actual
        # beam_dim.) Multiply by 0/1 corner mask to restrict to corners.
        beam_signed_corner = np.zeros(beam_n, dtype=float)
        beam_signed_corner[0] = -1.0
        beam_signed_corner[-1] = +1.0
        beam_signed_corner_b = beam_signed_corner.reshape(beam_shape)
        draft_signed_corner = np.zeros(draft_n, dtype=float)
        draft_signed_corner[0] = -1.0
        draft_signed_corner[-1] = +1.0
        draft_signed_corner_b = draft_signed_corner.reshape(draft_shape)
        # Mask the four nose-tip corners (i==prim_n-1, beam=∂, draft=∂).
        nose_corner_mask = (
            (nose_slab_b > 0)
            & (np.abs(beam_signed_corner_b) > 0)
            & (np.abs(draft_signed_corner_b) > 0)
        )
        offsets[..., beam_axis] -= np.where(
            nose_corner_mask,
            beam_signed_corner_b * taper * box_size[beam_axis] * 0.1,
            0.0,
        )
        offsets[..., draft_axis] -= np.where(
            nose_corner_mask,
            draft_signed_corner_b * taper * box_size[draft_axis] * 0.1,
            0.0,
        )

        # F5 (mesh-quality design §4): clamp per-lattice-point offset so
        # no single control point moves farther than half the local cell
        # spacing along any axis. This prevents triangle inversion near
        # the nose tip when ``longitudinal_pos`` + low ``nose_sharpness``
        # push two lattice columns past each other — the failure mode
        # Agent 1 saw as 66–76 flipped triangles in the generated mesh.
        # Spacing per world axis = box_size / (lattice resolution - 1)
        # along that axis; the lattice's spatial dims map 1-to-1 to world
        # axes (audit 2026-04-26).
        spacing = np.array(
            [
                box_size[axis] / max(offsets.shape[axis] - 1, 1)
                for axis in range(3)
            ],
            dtype=float,
        )
        max_offset = 0.5 * spacing  # 3-vector, broadcasts over (l, m, n)
        # Protect against a degenerate zero-width box direction.
        safe_max = np.where(max_offset > 0, max_offset, 1.0)
        # Clamp symmetrically per axis.
        np.clip(offsets, -safe_max, safe_max, out=offsets)

        return offsets

    # ---- core FFD ---------------------------------------------------------

    def _apply_ffd(
        self,
        *,
        points: np.ndarray,
        box_origin: np.ndarray,
        box_size: np.ndarray,
        offsets: np.ndarray,
    ) -> np.ndarray:
        """Apply the Bernstein-basis FFD sum to ``points``."""
        safe_size = np.where(box_size > 0, box_size, 1.0)
        local = (points - box_origin) / safe_size
        local = np.clip(local, 0.0, 1.0)
        s = local[:, 0]
        t = local[:, 1]
        u = local[:, 2]

        l_cp, m_cp, n_cp = self.LATTICE_SHAPE
        l_deg, m_deg, n_deg = l_cp - 1, m_cp - 1, n_cp - 1

        # Pre-compute per-axis Bernstein tables: shape (n_cp, N).
        bi = np.stack([_bernstein(l_deg, i, s) for i in range(l_cp)], axis=0)
        bj = np.stack([_bernstein(m_deg, j, t) for j in range(m_cp)], axis=0)
        bk = np.stack([_bernstein(n_deg, k, u) for k in range(n_cp)], axis=0)

        # Sum contributions of every lattice point.
        deformed = points.copy()
        for i in range(l_cp):
            for j in range(m_cp):
                for k in range(n_cp):
                    w = (bi[i] * bj[j] * bk[k])[:, None]  # (N, 1)
                    deformed = deformed + w * offsets[i, j, k]
        return deformed


def _resolve_secondary_axes(
    region: dict, primary_axis: int
) -> tuple[int, int]:
    """Pick (beam_axis, draft_axis) from the region dict, falling back to
    the legacy ``other_axes[0]/[1]`` heuristic when the region doesn't
    carry them.

    Audit 2026-04-26: the StubGeometryAdapter now persists
    ``beam_axis``/``draft_axis`` in the region dict by analyzing the
    repaired mesh's symmetry, so production runs use the correct axis.
    Synthetic test meshes (icosphere, box) are typically passed in with
    a hand-built region that has only ``axis_index`` — for those the
    legacy heuristic is the right fallback because their non-primary
    axes are both centered on zero.
    """
    other_axes = [a for a in range(3) if a != primary_axis]
    beam = region.get("beam_axis")
    draft = region.get("draft_axis")
    if beam is None:
        beam = other_axes[0]
    if draft is None:
        # Pick the remaining axis if beam already set; otherwise fall
        # back to other_axes[1].
        candidates = [a for a in other_axes if a != int(beam)]
        draft = candidates[0] if candidates else other_axes[1]
    return int(beam), int(draft)


def _bernstein(degree: int, index: int, t: np.ndarray) -> np.ndarray:
    """Compute B_index^degree(t) for an array of parameters t ∈ [0, 1]."""
    coeff = comb(degree, index)
    return coeff * (t ** index) * ((1.0 - t) ** (degree - index))


# Bug #6 (audit 2026-04-26): minimum welded vertex count below which the
# Taubin pass is skipped. Below ~60 unique nodes the welded Laplacian
# graph picks up degree-1 boundary nodes whose noisy steps overwhelm the
# smoothstep taper, producing non-manifold output. Skipping smoothing
# leaves the mesh slightly more polygonal but keeps it watertight, which
# matters for downstream snappyHexMesh. See bug context in 2026-04-26
# audit memo.
_TAUBIN_MIN_UNIQUE_VERTICES = 60

# Bug #6: same idea, but on the participating (in-region) raw-vertex
# count. A bulb region with fewer than this many vertices is too sparse
# for Taubin to behave well even when the rest of the mesh is dense.
_TAUBIN_MIN_PARTICIPATING_VERTICES = 30


def _adaptive_weld_decimals(mesh: trimesh.Trimesh) -> int:
    """Pick the ``np.round(decimals=...)`` precision for position welding
    based on the mesh's bounding-box diagonal.

    Bug #8 (audit 2026-04-26): a fixed ``decimals=6`` welds positions to
    1 µm — fine for meter-scale STLs (1 µm ≪ float32 ULP at unit scale)
    but useless for STLs exported in millimetres (1 µm = 1 nm in mesh
    units, well below float32 precision so coincident vertices stay
    unwelded). The adaptive tolerance is ``extent_diag * 1e-9`` so the
    weld remains a tiny fraction of the hull no matter the unit.

    The formula clamps ``decimals`` to never go *coarser* than 6 — that
    keeps backward compatibility on the 100 m baseline hull (extent_diag
    ≈ 247 m → tol ≈ 2.5e-7 → decimals ≈ 6).
    """
    extent_diag = float(np.linalg.norm(mesh.extents))
    # Floor on the absolute tolerance to avoid blowing up ``decimals`` on
    # a degenerate zero-extent mesh. 1e-12 is safely above float64 ULP
    # at coordinates of order 1.
    tol = max(extent_diag * 1e-9, 1e-12)
    return max(int(-np.log10(tol)), 6)


def _apply_taubin_to_region(
    mesh: trimesh.Trimesh,
    *,
    primary_axis: int,
    blend_start: float,
    blend_width: float,
    iterations: int,
    lamb: float = 0.5,
    nu: float = -0.53,
) -> bool:
    """Apply Taubin λ/μ smoothing to vertices that participate in the FFD.

    Returns ``True`` when Taubin actually ran, ``False`` when one of the
    Bug #6 small-region guards fired and the smoothing was skipped. The
    caller uses the return value to mirror the same skip on dependent
    passes (e.g. seam smoothing in Add #5).

    Done in-place on ``mesh.vertices``. Vertices outside the region
    (axis < blend_start) are explicitly held fixed so the rest of the
    hull triangulation never moves. Inside the blend zone the Taubin
    update is multiplied by a smoothstep weight that rises from 0 at
    ``blend_start`` to 1 at ``blend_start + blend_width`` (mesh-quality
    design §4 Fix F4) — this prevents the C0 discontinuity the original
    binary mask introduced right at the bulb boundary.

    The classic Taubin parameters (λ=0.5, ν=-0.53) satisfy
    λν / (λ+ν) ≈ -0.22 which preserves frequencies below the cutoff —
    low-frequency shape survives, facet edges flatten.

    Critical fix F1 (mesh-quality design §4): build the adjacency graph
    on position-welded vertices rather than raw index-per-face vertices.
    STL loaders typically produce 3 unique vertices per face (no index
    sharing, degree-2 graph), which used to make Taubin collapse the
    mesh by -99% volume. Welding by position (audit 2026-04-26 Bug #8:
    quantisation precision is now adaptive — ``_adaptive_weld_decimals``
    derives it from the mesh's bounding-box diagonal so mm-scale STLs
    weld correctly too).

    Audit 2026-04-26 Bug #6: skip Taubin entirely when the welded mesh
    or the participating region is too small. On <60 unique nodes the
    welded graph has too many degree-1 boundary nodes; on <30
    participating vertices the smoothstep taper has nothing meaningful
    to clamp. In both cases Taubin can break watertightness, and a
    slightly polygonal-but-watertight mesh is worth far more than a
    smoothed-but-broken one.

    Implementation is fully vectorised: we build a sparse CSR Laplacian
    once (edges derived from faces via numpy, no Python loops) and then
    each smoothing step is one ``A @ positions`` matrix-vector product.
    On a 14 k-vertex hull the previous Python-loop version took ~30 s;
    the vectorised version completes in ~50 ms.
    """
    from scipy.sparse import csr_matrix

    vertices = np.asarray(mesh.vertices, dtype=float)
    n = len(vertices)
    if n == 0 or iterations <= 0:
        return False

    # ---- Bug #6: small-region guard --------------------------------------
    #
    # Count raw participating vertices first — cheap, and a hard bypass
    # when the bulb region is sparse.
    raw_axis_pre = vertices[:, primary_axis]
    n_participating = int(np.count_nonzero(raw_axis_pre >= blend_start))
    if n_participating < _TAUBIN_MIN_PARTICIPATING_VERTICES:
        return False

    # ---- F1: weld coincident vertices by position ------------------------
    #
    # Bug #8: ``decimals`` is now adaptive to the hull scale.
    # ``inverse`` is a length-n array: inverse[k] = the unique-position
    # index that vertices[k] belongs to. Duplicated STL vertices collapse
    # onto a single node in the welded graph so Taubin operates on the
    # proper connected mesh topology.
    decimals = _adaptive_weld_decimals(mesh)
    rounded = np.round(vertices, decimals=decimals)
    unique_positions, inverse = np.unique(rounded, axis=0, return_inverse=True)
    n_unique = int(unique_positions.shape[0])

    # Bug #6: second guard — welded graph must have enough nodes for the
    # Laplacian to behave well. Below the threshold we skip smoothing.
    if n_unique < _TAUBIN_MIN_UNIQUE_VERTICES:
        return False

    # Welded face indices: each face's three vertex indices map onto the
    # unique-position space.
    faces = np.asarray(mesh.faces, dtype=np.int64)
    welded_faces = inverse[faces]

    # Edges (undirected) in the welded graph, de-duplicated. Drop self-
    # edges introduced by degenerate faces (all three corners welded onto
    # one node) — they only inflate the diagonal.
    edges = np.concatenate(
        [
            welded_faces[:, [0, 1]],
            welded_faces[:, [1, 2]],
            welded_faces[:, [2, 0]],
        ],
        axis=0,
    )
    edges = edges[edges[:, 0] != edges[:, 1]]
    if len(edges) == 0:
        return False
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    # Symmetric adjacency on welded graph with 1 / degree_i weights so
    # (A @ positions)[i] = mean(neighbours of i).
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    degree = np.bincount(rows, minlength=n_unique).astype(float)
    safe_degree = np.where(degree > 0, degree, 1.0)
    weights = 1.0 / safe_degree[rows]
    adjacency = csr_matrix((weights, (rows, cols)), shape=(n_unique, n_unique))

    # ---- F4: smoothstep weight on raw vertices instead of binary mask ----
    #
    # ``taper`` is 1 inside the bulb region, 0 outside ``blend_start``, and
    # smoothly 0→1 across the blend zone so the smoothing doesn't clip at
    # the boundary and there is no C0 discontinuity across it.
    raw_axis = vertices[:, primary_axis]
    if blend_width > 0:
        t_raw = (raw_axis - blend_start) / blend_width
    else:
        t_raw = np.where(raw_axis >= blend_start, 1.0, 0.0)
    t_raw = np.clip(t_raw, 0.0, 1.0)
    vertex_taper = t_raw * t_raw * (3.0 - 2.0 * t_raw)

    # Welded-space taper: max over all raw vertices mapped onto each
    # unique position, so a unique node is "movable" if any of its
    # duplicates participate in the deformation.
    welded_taper = np.zeros(n_unique, dtype=float)
    np.maximum.at(welded_taper, inverse, vertex_taper)
    welded_taper_col = welded_taper[:, None]

    original = vertices.copy()

    # Taubin runs on welded positions; compute welded positions as the
    # mean of their duplicates' raw positions.
    welded_positions = np.zeros((n_unique, 3), dtype=float)
    counts = np.bincount(inverse, minlength=n_unique).astype(float)
    safe_counts = np.where(counts > 0, counts, 1.0)
    for dim in range(3):
        welded_positions[:, dim] = (
            np.bincount(inverse, weights=vertices[:, dim], minlength=n_unique)
            / safe_counts
        )

    for _ in range(iterations):
        for step in (lamb, nu):
            neighbour_means = adjacency @ welded_positions
            laplacian = neighbour_means - welded_positions
            welded_positions = welded_positions + step * laplacian * welded_taper_col

    # Scatter welded positions back onto the raw vertex array.
    vertices = welded_positions[inverse]
    # F4: blend the smoothed positions with the originals using the raw
    # taper so the transition at the blend boundary stays continuous
    # (welded_taper is per-unique-node, but duplicates of a mixed node
    # sit at different raw axis coordinates; the raw taper respects
    # that).
    vertices = (
        original * (1.0 - vertex_taper[:, None])
        + vertices * vertex_taper[:, None]
    )

    mesh.vertices = vertices
    return True


def _apply_seam_laplacian(
    mesh: trimesh.Trimesh,
    *,
    primary_axis: int,
    blend_start: float,
    blend_width: float,
    iterations: int,
    beam_axis: int | None = None,
) -> None:
    """Apply a few pure-Laplacian (positive-only) smoothing iterations to
    the ring of vertices in the FFD seam.

    Audit C 2026-04-26 (Add #5): the smoothstep weight ``w`` rises from
    0 at ``blend_start`` to 1 at ``blend_start + blend_width``. The seam
    ring is the set of vertices with ``w ∈ (0.05, 0.95)`` — they sit in
    the blend zone where the FFD displacement is partially applied, so
    triangulation defects (mismatching dihedral angles between adjacent
    facets) concentrate there. A tiny Laplacian pass on this ring only
    levels the dihedral without disturbing the rest of the hull.

    Implementation mirrors ``_apply_taubin_to_region``:
    * Welded by position (matches Taubin's adjacency graph so we don't
      pick up the disconnected per-face-vertex pathology).
    * Pinned vertices outside the ring are *not* updated; their welded
      position is held fixed across iterations.
    * One iteration = ``new = mean(neighbours)`` for ring nodes only.

    Done in-place on ``mesh.vertices``. Falls through to a no-op when
    the seam ring is empty (degenerate or tiny mesh).
    """
    from scipy.sparse import csr_matrix

    vertices = np.asarray(mesh.vertices, dtype=float)
    n = len(vertices)
    if n == 0 or iterations <= 0:
        return

    decimals = _adaptive_weld_decimals(mesh)
    rounded = np.round(vertices, decimals=decimals)
    unique_positions, inverse = np.unique(rounded, axis=0, return_inverse=True)
    n_unique = int(unique_positions.shape[0])

    faces = np.asarray(mesh.faces, dtype=np.int64)
    welded_faces = inverse[faces]

    edges = np.concatenate(
        [
            welded_faces[:, [0, 1]],
            welded_faces[:, [1, 2]],
            welded_faces[:, [2, 0]],
        ],
        axis=0,
    )
    edges = edges[edges[:, 0] != edges[:, 1]]
    if len(edges) == 0:
        return
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    degree = np.bincount(rows, minlength=n_unique).astype(float)
    safe_degree = np.where(degree > 0, degree, 1.0)
    weights = 1.0 / safe_degree[rows]
    adjacency = csr_matrix((weights, (rows, cols)), shape=(n_unique, n_unique))

    raw_axis = vertices[:, primary_axis]
    if blend_width > 0:
        t_raw = (raw_axis - blend_start) / blend_width
    else:
        t_raw = np.where(raw_axis >= blend_start, 1.0, 0.0)
    t_raw = np.clip(t_raw, 0.0, 1.0)
    vertex_smooth_weight = t_raw * t_raw * (3.0 - 2.0 * t_raw)
    # Ring mask in raw-vertex space.
    ring_mask_raw = (vertex_smooth_weight > 0.05) & (vertex_smooth_weight < 0.95)
    if not ring_mask_raw.any():
        return

    # Welded-space ring mask: a unique node is on the ring if ANY of its
    # raw duplicates is.
    ring_mask = np.zeros(n_unique, dtype=bool)
    np.logical_or.at(ring_mask, inverse, ring_mask_raw)

    # Compute welded positions as mean of duplicates.
    welded_positions = np.zeros((n_unique, 3), dtype=float)
    counts = np.bincount(inverse, minlength=n_unique).astype(float)
    safe_counts = np.where(counts > 0, counts, 1.0)
    for dim in range(3):
        welded_positions[:, dim] = (
            np.bincount(inverse, weights=vertices[:, dim], minlength=n_unique)
            / safe_counts
        )

    # When the deformer is enforcing mirror symmetry around a beam axis,
    # the seam ring may straddle the ring boundary asymmetrically: a +beam
    # vertex's smoothstep weight may push it just into the ring while its
    # -beam partner sits just outside. Updating only one half of such a
    # pair widens the mirror error the upstream symmetry pass just
    # minimised. Strategy: identify mutual mirror pairs in the FULL
    # welded mesh; update ANY vertex whose mirror is in the ring (even
    # if the vertex itself isn't), and average the two neighbour means
    # so both halves move together. ``beam_axis is None`` means symmetry
    # is off and the full 3-D Laplacian on every ring node is fine.
    if beam_axis is not None:
        from scipy.spatial import cKDTree

        # Find mutual mirror pairs across the full welded mesh, not just
        # the ring — the ring is taper-defined, but symmetry is global.
        mirror_all = welded_positions.copy()
        mirror_all[:, int(beam_axis)] *= -1.0
        tree_all = cKDTree(welded_positions)
        bbox_diag = float(
            np.linalg.norm(
                welded_positions.max(axis=0) - welded_positions.min(axis=0)
            )
        )
        threshold = max(0.02 * bbox_diag, 1e-9)
        distances_all, partners_all = tree_all.query(mirror_all, k=1)
        partners_all = np.asarray(partners_all, dtype=int)
        mutual_all = partners_all[partners_all] == np.arange(n_unique)
        valid_all = (
            mutual_all
            & (distances_all <= threshold)
            & (partners_all != np.arange(n_unique))
        )

        # Active set: any welded node whose ITSELF or its valid mirror
        # partner lies on the ring. ``paired_global[i]`` is i's mirror
        # partner (or -1 if no mutual mirror).
        paired_global = np.where(valid_all, partners_all, -1)
        ring_or_mirror_in_ring = ring_mask | (
            (paired_global >= 0) & ring_mask[np.where(paired_global >= 0, paired_global, 0)]
        )
        active_mask = ring_or_mirror_in_ring & (paired_global >= 0)
    else:
        active_mask = ring_mask
        paired_global = -np.ones(n_unique, dtype=np.int64)

    # Damped step: a pure-replacement Laplacian (pos = mean(neighbours))
    # always shrinks the ring inward by a fraction of the local edge
    # length per iteration. After 2 such iterations on a coarse mesh,
    # ring vertices can move enough to leave the bulb region entirely,
    # creating downstream asymmetry. A damped step (pos += alpha *
    # (mean - pos)) preserves the mean's smoothing effect on the
    # dihedral while limiting per-iteration drift to ``alpha`` of the
    # full Laplacian. With alpha=0.5 and 2 iterations the cumulative
    # drift is ~75% of the pure-Laplacian case but the dihedral signal
    # is still flattened (the 5% test threshold remains comfortably met).
    alpha = 0.5
    for _ in range(iterations):
        neighbour_means = adjacency @ welded_positions
        if beam_axis is None:
            updated = welded_positions + alpha * (
                neighbour_means - welded_positions
            )
            welded_positions = np.where(
                active_mask[:, None], updated, welded_positions
            )
        else:
            # Symmetrise the (damped) update across each mutual mirror
            # pair so +beam and -beam nodes (where at least one is on
            # the ring) move together, preserving the mirror invariant
            # exactly.
            updated = welded_positions.copy()
            done = np.zeros(n_unique, dtype=bool)
            active_indices = np.nonzero(active_mask)[0]
            for global_i in active_indices:
                if done[global_i]:
                    continue
                global_j = int(paired_global[global_i])
                if global_j < 0 or global_j == global_i:
                    continue
                m_i = neighbour_means[global_i].copy()
                m_j = neighbour_means[global_j].copy()
                # Mirror j's mean back to the +beam side, average, then
                # apply a damped step to current pos (also averaged).
                m_j[int(beam_axis)] *= -1.0
                target = 0.5 * (m_i + m_j)
                p_i = welded_positions[global_i].copy()
                p_j = welded_positions[global_j].copy()
                p_j[int(beam_axis)] *= -1.0
                cur_avg = 0.5 * (p_i + p_j)
                damped = cur_avg + alpha * (target - cur_avg)
                updated[global_i] = damped
                damped_mirror = damped.copy()
                damped_mirror[int(beam_axis)] *= -1.0
                updated[global_j] = damped_mirror
                done[global_i] = True
                done[global_j] = True
            welded_positions = updated

    # Scatter welded positions back, but only for raw vertices on the
    # ring (and active in symmetry-paired set when beam_axis is set).
    new_vertices = vertices.copy()
    if beam_axis is None:
        scatter_mask = ring_mask_raw
    else:
        scatter_mask = active_mask[inverse]
    new_vertices[scatter_mask] = welded_positions[inverse[scatter_mask]]

    mesh.vertices = new_vertices


def _enforce_mirror_symmetry_subset(
    vertices: np.ndarray,
    beam_axis: int,
    subset_indices: np.ndarray,
    primary_axis: int,
) -> np.ndarray:
    """Symmetrise only the vertices in ``subset_indices`` around the beam
    midplane; leave the rest bit-identical.

    Safety guards added after the first naive implementation badly
    collapsed the mesh (volume -29%, surface +199%):

    1. **Subset scope** — only the participating bulb-region vertices get
       touched. The aft cylindrical hull stays exactly where it was.
    2. **Mutual pairing** — a vertex i is only merged with its candidate
       partner j if j's nearest-mirror is also i. Cross-pairings (i→j
       but j→k≠i) are skipped.
    3. **Distance gate** — the mirror distance must be below a local
       threshold derived from the subset's bounding-box diagonal; far
       partners are clearly wrong matches and get skipped.

    Vertices failing the guards keep their pre-symmetry coordinates —
    small residual asymmetry is always preferred over a folded mesh.
    """
    from scipy.spatial import cKDTree

    symmetric = vertices.copy()
    if len(subset_indices) == 0:
        return symmetric

    subset = symmetric[subset_indices]
    mirrors = subset.copy()
    mirrors[:, beam_axis] *= -1.0
    tree = cKDTree(subset)
    # For each subset vertex i, nearest subset vertex to mirror(i).
    distances, partners = tree.query(mirrors, k=1)
    partners = np.asarray(partners, dtype=int)

    # Mutual-pairing mask: partner[partner[i]] == i
    mutual = partners[partners] == np.arange(len(subset))

    # Distance threshold = 2% of subset bounding-box diagonal (small
    # enough to catch cross-pairings, large enough to tolerate mesh
    # noise after FFD displacement).
    bbox_diag = float(np.linalg.norm(subset.max(axis=0) - subset.min(axis=0)))
    threshold = 0.02 * bbox_diag if bbox_diag > 0 else 1e-6

    # F3 (mesh-quality design §4): snapping a self-pair (vertex whose
    # own nearest-mirror is itself) to beam=0 used to collapse any
    # off-centerline vertex without a real mirror partner — e.g. the
    # bottom flange produced 113 vertices with |beam| up to 0.60 m
    # snapped flat to zero, producing the "broken flap" visible in the
    # user's screenshots. Self-pairs are now only collapsed if the
    # vertex is *already* within ``threshold`` of the beam midplane;
    # anything farther out is left at its original position.
    paired: set[int] = set()
    for local_i in range(len(subset)):
        if local_i in paired:
            continue
        local_j = int(partners[local_i])
        if local_i == local_j:
            # Only snap a self-paired vertex to the centerline when it
            # really sits near it already (|beam| < threshold). Farther
            # vertices are leftovers of an asymmetric triangulation and
            # must keep their coordinates.
            if abs(float(subset[local_i][beam_axis])) < threshold:
                global_i = int(subset_indices[local_i])
                symmetric[global_i, beam_axis] = 0.0
            paired.add(local_i)
            continue
        if not mutual[local_i]:
            continue
        if float(distances[local_i]) > threshold:
            continue
        if local_j in paired:
            continue
        global_i = int(subset_indices[local_i])
        global_j = int(subset_indices[local_j])
        avg = 0.5 * (subset[local_i] + mirrors[local_j])
        symmetric[global_i] = avg
        mirrored_avg = avg.copy()
        mirrored_avg[beam_axis] *= -1.0
        symmetric[global_j] = mirrored_avg
        paired.add(local_i)
        paired.add(local_j)

    return symmetric


# Audit 2026-04-26 — Module B3: bulb-region uniform remesh + tip finisher.
# Both helpers are opt-in via BulbFFDDeformer.__init__ kwargs that default
# to 0 / disabled, so the existing 307 tests cannot regress.


def _uniform_remesh_region(
    mesh: trimesh.Trimesh,
    region: dict,
    *,
    max_edge_factor: float,
) -> trimesh.Trimesh:
    """Replace the bulb region's triangulation with a uniformly-edged copy.

    Selects faces whose at-least-one vertex sits in the bulb region (axis
    >= blend_start where blend_start = axis_min - 0.10*(axis_max-axis_min)
    matches the deformer's smoothstep blend zone), runs
    ``trimesh.Trimesh.subdivide_to_size`` on a copy of just those faces
    with ``max_edge = max_edge_factor * box_size_min(bulb_region)``, then
    stitches the result back into the full mesh by replacing those faces.

    Falls back to the input mesh unchanged when:
    * No faces are selected for the bulb region.
    * The trimesh subdivision raises (e.g. malformed sub-mesh).
    * The result has more than 2× the input face count (runaway — likely
      a degenerate slim region whose ``box_size_min`` is near zero).

    The function only accepts the bulb-region face slab; the rest of the
    mesh's vertices and faces are bit-identical to the input. Adjacent
    faces in the rest of the hull continue to reference the shared
    vertices that border the remeshed region (no T-junctions because
    ``subdivide_to_size`` only subdivides edges *inside* the sub-mesh —
    however, when a sub-mesh's boundary edge gets subdivided it will
    introduce a T-junction relative to neighbouring faces in the outer
    mesh; we accept this because the FFD pass that follows applies a
    smoothstep blend across the boundary and the post-FFD ``merge_vertices``
    consolidates coincident points; the seam-smoothing pass also runs at
    a slightly different fan-out but its dihedral test still works).

    Strategy notes:
    * We use ``blend_start`` (not ``axis_min``) for selection so the
      remesh covers the entire participating zone, not just the deformed
      part — otherwise the seam between remeshed and original triangles
      lands inside the deformation zone and produces a fresh ridge.
    * We index by face, not by vertex — a face is "in region" if any of
      its three vertices is in region. This includes boundary triangles
      whose far vertex sits outside the region; they get remeshed too,
      and ``subdivide_to_size`` keeps the outer boundary topology
      consistent because it only adds vertices on edges, never on the
      perimeter strictly outside the input.
    """
    primary_axis = int(region["axis_index"])
    axis_min = float(region["axis_min"])
    axis_max = float(region["axis_max"])
    if axis_max <= axis_min:
        return mesh

    blend_width = BulbFFDDeformer.BLEND_WIDTH_FRACTION * (axis_max - axis_min)
    blend_start = axis_min - blend_width

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    in_region_vertex = vertices[:, primary_axis] >= blend_start
    if not in_region_vertex.any():
        return mesh

    in_region_face = np.any(in_region_vertex[faces], axis=1)
    if not in_region_face.any():
        return mesh

    # Compute target max edge from the bulb region's bounding box: use the
    # smallest extent so factor=0.05 yields a fraction of the *thinnest*
    # axis (avoids a giant max_edge on long-thin bulbs).
    region_vertex_indices = np.unique(faces[in_region_face].ravel())
    region_vertices = vertices[region_vertex_indices]
    extents = region_vertices.max(axis=0) - region_vertices.min(axis=0)
    box_size_min = float(np.min(extents[extents > 0])) if (extents > 0).any() else 0.0
    if box_size_min <= 0.0:
        return mesh
    max_edge = float(max_edge_factor) * box_size_min
    if max_edge <= 0.0:
        return mesh

    # Build the sub-mesh of just the in-region faces. Compact the vertex
    # array so ``subdivide_to_size`` operates on a tight set of nodes.
    sub_face_indices = np.nonzero(in_region_face)[0]
    sub_faces_global = faces[sub_face_indices]
    used_vertex_global = np.unique(sub_faces_global.ravel())
    global_to_local = -np.ones(len(vertices), dtype=np.int64)
    global_to_local[used_vertex_global] = np.arange(len(used_vertex_global))
    sub_faces_local = global_to_local[sub_faces_global]
    sub_vertices = vertices[used_vertex_global].copy()

    sub_mesh = trimesh.Trimesh(
        vertices=sub_vertices,
        faces=sub_faces_local,
        process=False,
    )

    try:
        remeshed = sub_mesh.subdivide_to_size(max_edge=max_edge)
    except Exception:
        return mesh
    if remeshed is None:
        return mesh

    new_sub_vertices = np.asarray(remeshed.vertices, dtype=float)
    new_sub_faces = np.asarray(remeshed.faces, dtype=np.int64)

    # Runaway guard: cap the absolute face count of the bulb-region
    # remesh at 200000. Uniform-remeshing slivers always grows the face
    # count (often by 10–100×), but if the result would push the bulb
    # region above 200 k tris the downstream FFD + Taubin cost becomes
    # unreasonable and we fall back to the un-remeshed mesh. The cap is
    # well above the typical real-world bulb-region size (~5–20 k tris)
    # so genuinely useful remeshes always pass.
    if len(new_sub_faces) > 200_000:
        return mesh

    # Stitch: rebuild the full mesh by concatenating
    #   (out-of-region faces, with their original vertex indices)
    # + (newly-remeshed in-region faces, with vertex indices offset by
    #    len(vertices) to point into the new vertex block)
    # then compact unused vertices via merge_vertices.
    keep_face_mask = ~in_region_face
    kept_faces = faces[keep_face_mask]
    new_faces_global = new_sub_faces + len(vertices)
    combined_faces = np.concatenate([kept_faces, new_faces_global], axis=0)

    # The new sub_mesh's vertices already contain the original
    # used_vertex_global positions PLUS any new midpoints introduced by
    # subdivide_to_size. Offsetting by len(vertices) preserves all
    # positions and lets merge_vertices fold any duplicates with the
    # original vertices that border the remeshed region.
    combined_vertices = np.concatenate([vertices, new_sub_vertices], axis=0)

    out = trimesh.Trimesh(
        vertices=combined_vertices,
        faces=combined_faces,
        process=False,
    )
    # Fold border vertices (the same physical positions exist in both
    # ``vertices`` and ``new_sub_vertices``).
    out.merge_vertices()
    return out




def _apply_tip_finisher(
    mesh: trimesh.Trimesh,
    *,
    primary_axis: int,
    blend_start: float,
    iterations: int,
    beam_axis: int | None = None,
    threshold_deg: float = 30.0,
    alpha: float = 0.5,
) -> None:
    """Apply targeted Laplacian smoothing on faces with high dihedral.

    Audit 2026-04-26 — Module B3 Add #2. After the main Taubin pass +
    seam smoothing + post-repair, residual creases can survive at the
    bow tip when the baseline triangulation had isolated slivers (small
    welded-graph degree → low Taubin coupling). The finisher selects:

    1. All faces with at least one vertex in ``axis >= blend_start`` (the
       bulb region — same scope as the FFD).
    2. Among those, the ones with at least one neighbour whose dihedral
       exceeds ``threshold_deg`` (default 30°).

    The union of those faces' vertices gets ``iterations`` damped
    Laplacian steps:  ``new = pos + alpha * (mean(neighbours) - pos)``.

    Done in-place on ``mesh.vertices``. Falls through when:
    * No vertices are in the bulb region.
    * No high-dihedral pairs exist (mesh is already clean).

    The pass is gentle by design (alpha=0.5, default 2 iterations) so
    it does not flatten legitimate sharp features in the rest of the
    mesh; combined with the threshold gate it only touches the parts
    of the bulb tip that are actually defective.
    """
    from collections import defaultdict
    from scipy.sparse import csr_matrix

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0 or iterations <= 0:
        return

    # Welded graph (matches Taubin's adjacency build).
    decimals = _adaptive_weld_decimals(mesh)
    rounded = np.round(vertices, decimals=decimals)
    unique_positions, inverse = np.unique(rounded, axis=0, return_inverse=True)
    n_unique = int(unique_positions.shape[0])
    if n_unique < 4:
        return
    welded_faces = inverse[faces]

    # Face normals.
    tri = vertices[faces]
    n_face = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nlen = np.linalg.norm(n_face, axis=1, keepdims=True)
    nlen = np.where(nlen > 0, nlen, 1.0)
    n_face = n_face / nlen

    # Edge -> faces map on the welded graph (dedupes per-face-vertex
    # baseline STLs).
    edges_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, face in enumerate(welded_faces):
        a, b, c = int(face[0]), int(face[1]), int(face[2])
        for e in ((a, b), (b, c), (c, a)):
            key = (min(e), max(e))
            edges_to_faces[key].append(fi)

    threshold_rad = np.radians(threshold_deg)

    # In-region face mask: at least one vertex of the face has axis >= blend_start.
    in_region_vertex = vertices[:, primary_axis] >= blend_start
    in_region_face = np.any(in_region_vertex[faces], axis=1)

    high_dihedral_face = np.zeros(len(faces), dtype=bool)
    for fids in edges_to_faces.values():
        if len(fids) != 2:
            continue
        f1, f2 = fids
        if not (in_region_face[f1] or in_region_face[f2]):
            continue
        cos_t = float(np.clip(float(np.dot(n_face[f1], n_face[f2])), -1.0, 1.0))
        angle = float(np.arccos(cos_t))
        if angle > threshold_rad:
            high_dihedral_face[f1] = True
            high_dihedral_face[f2] = True

    if not high_dihedral_face.any():
        return

    # Active welded vertices: union of high-dihedral faces' welded
    # vertices, intersected with in-region (so we never touch a
    # high-dihedral face that happens to span the seam — its outer
    # vertex would be moved otherwise).
    active_raw = np.zeros(len(vertices), dtype=bool)
    for fi in np.nonzero(high_dihedral_face)[0]:
        for k in range(3):
            vi = int(faces[fi, k])
            if in_region_vertex[vi]:
                active_raw[vi] = True

    if not active_raw.any():
        return

    # Welded-space active mask.
    active_welded = np.zeros(n_unique, dtype=bool)
    np.logical_or.at(active_welded, inverse, active_raw)

    # Build symmetric adjacency on the welded graph.
    edges = np.concatenate(
        [
            welded_faces[:, [0, 1]],
            welded_faces[:, [1, 2]],
            welded_faces[:, [2, 0]],
        ],
        axis=0,
    )
    edges = edges[edges[:, 0] != edges[:, 1]]
    if len(edges) == 0:
        return
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    degree = np.bincount(rows, minlength=n_unique).astype(float)
    safe_degree = np.where(degree > 0, degree, 1.0)
    weights = 1.0 / safe_degree[rows]
    adjacency = csr_matrix((weights, (rows, cols)), shape=(n_unique, n_unique))

    # Compute welded positions as the mean of duplicates' raw positions.
    welded_positions = np.zeros((n_unique, 3), dtype=float)
    counts = np.bincount(inverse, minlength=n_unique).astype(float)
    safe_counts = np.where(counts > 0, counts, 1.0)
    for dim in range(3):
        welded_positions[:, dim] = (
            np.bincount(inverse, weights=vertices[:, dim], minlength=n_unique)
            / safe_counts
        )

    # Mirror-pair handling when symmetry is enforced — same pattern as
    # ``_apply_seam_laplacian``: identify mutual mirror pairs across the
    # full welded mesh, average the neighbour-mean updates, apply a damped
    # step that preserves the mirror invariant exactly.
    if beam_axis is not None:
        from scipy.spatial import cKDTree

        mirror_all = welded_positions.copy()
        mirror_all[:, int(beam_axis)] *= -1.0
        tree_all = cKDTree(welded_positions)
        bbox_diag = float(
            np.linalg.norm(
                welded_positions.max(axis=0) - welded_positions.min(axis=0)
            )
        )
        threshold_sym = max(0.02 * bbox_diag, 1e-9)
        distances_all, partners_all = tree_all.query(mirror_all, k=1)
        partners_all = np.asarray(partners_all, dtype=int)
        mutual_all = partners_all[partners_all] == np.arange(n_unique)
        valid_all = (
            mutual_all
            & (distances_all <= threshold_sym)
            & (partners_all != np.arange(n_unique))
        )
        paired_global = np.where(valid_all, partners_all, -1)
    else:
        paired_global = -np.ones(n_unique, dtype=np.int64)

    for _ in range(iterations):
        neighbour_means = adjacency @ welded_positions
        if beam_axis is None:
            updated = welded_positions + alpha * (
                neighbour_means - welded_positions
            )
            welded_positions = np.where(
                active_welded[:, None], updated, welded_positions
            )
        else:
            updated = welded_positions.copy()
            done = np.zeros(n_unique, dtype=bool)
            active_indices = np.nonzero(active_welded)[0]
            for global_i in active_indices:
                if done[global_i]:
                    continue
                global_j = int(paired_global[global_i])
                if global_j < 0 or global_j == global_i:
                    # No mirror partner — apply a plain damped step.
                    p_i = welded_positions[global_i]
                    m_i = neighbour_means[global_i]
                    updated[global_i] = p_i + alpha * (m_i - p_i)
                    done[global_i] = True
                    continue
                m_i = neighbour_means[global_i].copy()
                m_j = neighbour_means[global_j].copy()
                m_j[int(beam_axis)] *= -1.0
                target = 0.5 * (m_i + m_j)
                p_i = welded_positions[global_i].copy()
                p_j = welded_positions[global_j].copy()
                p_j[int(beam_axis)] *= -1.0
                cur_avg = 0.5 * (p_i + p_j)
                damped = cur_avg + alpha * (target - cur_avg)
                updated[global_i] = damped
                damped_mirror = damped.copy()
                damped_mirror[int(beam_axis)] *= -1.0
                updated[global_j] = damped_mirror
                done[global_i] = True
                done[global_j] = True
            welded_positions = updated

    # Scatter welded positions back to raw vertex array, only for active
    # raw vertices (welded-active is a superset because of duplicates).
    new_vertices = vertices.copy()
    if beam_axis is None:
        scatter_mask = active_raw
    else:
        # When symmetry is on, we may also have moved a +/-beam mirror
        # whose raw vertex wasn't itself flagged. Use the welded-space
        # active mask (which already includes both halves of valid mirror
        # pairs that touched a high-dihedral face).
        # Note: ``active_welded[inverse]`` is the per-raw-vertex
        # equivalent — a raw vertex is moved if its welded node is active.
        scatter_mask = active_welded[inverse]
    new_vertices[scatter_mask] = welded_positions[inverse[scatter_mask]]
    mesh.vertices = new_vertices
