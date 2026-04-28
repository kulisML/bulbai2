"""Integration tests: FFD on a parametric Wigley hull.

Audit C 2026-04-26 (item Add #7): every existing FFD/quality unit test
uses ``trimesh.creation.icosphere`` or ``trimesh.creation.box``. Real ship
hulls have asymmetric draft, sharp keel edges, and non-uniform panel
density — none of which are exercised by the synthetic-shape corpus.
These tests apply the deformer to the closed-form ITTC Wigley hull
(``tests/fixtures/wigley_hull.py``) so we can detect regressions on
real-shape topology before they hit the night-run CFD.
"""
from __future__ import annotations

import numpy as np
import trimesh

from bulbopt.optimization.parametric.ffd_deformer import BulbFFDDeformer
from bulbopt.optimization.parametric.kracht_space import KrachtVector
from bulbopt.optimization.quality.mesh_metrics import compute_mesh_quality

from tests.fixtures.wigley_hull import make_wigley_hull


# ---- Helpers ---------------------------------------------------------------


def _bulb_region(mesh: trimesh.Trimesh, fraction: float = 0.20) -> dict:
    """Build a bulb region dict for the forward ``fraction`` of a Wigley hull.

    The Wigley hull is built with the longitudinal axis along world X, the
    symmetric (port-starboard) axis along world Y, and the asymmetric
    (keel-deck) axis along world Z. We pin those explicitly so the FFD
    deformer's beam/draft heuristic can't pick the wrong axis on a hull
    where the Y and Z extents are similar (B=10, T=6.25 → close enough
    that ``argmax``/``argmin`` could go either way under jitter).
    """
    extents = mesh.extents.astype(float)
    primary_axis = int(np.argmax(extents))  # X — longest
    axis_max = float(mesh.vertices[:, primary_axis].max())
    axis_min_full = float(mesh.vertices[:, primary_axis].min())
    region_depth = (axis_max - axis_min_full) * fraction
    return {
        "axis_index": primary_axis,
        "axis_min": axis_max - region_depth,
        "axis_max": axis_max,
        # Wigley convention: ±Y mirror, Z draft.
        "beam_axis": 1,
        "draft_axis": 2,
    }


def _bulb_region_volume(mesh: trimesh.Trimesh, region: dict) -> float:
    """Convex-hull volume of all vertices forward of ``region['axis_min']``.

    The bulb sub-mesh extracted from the full hull is open at the seam
    where the FFD boundary cuts it, so a direct ``submesh.volume`` is
    undefined. The convex hull is closed by construction and a robust
    monotone proxy for the bulb volume — exactly the shape signal we
    care about for a ``volume_coef`` regression test.
    """
    pa = int(region["axis_index"])
    am = float(region["axis_min"])
    in_region = mesh.vertices[:, pa] >= am
    pts = np.asarray(mesh.vertices[in_region], dtype=float)
    if len(pts) < 4:
        return 0.0
    try:
        return float(trimesh.convex.convex_hull(pts).volume)
    except Exception:
        return 0.0


def _rms_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Vertex-to-vertex RMS (assumes equal length, same indexing)."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


# ---- Tests -----------------------------------------------------------------


def test_wigley_hull_is_watertight() -> None:
    """Sanity: the fixture builder produces a closed manifold mesh.

    Watertightness is the prerequisite for *every* downstream test —
    Taubin smoothing, FFD deformation, and the mesh-quality metric all
    depend on a closed surface. If this assertion ever flips, every
    later test in this module is meaningless.
    """
    mesh = make_wigley_hull()

    assert isinstance(mesh, trimesh.Trimesh)
    assert mesh.is_watertight, "Wigley fixture must be watertight"
    assert mesh.is_winding_consistent, "Wigley fixture must have consistent winding"
    assert mesh.volume > 0.0, "Wigley fixture must enclose a positive volume"
    # A 100 m / 10 m / 6.25 m hull's analytic underwater volume is
    # 4*B*L*T/9 = 2777.78 m³; the discrete mesh approximates this within
    # ~1% at n_x=40, n_z=20.
    assert 2700.0 < mesh.volume < 2800.0, (
        f"Wigley volume {mesh.volume:.2f} outside expected ~2778 m³ band"
    )


def test_ffd_on_wigley_produces_different_meshes_for_different_vectors() -> None:
    """Two different Kracht vectors must produce two clearly-different
    deformed bulbs that both stay watertight + winding-consistent.

    This is the headline guarantee of the FFD layer: the night-run is a
    no-op if the deformer can't actually distinguish two design
    proposals. The thresholds are deliberately loose (RMS > 0.001 m
    from baseline; > 0.01 m between the two outputs) so any small bug
    that collapses the mapping will trip them.
    """
    mesh = make_wigley_hull()
    region = _bulb_region(mesh)

    # Kracht-A: emphasize breadth (push the bulb laterally).
    vec_breadth = KrachtVector(
        values={
            "length_ratio":     0.020,
            "breadth_ratio":    0.18,   # high
            "height_ratio":     0.30,
            "axis_z_ratio":     0.25,
            "longitudinal_pos": 0.5,
            "cross_section_c":  0.7,
            "volume_coef":      0.50,
            "nose_sharpness":   0.5,
        }
    )
    # Kracht-B: emphasize length (push the bulb forward).
    vec_length = KrachtVector(
        values={
            "length_ratio":     0.040,  # high
            "breadth_ratio":    0.04,
            "height_ratio":     0.30,
            "axis_z_ratio":     0.25,
            "longitudinal_pos": 0.5,
            "cross_section_c":  0.7,
            "volume_coef":      0.50,
            "nose_sharpness":   0.5,
        }
    )

    deformer = BulbFFDDeformer(adaptive_subdivision=False)
    out_breadth = deformer.deform(mesh, region, vec_breadth)
    out_length = deformer.deform(mesh, region, vec_length)

    # Both deformed meshes must remain valid surfaces.
    assert out_breadth.is_watertight, "breadth-emphasis deformation broke watertightness"
    assert out_length.is_watertight, "length-emphasis deformation broke watertightness"
    assert out_breadth.is_winding_consistent, "breadth-emphasis flipped winding"
    assert out_length.is_winding_consistent, "length-emphasis flipped winding"

    # The deformer's post_repair pass calls ``merge_vertices`` which can
    # fold a single coincident pair on the centreline; tolerate a one-
    # vertex diff and align by closest-point if so. In practice both
    # outputs have the same vertex count so the cheap path runs.
    assert len(out_breadth.vertices) == len(out_length.vertices), (
        "two deformations must produce mesh arrays of the same length so "
        "the RMS comparison is meaningful — got "
        f"{len(out_breadth.vertices)} vs {len(out_length.vertices)}"
    )

    # Deformations must measurably move vertices vs the baseline.
    # Compare on whichever vertex count survived the merge_vertices pass.
    if len(out_breadth.vertices) == len(mesh.vertices):
        rms_baseline_breadth = _rms_distance(mesh.vertices, out_breadth.vertices)
        rms_baseline_length = _rms_distance(mesh.vertices, out_length.vertices)
    else:
        # Fall back to closest-point RMS using a KDTree.
        from scipy.spatial import cKDTree

        tree = cKDTree(mesh.vertices)
        d_b, _ = tree.query(out_breadth.vertices, k=1)
        d_l, _ = tree.query(out_length.vertices, k=1)
        rms_baseline_breadth = float(np.sqrt(np.mean(np.asarray(d_b) ** 2)))
        rms_baseline_length = float(np.sqrt(np.mean(np.asarray(d_l) ** 2)))

    assert rms_baseline_breadth > 0.001, (
        f"breadth-emphasis deformation barely moved vertices "
        f"(RMS {rms_baseline_breadth:.6f} m, threshold 0.001 m)"
    )
    assert rms_baseline_length > 0.001, (
        f"length-emphasis deformation barely moved vertices "
        f"(RMS {rms_baseline_length:.6f} m, threshold 0.001 m)"
    )

    # The two outputs must differ from each other meaningfully.
    rms_between = _rms_distance(out_breadth.vertices, out_length.vertices)
    assert rms_between > 0.01, (
        f"two clearly-different Kracht vectors produced near-identical "
        f"bulbs (RMS {rms_between:.6f} m, threshold 0.01 m)"
    )


def test_ffd_on_wigley_respects_negative_volume_coef() -> None:
    """A negative ``volume_coef`` must shrink the bulb relative to a
    positive one with the same other parameters.

    Note: the FFD's offset multiplier is ``0.5 + volume_coef``, so
    ``volume_coef = -0.30`` gives 0.20× scaling and ``volume_coef = +0.50``
    gives 1.00× scaling — both are still positive multipliers, so the
    bulb always GROWS relative to the undeformed Wigley hull. The
    physically meaningful "deflate" semantic is therefore *less growth
    than the inflated case*, not absolute shrinkage. This test asserts
    that semantic on a real-shape hull, with all other Kracht parameters
    held identical so volume_coef is the only variable.
    """
    mesh = make_wigley_hull()
    region = _bulb_region(mesh)

    common_values = {
        "length_ratio":     0.020,
        "breadth_ratio":    0.05,
        "height_ratio":     0.30,
        "axis_z_ratio":     0.25,
        "longitudinal_pos": 0.5,
        "cross_section_c":  0.7,
        "nose_sharpness":   0.5,
    }
    vec_deflate = KrachtVector(
        values={**common_values, "volume_coef": -0.30}
    )
    vec_inflate = KrachtVector(
        values={**common_values, "volume_coef": +0.50}
    )

    deformer = BulbFFDDeformer(adaptive_subdivision=False)
    deflated = deformer.deform(mesh, region, vec_deflate)
    inflated = deformer.deform(mesh, region, vec_inflate)

    assert deflated.is_watertight, "deflated mesh broke watertightness"
    assert inflated.is_watertight, "inflated mesh broke watertightness"

    vol_deflate = _bulb_region_volume(deflated, region)
    vol_inflate = _bulb_region_volume(inflated, region)

    assert vol_deflate < vol_inflate, (
        f"negative volume_coef produced a LARGER bulb than the inflated "
        f"case: deflate={vol_deflate:.4f} m³, inflate={vol_inflate:.4f} m³"
    )
    # Sanity: the gap should be non-trivial (>5% of the inflated volume)
    # so we know the test actually exercises the parameter.
    assert (vol_inflate - vol_deflate) / max(vol_inflate, 1e-9) > 0.05, (
        f"deflate→inflate gap too small to be a meaningful regression "
        f"signal: deflate={vol_deflate:.4f}, inflate={vol_inflate:.4f}"
    )


def test_ffd_on_wigley_does_not_intersect_aft_hull() -> None:
    """A strong forward push must not drive the bulb sub-mesh through
    the rest of the hull.

    After the audit-C B3-2 fix the L2 quality score includes a self-
    intersection penalty in [0, 1] that adds on top of the classical
    components (dihedral, symmetry, watertight). On a clean deformation
    that penalty stays at 0; if the FFD pushes the bulb tip past the
    aft half-space, the penalty grows. We use a high ``length_ratio``
    + high ``longitudinal_pos`` combo (the regime that historically
    caused the regression) and assert the total quality score is below
    a generous 1.0 sanity threshold.

    We use ``length_ratio = 0.045`` (the upper bound of the design
    space — the spec asked for 0.05 but the bound clips there) and
    ``longitudinal_pos = 0.95`` to put the strongest push into the
    forward-most lattice slab.
    """
    mesh = make_wigley_hull()
    region = _bulb_region(mesh)

    vec_strong_push = KrachtVector(
        values={
            "length_ratio":     0.045,  # design-space upper bound
            "breadth_ratio":    0.05,
            "height_ratio":     0.30,
            "axis_z_ratio":     0.25,
            "longitudinal_pos": 0.95,
            "cross_section_c":  0.7,
            "volume_coef":      0.50,
            "nose_sharpness":   0.5,
        }
    )

    deformer = BulbFFDDeformer(adaptive_subdivision=False)
    pushed = deformer.deform(mesh, region, vec_strong_push)

    assert pushed.is_watertight, "strong forward push broke watertightness"

    score = compute_mesh_quality(
        pushed, beam_axis=region["beam_axis"], region=region
    )
    assert np.isfinite(score), "mesh quality must be finite"
    assert score < 1.0, (
        f"strong forward push produced a quality score of {score:.4f} — "
        "the bulb is likely intersecting the aft hull or has gross "
        "dihedral/symmetry defects"
    )
