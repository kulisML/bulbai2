"""Unit tests for :mod:`bulbopt.optimization.parametric.anisotropic_remesh`.

The module exposes :func:`remesh_region`, which uses ACVD-based isotropic
remeshing (via :mod:`pyacvd`) to lower the aspect ratio of triangles in
the bulb region while leaving the out-of-region faces untouched. When
``pyacvd`` is not installed the function degrades gracefully to a no-op
so the rest of the optimisation pipeline continues to work on minimal
laptop installs.
"""
from __future__ import annotations

import numpy as np
import pytest
import trimesh

from bulbopt.optimization.parametric import anisotropic_remesh
from bulbopt.optimization.parametric.anisotropic_remesh import (
    is_anisotropic_remesh_available,
    remesh_region,
)


def _slivery_box() -> tuple[trimesh.Trimesh, dict]:
    """Return a watertight box mesh whose bulb half is intentionally
    populated with high-aspect-ratio sliver triangles.

    The base box is subdivided three times (≈768 faces) and then the
    half of the mesh on the positive ``axis_index`` side is collapsed
    along the Z axis by a factor of 10. That produces triangles whose
    Z-edge is 10× shorter than their X- and Y-edges, i.e. aspect ratio
    in the 30-50 range — comparable to the slivers `pyacvd` is meant
    to repair on the real bow tip.
    """
    mesh = trimesh.creation.box(extents=(10.0, 4.0, 4.0))
    mesh = mesh.subdivide().subdivide().subdivide()
    primary = int(np.argmax(mesh.extents))
    axis_min = float(mesh.vertices[:, primary].mean())

    verts = mesh.vertices.copy()
    region_mask = mesh.vertices[:, primary] >= axis_min
    verts[region_mask, 2] *= 0.1  # collapse z by 10×

    slivery = trimesh.Trimesh(vertices=verts, faces=mesh.faces.copy(), process=False)
    region = {"axis_index": primary, "axis_min": axis_min}
    return slivery, region


def _max_aspect_in_region(mesh: trimesh.Trimesh, region: dict) -> float:
    """Return the largest (longest_edge / shortest_edge) ratio over all
    triangles whose face has at least one vertex in the region.
    """
    primary = region["axis_index"]
    axis_min = region["axis_min"]
    in_region = (mesh.vertices[mesh.faces, primary] >= axis_min).any(axis=1)
    if not in_region.any():
        return 1.0
    tris = mesh.vertices[mesh.faces[in_region]]
    e0 = np.linalg.norm(tris[:, 1] - tris[:, 0], axis=1)
    e1 = np.linalg.norm(tris[:, 2] - tris[:, 1], axis=1)
    e2 = np.linalg.norm(tris[:, 0] - tris[:, 2], axis=1)
    edges = np.stack([e0, e1, e2], axis=1)
    return float(np.max(edges.max(axis=1) / np.maximum(edges.min(axis=1), 1e-12)))


def test_no_op_when_pyacvd_unavailable(monkeypatch) -> None:
    """If ``pyacvd`` cannot be imported, ``remesh_region`` must return
    the input mesh unchanged.

    We simulate the missing dependency by monkeypatching the module's
    availability flag. The function must short-circuit without touching
    pyvista / pyacvd, so this test runs even on systems where the
    optional deps *are* installed.
    """
    monkeypatch.setattr(anisotropic_remesh, "_PYACVD_AVAILABLE", False)
    mesh, region = _slivery_box()
    out = remesh_region(mesh, region)
    assert out is mesh, (
        "When pyacvd is unavailable, remesh_region must return the input "
        "mesh object unchanged (no copy, no remesh)."
    )


def test_remesh_lowers_max_aspect_ratio_in_region() -> None:
    """ACVD remeshing must lower the worst-case in-region aspect ratio
    by more than 50%.

    Without :mod:`pyacvd` installed the test is skipped so a laptop
    without optional deps can still run the unit suite.
    """
    if not is_anisotropic_remesh_available():
        pytest.skip("pyacvd is not installed; skipping ACVD remesh test")

    mesh, region = _slivery_box()
    before = _max_aspect_in_region(mesh, region)
    assert before > 5.0, "test fixture must actually contain slivers"

    out = remesh_region(mesh, region)
    after = _max_aspect_in_region(out, region)
    drop = (before - after) / before
    assert drop > 0.5, (
        f"Expected >50% drop in max aspect ratio after ACVD remesh, "
        f"got before={before:.2f}, after={after:.2f}, drop={drop:.1%}"
    )


def test_remesh_preserves_volume_within_one_percent() -> None:
    """The remesh must not change the enclosed volume by more than 1%.

    ACVD clusters vertices onto a uniform grid then re-triangulates;
    the boundary between the in-region submesh and the rest of the hull
    is preserved by snapping nearby new vertices back onto the original
    boundary positions. Volume drift should therefore stay well below
    1% even on aggressively slivered fixtures.

    Skipped on systems without :mod:`pyacvd` so the test suite still
    runs there.
    """
    if not is_anisotropic_remesh_available():
        pytest.skip("pyacvd is not installed; skipping ACVD volume test")

    mesh, region = _slivery_box()
    out = remesh_region(mesh, region)
    drift = abs(out.volume - mesh.volume) / abs(mesh.volume)
    assert drift < 0.01, (
        f"Expected <1% volume drift after ACVD remesh, "
        f"got input.volume={mesh.volume:.4f}, "
        f"output.volume={out.volume:.4f}, drift={drift:.2%}"
    )
