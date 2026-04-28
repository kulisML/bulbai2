"""Compare two bulb STLs on mesh-quality metrics.

Usage:
    python scripts/compare_mesh_quality.py <pre.stl> <post.stl>

Prints a side-by-side table. Run manually after night-run to see whether
the 2026-04-23 mesh-quality fixes (smooth blend, symmetry enforcement,
Taubin smoothing) actually improved the output.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh


def _mirror_rms(mesh: trimesh.Trimesh, beam_axis: int) -> float:
    """RMS distance between every vertex and its closest mirror partner."""
    from scipy.spatial import cKDTree

    verts = np.asarray(mesh.vertices, dtype=float)
    mirrors = verts.copy()
    mirrors[:, beam_axis] *= -1.0
    tree = cKDTree(verts)
    distances, _ = tree.query(mirrors, k=1)
    return float(np.sqrt(np.mean(distances ** 2)))


def _max_seam_dihedral(mesh: trimesh.Trimesh, primary_axis: int) -> float:
    """Max dihedral angle (degrees) between pairs of faces where one
    straddles the axis_min plane. Proxy for the welding-seam kink."""
    verts = np.asarray(mesh.vertices, dtype=float)
    axis_min = float(verts[:, primary_axis].min())
    axis_max = float(verts[:, primary_axis].max())
    candidate_plane = axis_max - 0.25 * (axis_max - axis_min)

    # Face normals
    mesh_copy = mesh.copy()
    mesh_copy.process()
    face_normals = mesh_copy.face_normals
    face_vert_means = mesh_copy.vertices[mesh_copy.faces].mean(axis=1)
    in_band = np.abs(face_vert_means[:, primary_axis] - candidate_plane) < 0.02 * (axis_max - axis_min)
    normals_band = face_normals[in_band]
    if len(normals_band) < 2:
        return 0.0
    max_angle = 0.0
    # Random pair sample
    rng = np.random.default_rng(0)
    indices = rng.integers(0, len(normals_band), size=min(500, len(normals_band) ** 2))
    for i in indices:
        for j in indices:
            if i >= j:
                continue
            cos = np.clip(float(np.dot(normals_band[i], normals_band[j])), -1.0, 1.0)
            angle_deg = float(np.degrees(np.arccos(cos)))
            if angle_deg > max_angle:
                max_angle = angle_deg
    return max_angle


def _summarise(label: str, mesh: trimesh.Trimesh) -> None:
    extents = mesh.extents.astype(float)
    primary = int(extents.argmax())
    beam = [i for i in range(3) if i != primary][0]
    print(f"\n=== {label} ===")
    print(f"Vertices:       {len(mesh.vertices)}")
    print(f"Faces:          {len(mesh.faces)}")
    print(f"Watertight:     {mesh.is_watertight}")
    print(f"Volume (m³):    {abs(mesh.volume):.4f}")
    print(f"Surface area:   {mesh.area:.4f}")
    print(f"Extents:        ({extents[0]:.3f}, {extents[1]:.3f}, {extents[2]:.3f})")
    print(f"Primary axis:   {primary}")
    print(f"Beam axis:      {beam}")
    print(f"Mirror RMS:     {_mirror_rms(mesh, beam):.6f}  (lower = more symmetric)")
    print(f"Seam dihedral:  {_max_seam_dihedral(mesh, primary):.2f}°  (lower = smoother)")


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    pre_path = Path(sys.argv[1])
    post_path = Path(sys.argv[2])
    if not pre_path.exists() or not post_path.exists():
        print(f"Missing: {pre_path} or {post_path}")
        return 1
    pre = trimesh.load(pre_path, force="mesh")
    post = trimesh.load(post_path, force="mesh")
    _summarise(f"PRE  ({pre_path.name})", pre)
    _summarise(f"POST ({post_path.name})", post)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
