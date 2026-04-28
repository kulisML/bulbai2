"""A/B compare BulbFFDDeformer legacy (hard cutoff, no sym, no Taubin) vs
defaults (smooth blend + mirror symmetry + Taubin).

Uses a single fixed Kracht vector so the only variable is the deformer
configuration. Outputs two STLs plus metrics side-by-side.

Usage:
    python scripts/ab_compare_deformer.py docs/base_hull.stl
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import trimesh


def main() -> int:
    source = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/base_hull.stl")
    mesh = trimesh.load(source, force="mesh")
    extents = mesh.extents.astype(float)
    primary = int(extents.argmax())
    axis_values = mesh.vertices[:, primary]
    region_depth = float(extents[primary]) * 0.15
    region = {
        "axis_index": primary,
        "axis_min": float(axis_values.max()) - region_depth,
        "axis_max": float(axis_values.max()),
    }

    from bulbopt.optimization.parametric.ffd_deformer import BulbFFDDeformer
    from bulbopt.optimization.parametric.kracht_space import KrachtVector

    vector = KrachtVector(
        values={
            "length_ratio":     0.035,
            "breadth_ratio":    0.12,
            "height_ratio":     0.45,
            "axis_z_ratio":     0.30,
            "longitudinal_pos": 0.65,
            "cross_section_c":  0.70,
            "volume_coef":      0.65,
            "nose_sharpness":   0.40,
        }
    )

    # Monkey-patch: "legacy" deformer simulates the old behaviour by
    # setting BLEND_WIDTH_FRACTION=0, symmetry off, Taubin off.
    legacy = BulbFFDDeformer(
        force_port_starboard_symmetry=False,
        post_smoothing_iterations=0,
    )
    legacy.BLEND_WIDTH_FRACTION = 0.0  # type: ignore[attr-defined]
    improved = BulbFFDDeformer()  # defaults: blend 10%, symmetry on, Taubin 3x

    legacy_mesh = legacy.deform(mesh, region, vector)
    improved_mesh = improved.deform(mesh, region, vector)

    snap_dir = Path("mesh_quality_snapshots")
    snap_dir.mkdir(exist_ok=True)
    legacy_path = snap_dir / "ab_legacy.stl"
    improved_path = snap_dir / "ab_improved.stl"
    legacy_path.write_bytes(trimesh.exchange.stl.export_stl(legacy_mesh))
    improved_path.write_bytes(trimesh.exchange.stl.export_stl(improved_mesh))

    # Metrics
    from scipy.spatial import cKDTree

    def mirror_rms(m: trimesh.Trimesh, beam_axis: int) -> float:
        v = np.asarray(m.vertices, dtype=float)
        mirror = v.copy()
        mirror[:, beam_axis] *= -1.0
        tree = cKDTree(v)
        d, _ = tree.query(mirror, k=1)
        return float(np.sqrt(np.mean(d ** 2)))

    def seam_step(m: trimesh.Trimesh, primary_axis: int, axis_min: float) -> float:
        """Max vertex-displacement discontinuity across the axis_min plane:
        for each vertex within ±1% of axis_min, compare its neighbour
        displacement stats. Simpler proxy: stdev of displacement around
        the boundary band."""
        v = np.asarray(m.vertices, dtype=float)
        base = np.asarray(mesh.vertices, dtype=float)
        disp = np.linalg.norm(v - base, axis=1)
        axis_val = v[:, primary_axis]
        span = region["axis_max"] - region["axis_min"]
        band = (axis_val >= region["axis_min"] - 0.02 * span) & (
            axis_val <= region["axis_min"] + 0.02 * span
        )
        if not band.any():
            return 0.0
        return float(np.std(disp[band]))

    beam = [i for i in range(3) if i != primary][0]
    buf = io.StringIO()
    buf.write("Metric                        Legacy      Improved    Delta\n")
    buf.write("-" * 68 + "\n")
    legacy_vol = abs(legacy_mesh.volume)
    improved_vol = abs(improved_mesh.volume)
    legacy_area = legacy_mesh.area
    improved_area = improved_mesh.area
    legacy_mirror = mirror_rms(legacy_mesh, beam)
    improved_mirror = mirror_rms(improved_mesh, beam)
    legacy_seam = seam_step(legacy_mesh, primary, region["axis_min"])
    improved_seam = seam_step(improved_mesh, primary, region["axis_min"])
    buf.write(f"Vertices                      {len(legacy_mesh.vertices):<11} {len(improved_mesh.vertices):<11} {len(improved_mesh.vertices) - len(legacy_mesh.vertices):+d}\n")
    buf.write(f"Faces                         {len(legacy_mesh.faces):<11} {len(improved_mesh.faces):<11} {len(improved_mesh.faces) - len(legacy_mesh.faces):+d}\n")
    buf.write(f"Watertight                    {legacy_mesh.is_watertight:<11} {improved_mesh.is_watertight:<11}\n")
    buf.write(f"Volume (m^3)                  {legacy_vol:<11.2f} {improved_vol:<11.2f} {100.0*(improved_vol-legacy_vol)/legacy_vol:+.2f}%\n")
    buf.write(f"Surface area (m^2)            {legacy_area:<11.2f} {improved_area:<11.2f} {100.0*(improved_area-legacy_area)/legacy_area:+.2f}%\n")
    buf.write(f"Mirror RMS (m)                {legacy_mirror:<11.4f} {improved_mirror:<11.4f} {100.0*(improved_mirror-legacy_mirror)/max(legacy_mirror,1e-9):+.2f}%\n")
    buf.write(f"Seam displacement std (m)     {legacy_seam:<11.4f} {improved_seam:<11.4f} {100.0*(improved_seam-legacy_seam)/max(legacy_seam,1e-9):+.2f}%\n")
    sys.stdout.buffer.write(buf.getvalue().encode("utf-8"))
    sys.stdout.buffer.write(f"\nSTLs:\n  legacy   -> {legacy_path}\n  improved -> {improved_path}\n".encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
