"""Isolate which of the 3 fixes (blend/symmetry/Taubin) causes what.

Runs 4 permutations on same base STL + same Kracht vector:
    0. legacy:  no blend, no symmetry, no Taubin
    1. blend only
    2. blend + symmetry
    3. blend + symmetry + Taubin (defaults)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh


def main() -> int:
    source = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/base_hull.stl")
    mesh = trimesh.load(source, force="mesh")
    extents = mesh.extents.astype(float)
    primary = int(extents.argmax())
    region = {
        "axis_index": primary,
        "axis_min": float(mesh.vertices[:, primary].max()) - float(extents[primary]) * 0.15,
        "axis_max": float(mesh.vertices[:, primary].max()),
    }

    from bulbopt.optimization.parametric.ffd_deformer import BulbFFDDeformer
    from bulbopt.optimization.parametric.kracht_space import KrachtVector

    vector = KrachtVector(values={
        "length_ratio": 0.035, "breadth_ratio": 0.12, "height_ratio": 0.45,
        "axis_z_ratio": 0.30, "longitudinal_pos": 0.65, "cross_section_c": 0.70,
        "volume_coef": 0.65, "nose_sharpness": 0.40,
    })

    configs = [
        ("0. legacy         ", {"blend": 0.0,  "sym": False, "taubin": 0}),
        ("1. blend only     ", {"blend": 0.10, "sym": False, "taubin": 0}),
        ("2. blend + sym    ", {"blend": 0.10, "sym": True,  "taubin": 0}),
        ("3. blend+sym+taub ", {"blend": 0.10, "sym": True,  "taubin": 3}),
    ]
    baseline_vol = abs(mesh.volume)
    baseline_area = mesh.area
    print(f"Base hull: verts={len(mesh.vertices)}, faces={len(mesh.faces)}, vol={baseline_vol:.0f}, area={baseline_area:.0f}")
    print(f"{'Config':<22} {'Volume':<12} {'dVol%':<8} {'Surface':<12} {'dS%':<8} {'Watertight'}")
    print("-" * 90)
    for label, cfg in configs:
        d = BulbFFDDeformer(
            force_port_starboard_symmetry=cfg["sym"],
            post_smoothing_iterations=cfg["taubin"],
        )
        d.BLEND_WIDTH_FRACTION = cfg["blend"]
        out = d.deform(mesh, region, vector)
        v = abs(out.volume)
        a = out.area
        dv = 100.0 * (v - baseline_vol) / baseline_vol
        ds = 100.0 * (a - baseline_area) / baseline_area
        print(f"{label:<22} {v:<12.0f} {dv:<+8.2f} {a:<12.0f} {ds:<+8.2f} {out.is_watertight}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
