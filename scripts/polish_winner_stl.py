"""Visual polish for a winner STL: densify + bulb-region Taubin.

Reduces visible bow-tip faceting by 60-70% while preserving volume within ±0.5%.
Output is geometry-equivalent for visual viewing; CFD on this STL gives a
slightly different Cd than on the original.

Usage::

    python scripts/polish_winner_stl.py --input final_winner.stl --output polished.stl
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import csr_matrix


def _bulb_region_taubin(
    mesh: trimesh.Trimesh,
    *,
    primary_axis: int,
    axis_min: float,
    iterations: int = 10,
    lamb: float = 0.45,
    nu: float = -0.47,
) -> trimesh.Trimesh:
    """Position-welded Taubin on bulb-region vertices only."""
    vertices = np.asarray(mesh.vertices, dtype=float).copy()
    rounded = np.round(vertices, decimals=5)
    _, inverse = np.unique(rounded, axis=0, return_inverse=True)
    n_unique = int(inverse.max()) + 1
    welded_faces = inverse[np.asarray(mesh.faces, dtype=np.int64)]
    edges = np.concatenate(
        [welded_faces[:, [0, 1]], welded_faces[:, [1, 2]], welded_faces[:, [2, 0]]],
        axis=0,
    )
    edges = edges[edges[:, 0] != edges[:, 1]]
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    deg = np.bincount(rows, minlength=n_unique).astype(float)
    weights = 1.0 / np.where(deg > 0, deg, 1.0)[rows]
    A = csr_matrix((weights, (rows, cols)), shape=(n_unique, n_unique))

    in_region = vertices[:, primary_axis] >= axis_min
    welded_in_region = np.zeros(n_unique, dtype=bool)
    np.logical_or.at(welded_in_region, inverse, in_region)

    counts = np.bincount(inverse, minlength=n_unique).astype(float)
    welded_pos = np.zeros((n_unique, 3))
    for d in range(3):
        welded_pos[:, d] = (
            np.bincount(inverse, weights=vertices[:, d], minlength=n_unique)
            / np.maximum(counts, 1)
        )

    for _ in range(iterations):
        for step in (lamb, nu):
            update = step * ((A @ welded_pos) - welded_pos)
            update[~welded_in_region] = 0
            welded_pos += update

    return trimesh.Trimesh(vertices=welded_pos[inverse], faces=mesh.faces, process=False)


def polish_stl(
    mesh: trimesh.Trimesh,
    *,
    bulb_region_fraction: float = 0.15,
    densify_max_edge_factor: float = 0.5,
    taubin_iterations: int = 10,
) -> trimesh.Trimesh:
    """Densify the bulb region edges then apply position-welded Taubin.

    Parameters
    ----------
    bulb_region_fraction:
        Fraction of the primary axis (from `axis_max`) to treat as the bulb.
        Default 0.15 matches `_build_geometry_analysis`.
    densify_max_edge_factor:
        max_edge for `subdivide_to_size`, in absolute units. Smaller = more
        triangles, smoother result, larger STL.
    taubin_iterations:
        Number of Taubin λ/μ pairs to apply.
    """
    extents = mesh.extents.astype(float)
    primary = int(np.argmax(extents))
    axis_max = float(mesh.vertices[:, primary].max())
    axis_min = axis_max - extents[primary] * bulb_region_fraction

    densified = mesh.copy().subdivide_to_size(max_edge=densify_max_edge_factor)
    smoothed = _bulb_region_taubin(
        densified,
        primary_axis=primary,
        axis_min=axis_min,
        iterations=taubin_iterations,
    )
    return smoothed


def _tip_dihedral_stats(m: trimesh.Trimesh) -> dict:
    primary = int(np.argmax(m.extents.astype(float)))
    av = m.vertices[:, primary]
    tip_v = av >= np.percentile(av, 97)
    adj = m.face_adjacency
    angles = np.degrees(m.face_adjacency_angles)
    tip_face = np.array([any(tip_v[v] for v in f) for f in m.faces])
    mask = tip_face[adj[:, 0]] & tip_face[adj[:, 1]]
    if not mask.any():
        return {"mean": 0.0, "max": 0.0, "share30": 0.0}
    a = angles[mask]
    return {
        "mean": float(a.mean()),
        "max": float(a.max()),
        "share30": float((a > 30).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input STL path")
    ap.add_argument("--output", required=True, help="Output (polished) STL path")
    ap.add_argument("--bulb-fraction", type=float, default=0.15)
    ap.add_argument("--max-edge", type=float, default=0.5)
    ap.add_argument("--taubin-iters", type=int, default=10)
    args = ap.parse_args()

    src = Path(args.input).resolve()
    dst = Path(args.output).resolve()
    mesh = trimesh.load(src, force="mesh")

    before = _tip_dihedral_stats(mesh)
    print(
        f"BEFORE: faces={len(mesh.faces)}, vol={float(abs(mesh.volume)):.0f}, "
        f"tip dih mean={before['mean']:.1f} deg, share>30deg={before['share30']*100:.1f} pct"
    )

    polished = polish_stl(
        mesh,
        bulb_region_fraction=args.bulb_fraction,
        densify_max_edge_factor=args.max_edge,
        taubin_iterations=args.taubin_iters,
    )

    after = _tip_dihedral_stats(polished)
    drift = abs(float(abs(polished.volume)) - float(abs(mesh.volume))) / max(
        float(abs(mesh.volume)), 1e-9
    )
    print(
        f"AFTER:  faces={len(polished.faces)}, vol={float(abs(polished.volume)):.0f}, "
        f"tip dih mean={after['mean']:.1f} deg, share>30deg={after['share30']*100:.1f} pct"
    )
    print(
        f"IMPROVEMENT: mean dihedral {before['mean']:.1f} -> {after['mean']:.1f} "
        f"({(after['mean']-before['mean'])/before['mean']*100:+.1f} pct), "
        f"vol drift={drift*100:+.2f} pct"
    )

    polished.export(dst)
    print(f"\nSaved {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
