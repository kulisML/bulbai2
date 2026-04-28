# BulbOpt — Bulb Mesh Quality Design

**Date:** 2026-04-23
**Status:** In execution
**Supersedes:** additive to `2026-04-22-bulbopt-night-optimization-design.md` §4.2

---

## 1. Problem

First real night-run produced deformed bulbs with three visible artifacts:

1. **Welding seam** — a vertical step where the deformed bulb region joins the
   unchanged rest of the hull (C⁰ discontinuity). Visible in the side view as
   a sharp cliff about 30 cm tall between the smooth nose and the cylindrical
   hull continuation.

2. **Asymmetric bulb** — port and starboard differ. Close-up of the underside
   shows a twisting crease that should not exist on a ship hull
   mirror-symmetric around the centerline.

3. **Polygonal facets** — the bulb surface has visible flat triangles instead
   of a smooth curve. The baseline mesh has finite resolution; FFD moves
   existing vertices but does not create new ones, so any insufficiently
   triangulated region becomes a polygonal shell after deformation.

The night-run itself is engineering-honest — NSGA-II converged to Cd=0.4003
for the best candidate — but no engineer would accept the *shape* for
production. We need a cleaner geometry pipeline before the optimization.

---

## 2. Root causes (code audit)

### 2.1 Welding seam → hard boundary cutoff

`src/bulbopt/optimization/parametric/ffd_deformer.py:59`:

```python
in_region = mesh.vertices[:, primary_axis] >= axis_min
# ...
deformed_vertices[in_region] = deformed_region
```

Vertices with `axis < axis_min` are bit-identical to input; vertices with
`axis >= axis_min` are fully FFD-displaced. At the boundary the FFD weight
is still positive (the nearest Bezier control slab applies), so the
displacement jumps from non-zero to zero between two adjacent triangles.

### 2.2 Asymmetry → no mirror enforcement + axis heuristic

`_kracht_to_lattice_offsets` builds offsets with `sign = 1 if centred > 0
else -1 if centred < 0 else 0`. That is *point-symmetric in the lattice
metric*, but:

* The underlying STL is not guaranteed perfectly symmetric around y=0; any
  baseline asymmetry is amplified by the displacement.
* The "beam axis" is picked as `secondary_axes[0]`, which is just
  `[a for a in range(3) if a != primary_axis][0]`. If a DCAD tool exports
  with the depth/height axis as Y instead of Z, the port-starboard roles
  swap silently.
* The lattice shape `(5, 4, 4)` has an even number of control points along
  beam — there is no midplane control point to pin `y=0`, so small
  numerical asymmetries can survive the FFD composition.

### 2.3 Facets → no post-smoothing + no subdivision

The adapter writes `trimesh.Trimesh(vertices=..., faces=mesh.faces)` after
the FFD. No Laplacian / Taubin / subdivision pass. Result: whatever
triangulation was in the baseline is preserved, with every triangle now
tilted differently. Coarse regions → visible facets.

---

## 3. Fixes (this PR)

### Fix A — Smooth radial blend across the boundary

Replace the hard `in_region` mask with a C¹ falloff envelope:

```
blend_width  = 0.10 * (axis_max - axis_min)   # 10% of bulb length
blend_start  = axis_min - blend_width

w(axis) = 0                                   if axis <= blend_start
         = smoothstep((axis - blend_start)    # Hermite s-curve
                      / blend_width)           if blend_start < axis < axis_min
         = 1                                    if axis >= axis_min

deformed[v] = mesh[v] + w(axis(v)) * (ffd(v) - mesh[v])
```

Every vertex with `axis > blend_start` participates in the FFD; the blend
zone smoothly interpolates between "unchanged" and "fully FFD-displaced".

### Fix B — Enforce port-starboard mirror symmetry

After building `offsets[l, m, n, 3]` in `_kracht_to_lattice_offsets`,
symmetrize around the beam midplane:

```python
beam = beam_axis
for j in range(m_cp):
    jm = m_cp - 1 - j
    # average lattice offsets for beam component with negated pair
    avg = 0.5 * (offsets[..., j, :, :] - offsets[..., jm, :, :])
    offsets[..., j, :, beam]  = +avg[..., beam]
    offsets[..., jm, :, beam] = -avg[..., beam]
    # average other components with same sign
    for other in (primary_axis, draft_axis):
        sym = 0.5 * (offsets[..., j, :, other] + offsets[..., jm, :, other])
        offsets[..., j, :, other]  = sym
        offsets[..., jm, :, other] = sym
```

This guarantees the lattice is mirror-symmetric about the beam midplane
regardless of any asymmetric bug in the mapping.

Additionally: after deformation, mirror-project the port-side vertices
onto the starboard side so any residual baseline asymmetry is removed
*in the deformed output*, not just the lattice. Opt-in via
`force_port_starboard_symmetry=True` (default: true in night-run).

### Fix C — Taubin volume-preserving post-smoothing

After FFD, run:

```python
import trimesh
trimesh.smoothing.filter_taubin(deformed, lamb=0.5, nu=-0.53, iterations=3)
```

Taubin alternates positive (smooth) and negative (anti-smooth) Laplacian
steps with lambda=0.5 and nu=-0.53 so the low-frequency shape is preserved
while high-frequency noise (facet edges) is flattened. Volume drift after
3 iterations is typically < 0.1%.

---

## 4. Learning from mistakes — follow-ups

Prioritised ideas for a future PR. Not landing in this design but listed so
future sessions have a clear queue.

### L1 — Historical surrogate with warm start
Persist every `(Kracht vector, high-fidelity Cd)` pair to
`~/.bulbopt/history.jsonl`. On the next night-run:
* Train a Gaussian Process on history.
* Warm-start NSGA-II initial population with top-K from history.
* Use GP prediction as the mid-gate evaluator instead of geometric proxy.

Sample efficiency boost: GP converges to within 5% of HF Cd after ~20
training points; after 100 it often beats physical proxies by 3-5× on
correlation.

### L2 — Mesh-quality as a third GA objective
Add `mesh_quality = max(dihedral_angle_deviation, symmetry_error)` to the
NSGA-II objective vector. Any candidate violating mesh invariants
(watertightness, self-intersection, curvature jumps > threshold) gets a
hard rejection via constraint. NSGA-II learns the valid subset of the
Kracht box.

### L3 — Bayesian optimisation on top of NSGA-II
Once the GP surrogate exists, switch the final few generations to
BoTorch qEHVI. qEHVI picks the point that most improves expected
hypervolume — much more sample-efficient than pure GA for expensive
objectives (real CFD).

### L4 — Shape validity prefilter
After the first 50 HF evaluations, learn a classifier (logistic
regression) predicting `mesh_invalid` from the 8 Kracht parameters. Use
the classifier to reject candidates before mesh generation — saves both
mesh and CFD cost.

### L5 — Adaptive subdivision
If a candidate's bulb region has fewer than N triangles in the FFD zone,
subdivide before deforming. This eliminates facets without over-meshing
flat baseline regions.

### L6 — Sanity-check CAD export
Before writing the STL, check:
* `mesh.is_watertight`
* `mesh.is_winding_consistent`
* `mesh.volume > 0`
Surface any failure as an artifact flag, so the HTML report can warn the
engineer that a particular candidate has a broken mesh.

---

## 5. Acceptance

This PR is accepted when:
1. A candidate's deformed mesh has no vertex displacement discontinuity
   across `axis_min` greater than 1% of the bulb length (test).
2. A candidate's deformed mesh has RMS vertex-to-mirror distance below
   1e-3 × mesh bounding box diagonal when
   `force_port_starboard_symmetry=True` (test).
3. After Taubin smoothing, the mesh's volume drift is within ±1% of the
   raw FFD output (test).
4. A real night-run on `docs/base_hull.stl` produces top-5 candidates whose
   STLs render without visible seams, and whose port-starboard silhouettes
   are visually identical.
5. All 144 existing tests still pass.

---

## 6. Execution plan (TDD)

| # | File | Test | Implementation |
|---|---|---|---|
| 1 | `test_ffd_deformer_symmetry.py` | vertices at (x, y, z) and (x, -y, z) match within 1e-6 for any KrachtVector | `_kracht_to_lattice_offsets` symmetrize pass + optional output-stage mirror |
| 2 | `test_ffd_deformer_blend.py` | max displacement jump across `axis_min` is ≤ 1% of blend_width | smooth falloff weight `w(axis)` applied to all vertices |
| 3 | `test_ffd_deformer_taubin.py` | volume drift ≤ 1% after 3 Taubin iterations | `trimesh.smoothing.filter_taubin` after FFD |

Each fix lands as one commit with the test + impl + updated existing
tests to account for the new smoother output. Real night-run at the end
to confirm the engineering outcome.
