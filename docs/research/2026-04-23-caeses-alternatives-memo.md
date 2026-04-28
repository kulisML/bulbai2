# CAESES + parametric CAD alternatives — research memo

**Date:** 2026-04-23
**Author:** Agent (BulbOpt)
**Scope:** identify replacement / augmentation candidates for the current
home-grown FFD deformer (`bulbopt.optimization.parametric.ffd_deformer`).
Evaluation only — no installs, no integration code.

---

## Summary

| Tool | License | Free tier | Python API | Headless | Ship-hull fit (1-5) | Integration (person-days) |
|---|---|---|---|---|---|---|
| CAESES | Commercial (FRIENDSHIP SYSTEMS) | "Free Edition" exists, limited | yes (`.fsc`/Python) | yes (batch) | 5 | ~8-12 (external process) |
| OpenVSP | NASA open source (Apache-style) | yes (100% free) | yes (`openvsp` pkg) | yes | 4 | ~5-7 |
| FreeCAD | LGPL-2 | yes | yes (`import FreeCAD`) | yes (`FreeCADCmd`) | 3 | ~6-8 |
| Blender Python (`bpy`) | GPL-2 | yes | yes | yes (`blender --background`) | 2 | ~3-5 |
| Salome-Meca | LGPL | yes | yes (`salome.geom`) | yes | 3 | ~10-14 |

**Recommendation:** prototype OpenVSP first (NASA, Apache-style, canonical for
hull parametrics, best-documented Python API), CAESES second iff the Free
Edition covers bulbous-bow feature parameters (unverified — see §1 caveat).

---

## 1. CAESES (FRIENDSHIP SYSTEMS)

### License / free tier

- Commercial product; headline pricing is quote-based. **"CAESES Free"** edition
  exists (sometimes branded "CAESES Framework Free") and historically included:
  - full GUI, parametric modelling, and the FSC scripting language,
  - STL / IGES export,
  - limited automation (no batch parallelism, no commercial OpenFOAM coupling),
  - watermark on exported geometry in older versions (unverified current state).
- The bulbous-bow parametric template (the "Kracht" style templates shipped by
  FRIENDSHIP SYSTEMS) is **not** guaranteed to be in the free edition; the
  "Naval Architect" bundle that includes it is typically paid-tier.
- Caveat: FRIENDSHIP's website wording around free-tier scope has shifted
  multiple times since 2019. Before committing, ping sales for written confirmation
  that (a) FSC scripting is free-tier, (b) STL export is unwatermarked,
  (c) headless batch mode is available.

### Python story

- CAESES ships two scripting surfaces:
  - **FSC** (Feature Scripting, the native language). Python-like but not Python.
  - **Python API** via the `FriendshipFramework` module, accessible from within
    the CAESES runtime. Not pip-installable — it ships with the CAESES binary.
- Headless mode: `caeses -b script.fsc` on Windows/Linux runs without the GUI.
  Integration would be an **external-process adapter**, not an in-process Python
  import.

### Ship-hull fit

- 5/5. CAESES is the canonical tool for naval parametric modelling; it ships
  with validated bulbous-bow, stern, and transom templates, and hull fairness
  constraints (curvature continuity, developable surfaces).
- The generated geometry tends to be higher-quality than direct FFD on a
  triangulated STL.

### Integration sketch (if free-tier sufficient)

- New adapter `CAESESDeformer` as a peer to `BulbFFDDeformer`:
  ```
  class CAESESDeformer:
      def deform(self, mesh, region, vector) -> trimesh.Trimesh:
          # 1. write .fsc input file with vector.values mapped to CAESES
          #    parameter names (length_ratio → L_BB, ...)
          # 2. subprocess caeses -b run.fsc
          # 3. load resulting STL with trimesh and return
  ```
- Would need a round-trip time benchmark — CAESES has a non-trivial startup cost
  (several seconds per invocation).
- Estimate: **8-12 person-days** (parameter mapping, STL round-trip, error handling,
  license detection).

### Blockers

- Uncertain free-tier bulbous-bow template availability.
- License ceremony (activation file, online check) is awkward for night-run automation.

---

## 2. OpenVSP (NASA)

### License / free tier

- Apache-style (NASA Open Source Agreement 1.3) — 100% free, commercial use allowed.
- Source + binaries for Windows, Linux, macOS at https://openvsp.org.

### Python story

- First-class Python API via the `openvsp` pip package (ships with binary wheels
  matching the OpenVSP version). Usage:
  ```python
  import openvsp as vsp
  vsp.VSPCheckSetup()
  geom_id = vsp.AddGeom("FUSELAGE")  # or "WING", "HULL"
  vsp.SetParmValUpdate(geom_id, "Length", "Design", 142.0)
  vsp.WriteSTLFile("hull.stl", vsp.SET_ALL)
  ```
- Supports headless operation; no GUI required.

### Ship-hull fit

- 4/5. OpenVSP was built for aircraft but the `FUSELAGE` geom is extensively
  used for hull-form parametrisation (NASA/DARPA submarine studies;
  DTMB-5415 parametric decomposition). There is no bulbous-bow component — you'd
  attach it as a separate `POD` (streamlined body) geom to the fuselage nose.
- The 8 Kracht parameters map cleanly onto OpenVSP POD + FUSELAGE parameters.

### Integration sketch

```
class OpenVSPDeformer:
    def deform(self, mesh, region, vector) -> trimesh.Trimesh:
        import openvsp as vsp
        # 1. create POD geom with vector-derived size
        # 2. write STL to temp, load with trimesh, return
```

- Estimate: **5-7 person-days**. Main cost is the parameter mapping and
  validation (NSGA-II needs monotonic, well-scaled inputs).

### Blockers

- Running in-process in a PySide6 app requires care — `openvsp` has a C++
  OpenGL-enabled singleton, calling it from multiple threads would be brittle.
  Easy workaround: one-process-per-call via `subprocess` + a CLI wrapper.

---

## 3. FreeCAD

### License / free tier

- LGPL-2 — free, open source.

### Python story

- Mature: every FreeCAD workbench is scripted in Python. Supports headless
  via `FreeCADCmd` (no GUI binary) and direct `import FreeCAD` from a normal
  CPython (requires the FreeCAD libs on `LD_LIBRARY_PATH` / `PATH`).
- The **Ship Workbench** (community-maintained) parametrises hulls (wetted
  surface, displacement curves, stability). Less mature than OpenVSP for
  bulbous bows — typical workflow imports a parent hull as B-rep and uses
  form-feature fairing rather than parametric generation.

### Ship-hull fit

- 3/5. Good for generic CAD B-rep work; less mature than OpenVSP for
  bulbous-bow generation. Strengths: exact B-rep (vs. triangle mesh) → easier
  hydrostatics.

### Integration sketch

- `FreeCADDeformer` adapter; primary work is building the Python driver script
  that loads the baseline `.FCStd`, adjusts parameters, exports STL.
- Estimate: **6-8 person-days** (FreeCAD's Python API is full-featured but
  discoverability is low; a lot of trial-and-error mapping).

---

## 4. Blender Python (`bpy`)

### License / free tier

- GPL-2 — free, open source. GPL triggers **copyleft** on anything linked
  against `bpy`. This is a real concern: if BulbOpt links `bpy` in-process,
  the whole BulbOpt codebase would arguably need to be GPL-2 as well.
  Mitigation: run Blender as an external process via `blender --background --python script.py`.

### Python story

- `bpy` is bundled with Blender (no pip install for production use, though
  `bpy` wheels exist for specific CPython versions).
- Geometry-nodes + drivers enable procedural parametric workflows. Blender's
  native sculpt tools support shrinkwrap / proportional editing — conceptually
  similar to FFD but designed for artistic workflows.

### Ship-hull fit

- 2/5. Possible but Blender is fundamentally a DCC (digital content creation)
  tool. Watertightness, exact dimensions, and symmetry enforcement require
  add-ons. Not the right abstraction for a naval-optimisation pipeline.

### Integration sketch

- External-process adapter to avoid GPL contamination.
- Estimate: **3-5 person-days** for a naive "load + nudge + export" pipeline
  but likely 2-3× more for robust parametric mapping.

### Blockers

- License (see above).
- No native Kracht / bulbous-bow model.

---

## 5. Salome-Meca

### License / free tier

- LGPL — free, open source. Salome itself is a platform (CAD + mesh +
  simulation). "Salome-Meca" is EDF's Code_Aster distribution that bundles
  Salome with FEA solvers.

### Python story

- `salome.geom` + `salome.smesh` modules are the CAD + meshing scripting
  surfaces. Scripts run inside the Salome runtime (`salome -t script.py`).
- Ship-hull modelling is documented in EDF training materials (typically
  for propeller / rudder HPC studies).

### Ship-hull fit

- 3/5. Strong exact-B-rep CAD, but no built-in bulbous-bow template. Would
  require hand-crafting a parametric hull from primitives.

### Integration sketch

- External-process adapter, similar to CAESES.
- Estimate: **10-14 person-days** — Salome's Python API has a steep learning
  curve and the documentation trails the implementation.

---

## Recommendation

1. **OpenVSP first.** Open-source, pip-installable, strong Python story, NASA
   provenance. Build an `OpenVSPDeformer` adapter behind the existing
   `BulbFFDDeformer` interface and A/B test on a real night-run. Estimate 1 week.
2. **CAESES second, contingent on free-tier confirmation.** If FRIENDSHIP
   SYSTEMS confirms the free edition covers FSC + STL export + bulbous-bow
   templates, build a subprocess-based adapter. Fall-back if OpenVSP's
   bulbous-bow fit isn't good enough.
3. **Defer FreeCAD, Blender, Salome.** FreeCAD is a distant third (no strong
   ship-hull advantage over OpenVSP); Blender's licence creates unnecessary
   friction; Salome's API curve doesn't justify the integration cost.

Explicitly skip any in-process Python import of `bpy` — the GPL contamination
risk is not worth the convenience.
