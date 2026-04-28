# BulbOpt Desktop

`BulbOpt Desktop` is a modular desktop application for automated generation,
optimization, and engineering evaluation of a ship bulbous bow from a 3D hull
model. The first executable slice is STL-first, HTML-report first, and keeps
OpenFOAM optional so it runs end-to-end on mid-range laptops without a GPU.

## Vertical slice scope

| Area | Status |
|------|--------|
| Case creation + file-based storage | Done (§4.13, §9) |
| STL import + PyMeshFix repair | Done (§4.7) |
| Auto-detected bulb region + engineer override | Done (§11.2) |
| Candidate generation (`generate_new_bulb`, `local_optimize`) | Done (§11.3, §11.4) |
| Fast + mid-fidelity evaluation (hydrostatics-lite, calm-water, wave, multi-condition) | Done (§4.8) |
| Optimization ranking with acceptability thresholds | Done (§4.9) |
| OpenFOAM optional high-fidelity execution | Done (§4.8) |
| Checkpoint-based resume | Done (§10) |
| HTML report with repair / bulb / before-after / timing / CFD boundary | Done (§14) |
| Archived case package (`.zip`) | Done (§6.5) |
| Per-stage JSONL log (`logs/case.log`) | Done (§9) |
| Headless CLI for overnight / SSH / CI | Done (§1) |
| Desktop shell (PySide6) with history + Resume + Detect preview | Done (§4.1) |

## Install

Python 3.12+ and Git required.

```
git clone <repo>
cd bulbopt-desktop
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux / WSL / macOS
pip install -e ".[dev]"
```

Optional engineering tools:

- **OpenFOAM** — high-fidelity path. Detected automatically when `WM_PROJECT`
  is set or `blockMesh` is on `PATH`. Without it, the pipeline still completes
  and the report marks `high_fidelity_used=False` honestly.

## Desktop shell

```
python -m bulbopt.app.main
```

The desktop shell offers:

- **Case Wizard** with vessel geometry, speed profile, wave scenarios,
  objective weights, acceptability thresholds, and optional bulb region
  `axis_min` / `axis_max` overrides.
- **Detect Bulb Region** button to review auto-detected region without
  committing to a full run.
- **Run Vertical Slice** button to execute the whole pipeline.
- **Results panel** with 20+ summary labels (geometry, evaluation,
  hydrostatics, calm-water, wave response, multi-condition, CFD boundary,
  optimization trace, candidate comparison table with filter/sort).
- **Previous cases** panel listing persisted cases with status +
  recoverable flag.
- **Resume Selected Case** button enabled only for recoverable cases.
- **Open Case Folder / Report / Best Candidate STL / Case Package** buttons.

## Headless CLI

```
# Run a new case
python -m bulbopt.app.main run \
  --source docs/base_hull.stl \
  --project ./projects \
  --case-name dtmb-overnight \
  --candidate-count 5 \
  --optimization-mode generate_new_bulb

# List persisted cases
python -m bulbopt.app.main list --project ./projects

# Resume a recoverable case
python -m bulbopt.app.main resume --project ./projects --case case-abc12345

# Run a full night optimization
python -m bulbopt.app.main night-run \
  --source docs/base_hull.stl \
  --project ./projects \
  --case-name dtmb-night \
  --budget-hours 8 \
  --population 50 \
  --generations 20 \
  --high-fidelity-budget 10

# Inspect compatible CFD evidence before reusing warm-start / surrogate data
python -m bulbopt.app.main evidence \
  --project ./projects \
  --source docs/base_hull.stl \
  --backend external \
  --limit 10

# Machine-readable readiness gate for scripts / CI
python -m bulbopt.app.main evidence \
  --project ./projects \
  --source docs/base_hull.stl \
  --backend external \
  --json \
  --min-eligible 5
```

Exit codes: `0` on success, `1` on pipeline failure or failed readiness gate,
`2` on usage error.

The `evidence` command filters historical CFD rows by source-hull fingerprint
and solver-settings hash. Only compatible, engineering-valid, improving rows
are eligible for safe warm-start and surrogate training.

## Case folder layout

```
projects/
  case-<uuid>/
    case.json               # status, summary_metrics, is_recoverable
    metadata.json           # CreateCaseCommand snapshot
    artifacts_index.json    # pointers to every artifact
    candidate_index.json
    evaluation_index.json
    input/                  # source STL verbatim
    working/
      repaired/             # repaired.stl + geometry_analysis.json
      candidates/           # candidate-1.stl, candidate-2.stl, ...
      evaluation/           # optimization_summary.json
      openfoam_case/        # system/, constant/, run_manifest.json
      checkpoints/          # <case-id>-<stage>.json (status + elapsed_seconds)
    outputs/
      reports/report.html
      packages/<case-id>.zip
    logs/case.log           # JSONL per-stage transitions
```

## Pipeline stages (via LocalWorker)

Every stage is wrapped by `LocalWorker.run_strict`:

1. `prepare_geometry` — load STL, repair via PyMeshFix if non-watertight, detect bulb region (auto or user override).
2. `generate_candidates` — deform bow region into 3+ variants; amplitude set picked per `optimization_mode`.
3. `evaluate_candidates` — geometry metrics, hydrostatics-lite, calm-water surrogate, wave-response surrogate, multi-condition penalty, acceptability (ok/warn/reject).
4. `rank_candidates` — sort by `(acceptability_priority, mid_score)`.
5. `openfoam_build_case` — build `working/openfoam_case/` with blockMeshDict + snappyHexMeshDict + best candidate STL.
6. `openfoam_run_case` — execute `blockMesh` + `snappyHexMesh` when solver is available; skipped (recoverable) otherwise.
7. `build_html_report` — render `outputs/reports/report.html` from Jinja2 template.
8. `export_case_package` — zip the case folder into `outputs/packages/<case-id>.zip`.

Each stage writes a checkpoint with `status`, `result`, and `elapsed_seconds`.
A mid-pipeline failure marks the case `failed` + `is_recoverable=True`;
`resume_vertical_slice` short-circuits completed stages and re-runs only the
failing one.

## Tests

```
python -m pytest tests/ -v
```

On Windows, the desktop tests pick the offscreen Qt platform automatically;
the same command works headlessly in CI with `QT_QPA_PLATFORM=offscreen`.

## Module boundaries

```
ui.desktop           -> application.use_cases (no direct adapter access)
application.use_cases -> application.services.ports (Protocols)
infrastructure.adapters -> concrete tools (Trimesh, PyMeshFix, OpenFOAM)
domain.core             -> pure dataclasses, no dependencies on tools
storage.project_repository -> filesystem state, case-aware API
execution.worker + logging + checkpoints -> runtime plumbing
```

See `docs/superpowers/specs/2026-04-18-bulbopt-desktop-design.md` for the full
design rationale.
