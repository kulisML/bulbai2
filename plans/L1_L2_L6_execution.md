# L1 + L2 + L6 execution plan

Follow-ups from `docs/superpowers/specs/2026-04-23-bulbopt-mesh-quality-design.md` §4.
One commit per L-item, TDD discipline. All 147 existing tests must stay green
between commits.

## Execution order

### Commit 0 — this plan
`docs: plan L1+L2+L6 mesh-quality follow-ups`

### Commit 1 — L1 Historical GP surrogate with warm start
- `src/bulbopt/optimization/learning/__init__.py`
- `src/bulbopt/optimization/learning/history_store.py` — JSONL append-only with
  `record(kracht, cd)`, `load_all()`, `top_k(n)`; default
  `~/.bulbopt/history.jsonl` with override.
- `src/bulbopt/optimization/learning/gp_surrogate.py` — wraps
  `sklearn.gaussian_process.GaussianProcessRegressor`; RBF + white-noise
  kernel; `fit`/`predict`; returns `None` if < 5 training points.
- `src/bulbopt/optimization/strategies/nsga2_strategy.py` — optional
  `warm_start_vectors` kwarg that seeds initial pymoo population.
- `src/bulbopt/application/use_cases/run_night_optimization.py` — write history
  after HF eval; load top-10 on start to warm-start; if ≥20 points, wrap mid
  evaluator with GP fallback.
- Tests: `test_history_store.py`, `test_gp_surrogate.py`,
  `test_nsga2_warm_start.py`; extend `test_run_night_optimization.py`.
- Dependency: `scikit-learn>=1.3` in pyproject.toml.

### Commit 2 — L2 Mesh quality as 3rd GA objective
- `src/bulbopt/optimization/quality/mesh_metrics.py` — `compute_mesh_quality`
  returns `max(dihedral_angle_deviation, symmetry_error,
  watertight_penalty)`; watertight_penalty=100 if broken, else 0.
- `cascade_strategy.py` — pass `n_objectives=3` to NSGA2Strategy.
- `run_night_optimization.py` — mid evaluator returns 3 objectives.
- Tests: `test_mesh_metrics.py`; extend `test_cascade_strategy.py` to assert
  3 objectives flow end-to-end.

### Commit 3 — L6 Sanity-check CAD export
- `src/bulbopt/infrastructure/adapters/stl_sanity.py` — `validate_stl`
  returns dict `{watertight, winding_consistent, volume, vertex_count,
  face_count, checks_passed}`.
- `run_night_optimization.py` — write `stl_valid.json` beside each STL;
  track candidates with failures and pass to report.
- `night_report.html.j2` — new warning section under
  `{% if stl_invalid_candidates %}`.
- Tests: `test_stl_sanity.py`.

## Invariants
- `QT_QPA_PLATFORM=offscreen python -m pytest tests/ --tb=short` green
  after every commit.
- No touching `ffd_deformer.py`, `simple_foam_gate.py` beyond L6 hook,
  `openfoam_adapter.py`, or `openfoam_runner.py`.
- Commits authored via
  `git -c user.email=kulismlengineer107@gmail.com -c user.name=Kirill commit`.
