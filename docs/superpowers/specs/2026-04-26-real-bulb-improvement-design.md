# Design — Real Bulb Improvement via DOE-Bootstrapped GP Surrogate

**Date**: 2026-04-26  
**Author**: Claude session (subagent-driven-development)

## Problem

Today's automated night-run with simpleFoam high-gate produced 3 candidates ALL worse than baseline (Cd 0.40-0.45 vs baseline 0.20). Hand-picked `tiny_sharp` (small bulb + sharp nose) achieved -2.48% real Cd reduction. **Conclusion**: the analytic mid-gate proxy `(beam*draft)/axial` anti-correlates with real Cd, sending NSGA-II in the wrong direction. Until enough real-CFD history accumulates (current threshold 20 rows), the proxy steers everything blindly.

## Goal

Within ~8 hours wall-clock, demonstrate at least one auto-discovered candidate that beats the baseline by ≥1% real Cd in OpenFOAM simpleFoam — and a path to compounding improvements over subsequent nights.

## Approach: Latin Hypercube DOE → GP Surrogate → NSGA-II + CFD

Three modules, dispatched in parallel.

### Module A — Latin Hypercube DOE seeder

`src/bulbopt/optimization/doe/latin_hypercube.py` (NEW):
- `latin_hypercube_sample(n: int, design_space, seed: int) -> list[KrachtVector]` — quasi-random space-filling sample using stratified random in 8 dimensions.
- Uses ONLY stdlib + numpy.

`scripts/cfd_doe_seed.py` (NEW): top-level script that
1. Runs Latin hypercube of N=30 over the (now-tightened) Kracht bounds.
2. For each sample: deform `docs/base_hull.stl`, run blockMesh+snappyHexMesh+checkMesh+simpleFoam, parse Cd.
3. Append every (vector, Cd, backend="simple_foam") row to the project's `~/.bulbopt/history.jsonl` and CFD evidence store.
4. Writes a small `seed_summary.json` with min/max/mean Cd, time-elapsed.

### Module B — Lower GP-surrogate threshold + smarter mid-gate fallback

`src/bulbopt/application/use_cases/run_night_optimization.py`:
- Lower `gp_surrogate_min_history` default from 20 to **12** (matches DOE bootstrap of ≥12 points).
- When GP available: ALSO replace objective[1] (volume_delta — already correct) and objective[2] (mesh_quality — already correct), keep objective[0] = GP-mean.
- When GP unavailable AND fewer than 12 simple_foam rows: log a clear warning that NSGA-II is running blind on the proxy.

`src/bulbopt/optimization/learning/gp_surrogate.py`:
- Add `predict_with_uncertainty(vectors) -> tuple[mean, std]` already exists. Confirm it's used.
- Add a unit test for the warm-start integration.

### Module C — Tightened Kracht bounds + known-good seed

`src/bulbopt/optimization/parametric/kracht_space.py`:
- Add `KrachtDesignSpace.tightened()` factory returning a `KrachtDesignSpace` with bounds biased toward `tiny_sharp` neighborhood:
  - `length_ratio: (0.005, 0.030)` — was (0.005, 0.045), drop the wide-bulb upper end
  - `breadth_ratio: (0.0, 0.20)` — was (0.0, 0.20), unchanged
  - `nose_sharpness: (0.5, 1.0)` — was (0.0, 1.0), drop the round-dome end
  - `volume_coef: (-0.40, 0.40)` — was (-0.40, 0.90), drop the inflate-massive end
  - Other 4 dimensions: unchanged
- DOE seeder uses `tightened()` by default; full Kracht space remains as alternative for exploration.
- Adds the hand-picked `tiny_sharp` Kracht vector as a constant `TINY_SHARP_KRACHT_VECTOR` (importable for tests + warm-start hint).

## Acceptance criteria

1. After Module A's DOE-30 run completes, `history.jsonl` has ≥30 rows tagged `backend="simple_foam"`.
2. With ≥12 rows of real CFD, GP-mid-gate path is exercised in `run_night_optimization` (logged as `surrogate.status=trained`).
3. A subsequent NSGA-II run with population=12, generations=4, high_fidelity_budget=4 produces at least 1 candidate with measured CFD Cd < baseline Cd × 0.99 (i.e. ≥1% improvement). Demonstrated on `docs/base_hull.stl`.
4. All 300 prior tests still pass; new tests for DOE + GP threshold cross verified.

## Out of scope (deferred)

- Multi-Froude evaluation in CFD (currently single Fr~0.27 via reference values in adapter).
- interFoam/VoF + 6DoF (Gate 4) — needs design-spec extension.
- Kracht space `breadth_ratio`/`height_ratio` going negative.

## Test plan

Per module:
- Module A: 3 unit tests (deterministic seed, bounds respect, dedup at distance 0.02).
- Module B: 2 tests (GP activates at threshold-1=11→11→12 rows, log message present when blind).
- Module C: 2 tests (tightened bounds shape, factory returns same Kracht names).

Total expected delta: +7 tests → 307 passing.

## Empirical validation plan (after agents land)

1. Push commits.
2. Run `scripts/cfd_doe_seed.py --n 30 --bounds tightened --case-name doe_seed --root C:/tmp/doe_seed`. ETA ≈ 2.5h.
3. After DOE completes, run `bulbopt night-run` on `docs/base_hull.stl` with population=12, generations=4, high_fidelity_budget=4 and `history_path=~/.bulbopt/history.jsonl`. ETA ≈ 30 min.
4. Run `scripts/cfd_baseline_vs_winners.py` on the case's top candidates.
5. Report: baseline Cd, candidate Cd, % improvement, wall-clock totals.
