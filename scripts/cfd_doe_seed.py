"""Seed the BulbOpt history store with a Latin hypercube DOE of real CFD runs.

Audit 2026-04-26 fix: the analytic mid-gate proxy ``(beam*draft)/axial``
*anti-correlates* with simpleFoam Cd, so the GP surrogate can only
become useful once we have a small bootstrap of real (Kracht, Cd) pairs.
This script deforms the repaired baseline mesh by a Latin hypercube
sample of Kracht vectors, runs each through the OpenFOAM solver chain,
parses Cd from ``forceCoeffs/0/coefficient.dat``, and appends every
result to the project's ``HistoryStore`` with ``backend="simple_foam"``.

Once the history accumulates ``>=12`` ``simple_foam`` rows, the existing
GP-mid-gate path activates automatically (see
``run_night_optimization``).

Usage::

    set BULBOPT_OPENFOAM_BIN=C:/Users/.../OpenFOAM-v2512/.../bin
    python scripts/cfd_doe_seed.py \
        --case-name doe_2026_04_26 \
        --root C:/Users/.../bulbopt_projects \
        --n 30 \
        --bounds tightened

The script is intentionally **defensive on failure** — a single failed
case does not abort the sweep. ``cd`` is recorded as ``None`` in the
per-case manifest and that vector is *not* appended to the history
store (we never want to train the GP on ``None``).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import trimesh

from bulbopt.infrastructure.adapters.force_coeffs_parser import (
    ForceCoeffsNotFoundError,
    parse_drag_coefficient_dat,
)
from bulbopt.infrastructure.adapters.openfoam_adapter import OpenFOAMAdapter
from bulbopt.infrastructure.adapters.openfoam_runner import OpenFOAMRunnerAdapter
from bulbopt.infrastructure.adapters.stub_geometry import StubGeometryAdapter
from bulbopt.optimization.doe.latin_hypercube import latin_hypercube_sample
from bulbopt.optimization.learning.history_store import (
    HistoryStore,
    default_history_path,
)
from bulbopt.optimization.parametric.ffd_deformer import BulbFFDDeformer
from bulbopt.optimization.parametric.kracht_space import (
    KRACHT_PARAMETER_NAMES,
    KrachtDesignSpace,
    KrachtVector,
)


def _resolve_design_space(bounds_choice: str) -> KrachtDesignSpace:
    """Build the KrachtDesignSpace requested by ``--bounds``.

    Module C may or may not have shipped ``KrachtDesignSpace.tightened()``
    yet; if it's missing we degrade gracefully to the default space and
    print a clear note.
    """
    if bounds_choice == "full":
        return KrachtDesignSpace()
    if bounds_choice == "tightened":
        if hasattr(KrachtDesignSpace, "tightened"):
            return KrachtDesignSpace.tightened()
        print(
            "  NOTE: KrachtDesignSpace.tightened() not available yet — "
            "falling back to default bounds.",
            file=sys.stderr,
        )
        return KrachtDesignSpace()
    raise ValueError(f"Unknown --bounds: {bounds_choice!r}")


def _build_baseline(
    *,
    case_dir: Path,
    source_stl: Path,
) -> tuple[trimesh.Trimesh, dict]:
    """Run ``prepare_geometry`` on the case dir and return (mesh, region)."""
    case_dir.mkdir(parents=True, exist_ok=True)
    # StubGeometryAdapter.prepare_geometry writes to <case_dir>/input/,
    # <case_dir>/working/repaired/, and reads/writes <case_dir>/artifacts_index.json.
    # None of those are created by the adapter itself. Set up the skeleton first.
    (case_dir / "input").mkdir(parents=True, exist_ok=True)
    (case_dir / "working" / "repaired").mkdir(parents=True, exist_ok=True)
    artifacts_path = case_dir / "artifacts_index.json"
    if not artifacts_path.exists():
        artifacts_path.write_text("{}", encoding="utf-8")
    geometry = StubGeometryAdapter()
    analysis = geometry.prepare_geometry(case_dir, source_stl)
    repaired_path = case_dir / "working" / "repaired" / "repaired.stl"
    mesh = trimesh.load(repaired_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise RuntimeError(f"Failed to load repaired mesh from {repaired_path}")
    return mesh, analysis["bulb_region"]


def _run_simple_foam_for_vector(
    *,
    label: str,
    vector: KrachtVector,
    baseline_mesh: trimesh.Trimesh,
    region: dict,
    deformer: BulbFFDDeformer,
    builder: OpenFOAMAdapter,
    runner: OpenFOAMRunnerAdapter,
    work_root: Path,
    timeout_seconds: int,
) -> dict:
    """Build + execute one OpenFOAM case for a Kracht vector. Returns a manifest dict.

    Defensive: any exception inside the deform / build / run stages is
    captured and surfaced as ``status="failed"`` with ``cd=None``.
    """
    case_dir = work_root / label
    if case_dir.exists():
        shutil.rmtree(case_dir, ignore_errors=True)
    case_dir.mkdir(parents=True, exist_ok=True)

    deformed_path = case_dir / "deformed.stl"
    of_case_dir = case_dir / "working" / "openfoam_case"

    manifest: dict = {
        "label": label,
        "vector": dict(vector.values),
        "cd": None,
        "status": "started",
        "reason": None,
        "openfoam_case_dir": str(of_case_dir),
    }

    try:
        deformed = deformer.deform(baseline_mesh, region, vector)
        deformed_path.write_bytes(trimesh.exchange.stl.export_stl(deformed))

        case_manifest = builder.build_case(
            case_dir=case_dir,
            best_candidate_id=label,
            best_candidate_geometry_path=deformed_path,
        )
        run_manifest = runner.run_case(
            of_case_dir,
            case_manifest=case_manifest,
            execute=True,
            timeout_seconds=timeout_seconds,
        )

        runner_status = run_manifest.get("status") or run_manifest.get("runner_status")
        manifest["runner_status"] = runner_status
        manifest["runner_reason"] = (
            run_manifest.get("reason") or run_manifest.get("runner_reason")
        )

        if runner_status != "executed_ok":
            manifest["status"] = "failed"
            manifest["reason"] = manifest["runner_reason"] or "solver_chain_failed"
            return manifest

        cd_value = _read_force_coeffs_cd(of_case_dir)
        if cd_value is None:
            manifest["status"] = "failed"
            manifest["reason"] = "force_coeffs_unavailable"
            return manifest

        manifest["cd"] = float(cd_value)
        manifest["status"] = "succeeded"
        return manifest

    except Exception as exc:  # pragma: no cover - defensive logging path
        manifest["status"] = "failed"
        manifest["reason"] = f"{type(exc).__name__}: {exc}"
        manifest["traceback"] = traceback.format_exc()
        return manifest


def _read_force_coeffs_cd(of_case_dir: Path) -> Optional[float]:
    """Locate ``forceCoeffs/0/coefficient.dat`` and return its final Cd, if any.

    Mirrors :func:`bulbopt.infrastructure.adapters.simple_foam_gate._read_force_coeffs`
    but trimmed to just the ``final_cd`` lookup so the script depends on
    the public parser only.
    """
    post_root = of_case_dir / "postProcessing"
    if not post_root.exists():
        return None
    candidate_roots = [post_root / "forceCoeffs", post_root / "forces"]
    forces_root = next((c for c in candidate_roots if c.exists()), None)
    if forces_root is None:
        return None
    subdirs = [p for p in forces_root.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    latest = max(subdirs, key=lambda p: p.stat().st_mtime)
    dat_path = next(
        (
            latest / filename
            for filename in ("forceCoeffs.dat", "coefficient.dat")
            if (latest / filename).exists()
        ),
        latest / "coefficient.dat",
    )
    try:
        report = parse_drag_coefficient_dat(dat_path)
    except (ForceCoeffsNotFoundError, ValueError):
        return None
    return float(report["final_cd"])


def _vectors_to_resume_keys(
    vectors: Sequence[KrachtVector],
    *,
    digits: int = 12,
) -> List[Tuple[float, ...]]:
    """Map LHS vectors to round-tripped float tuples for resume comparison.

    The LHS sampler is deterministic for a fixed ``(n, seed)`` pair, but
    JSON round-trip through ``HistoryStore`` can introduce sub-ULP
    rounding noise on the float values. We compare with the same number
    of significant digits the JSONL store keeps, which is plenty more
    than the 6-7 digits LHS strata actually need.
    """
    return [
        tuple(round(float(v.values[name]), digits) for name in KRACHT_PARAMETER_NAMES)
        for v in vectors
    ]


def _completed_indices_from_history(
    history_path: Path,
    samples: Sequence[KrachtVector],
    *,
    backend: str = "simple_foam",
    digits: int = 9,
) -> set[int]:
    """Return the 0-based sample indices already recorded in ``history_path``.

    A sample is considered completed when ``history_path`` contains a
    row with the same ``backend`` and parameters that match within
    ``10**-digits``. Used by ``--resume`` to skip already-evaluated
    samples without re-running CFD.

    Missing history file or empty history → empty set (nothing to skip).
    """
    if not history_path.exists():
        return set()
    store = HistoryStore(path=history_path)
    rows = store.load_all(backend=backend)
    if not rows:
        return set()
    sample_keys = _vectors_to_resume_keys(samples, digits=digits)
    completed: set[int] = set()
    for vec, _cd in rows:
        row_key = tuple(
            round(float(vec.values[name]), digits) for name in KRACHT_PARAMETER_NAMES
        )
        for index, sample_key in enumerate(sample_keys):
            if index in completed:
                continue
            if all(
                math.isclose(a, b, rel_tol=0.0, abs_tol=10 ** (-digits))
                for a, b in zip(row_key, sample_key)
            ):
                completed.add(index)
                break
    return completed


# ---------- parallel worker entry point -----------------------------------
#
# ``ProcessPoolExecutor`` requires a top-level (importable) callable so it
# can pickle a reference to it on the parent side and resolve it by name
# inside the child interpreter. Lambdas / closures don't survive that
# round-trip on Windows, so the parallel chunk-runner lives at module
# scope and rebuilds its own adapters instead of receiving them from the
# parent.


def _worker_run_chunk(payload: dict) -> List[dict]:
    """Run one chunk of LHS samples in a fresh subprocess.

    ``payload`` is a JSON-friendly dict with the per-chunk arguments
    (case dir, baseline STL bytes, vectors as JSON, timeout, ...). We
    intentionally re-instantiate the OpenFOAM adapters inside the
    worker so each subprocess has a clean state — sharing them across
    fork would not be safe given OpenFOAM's reliance on per-process
    environment configuration.

    Returns a list of per-sample manifest dicts (same shape as
    ``_run_simple_foam_for_vector``).
    """
    case_dir = Path(payload["case_dir"])
    work_root = Path(payload["work_root"])
    doe_results_dir = Path(payload["doe_results_dir"])
    repaired_path = Path(payload["repaired_stl"])
    timeout_seconds = int(payload["timeout"])
    items = payload["items"]
    region = payload["region"]

    # Lazy imports inside the worker so the parent's ``sys.path`` setup
    # propagates and the child does not inherit any half-initialised
    # state from the importer.
    import trimesh as _trimesh
    from bulbopt.infrastructure.adapters.openfoam_adapter import (
        OpenFOAMAdapter as _OpenFOAMAdapter,
    )
    from bulbopt.infrastructure.adapters.openfoam_runner import (
        OpenFOAMRunnerAdapter as _OpenFOAMRunnerAdapter,
    )
    from bulbopt.optimization.parametric.ffd_deformer import (
        BulbFFDDeformer as _BulbFFDDeformer,
    )
    from bulbopt.optimization.parametric.kracht_space import (
        KrachtVector as _KrachtVector,
    )

    baseline_mesh = _trimesh.load(repaired_path, force="mesh")
    builder = _OpenFOAMAdapter()
    runner = _OpenFOAMRunnerAdapter()
    deformer = _BulbFFDDeformer()

    out: List[dict] = []
    for item in items:
        label = item["label"]
        vector = _KrachtVector(values={k: float(v) for k, v in item["vector"].items()})
        manifest = _run_simple_foam_for_vector(
            label=label,
            vector=vector,
            baseline_mesh=baseline_mesh,
            region=region,
            deformer=deformer,
            builder=builder,
            runner=runner,
            work_root=work_root,
            timeout_seconds=timeout_seconds,
        )
        manifest_path = doe_results_dir / f"{label}.json"
        try:
            manifest_path.write_text(
                json.dumps(manifest, indent=2, default=str), encoding="utf-8"
            )
        except OSError:
            # Persisting is best-effort inside a worker; the parent
            # also re-writes the manifest after aggregation.
            pass
        out.append(manifest)
    return out


def _run_baseline_cd(
    *,
    baseline_mesh: trimesh.Trimesh,
    builder: OpenFOAMAdapter,
    runner: OpenFOAMRunnerAdapter,
    work_root: Path,
    source_stl: Path,
    timeout_seconds: int,
) -> Optional[float]:
    """Run the undeformed (repaired) baseline through OpenFOAM. Returns Cd or None."""
    case_dir = work_root / "baseline"
    if case_dir.exists():
        shutil.rmtree(case_dir, ignore_errors=True)
    case_dir.mkdir(parents=True, exist_ok=True)
    deformed_path = case_dir / "baseline.stl"
    deformed_path.write_bytes(trimesh.exchange.stl.export_stl(baseline_mesh))
    of_case_dir = case_dir / "working" / "openfoam_case"
    try:
        case_manifest = builder.build_case(
            case_dir=case_dir,
            best_candidate_id="baseline",
            best_candidate_geometry_path=deformed_path,
        )
        run_manifest = runner.run_case(
            of_case_dir,
            case_manifest=case_manifest,
            execute=True,
            timeout_seconds=timeout_seconds,
        )
        if (
            run_manifest.get("status") != "executed_ok"
            and run_manifest.get("runner_status") != "executed_ok"
        ):
            return None
        return _read_force_coeffs_cd(of_case_dir)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  WARN: baseline CFD failed: {exc}", file=sys.stderr)
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Latin hypercube DOE of real OpenFOAM Cd evaluations and "
            "seed the GP surrogate via the BulbOpt history store."
        )
    )
    parser.add_argument("--n", type=int, default=30, help="Number of LHS samples (default 30)")
    parser.add_argument(
        "--bounds",
        choices=("tightened", "full"),
        default="tightened",
        help="Use tightened (Module C) or full default Kracht bounds (default tightened)",
    )
    parser.add_argument(
        "--case-name",
        required=True,
        help="Logical case-name; the script writes under <root>/<case-name>/",
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Project root containing the seed run (a sibling of normal cases)",
    )
    parser.add_argument(
        "--source-stl",
        default=str(REPO_ROOT / "docs" / "base_hull.stl"),
        help="Source STL to load and repair (default docs/base_hull.stl)",
    )
    parser.add_argument(
        "--history-path",
        default=str(default_history_path()),
        help="Path to the history.jsonl store (default ~/.bulbopt/history.jsonl)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-case OpenFOAM solver-chain timeout in seconds (default 600)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for LHS sampling (default 0)",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip the baseline CFD run (useful for resuming a partial sweep)",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help=(
            "Run DOE samples concurrently across N subprocesses (default 1, "
            "i.e. fully sequential — bit-identical to historical behaviour). "
            "N>1 splits the LHS samples into N chunks and dispatches each to "
            "its own ProcessPoolExecutor worker."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip LHS samples whose KrachtVector already has a "
            "simple_foam Cd row in --history-path. Use this to restart a "
            "partial sweep without re-running the completed cases."
        ),
    )
    args = parser.parse_args(argv)

    if args.parallel_workers < 1:
        print(
            f"--parallel-workers must be >= 1; got {args.parallel_workers}",
            file=sys.stderr,
        )
        return 2

    if not os.environ.get("BULBOPT_OPENFOAM_BIN"):
        print(
            "BULBOPT_OPENFOAM_BIN is not set. Point it at the bin/ directory of "
            "your OpenFOAM-v2512 install before running this script. Aborting.",
            file=sys.stderr,
        )
        return 2

    source_stl = Path(args.source_stl).resolve()
    if not source_stl.is_file():
        print(f"Source STL not found: {source_stl}", file=sys.stderr)
        return 2

    project_root = Path(args.root).resolve()
    case_dir = project_root / args.case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    doe_results_dir = case_dir / "doe_results"
    doe_results_dir.mkdir(parents=True, exist_ok=True)
    work_root = case_dir / "working" / "doe"
    work_root.mkdir(parents=True, exist_ok=True)

    history_path = Path(args.history_path).resolve()
    history_store = HistoryStore(path=history_path)

    print(f"Loading + repairing baseline from {source_stl}")
    baseline_mesh, region = _build_baseline(case_dir=case_dir, source_stl=source_stl)

    design_space = _resolve_design_space(args.bounds)
    samples = latin_hypercube_sample(args.n, design_space, seed=args.seed)
    print(
        f"Generated {len(samples)} Latin hypercube samples "
        f"(bounds={args.bounds!r}, seed={args.seed})"
    )

    builder = OpenFOAMAdapter()
    runner = OpenFOAMRunnerAdapter()
    deformer = BulbFFDDeformer()

    start_time = time.monotonic()

    baseline_cd: Optional[float] = None
    if not args.skip_baseline:
        print("== Running baseline (undeformed) through OpenFOAM ==")
        baseline_cd = _run_baseline_cd(
            baseline_mesh=baseline_mesh,
            builder=builder,
            runner=runner,
            work_root=work_root,
            source_stl=source_stl,
            timeout_seconds=args.timeout,
        )
        print(f"  baseline Cd = {baseline_cd}")

    cd_values: list[float] = []
    n_succeeded = 0
    n_failed = 0
    n_skipped = 0

    # ---- resume gate ------------------------------------------------------
    completed_indices: set[int] = set()
    if args.resume:
        completed_indices = _completed_indices_from_history(history_path, samples)
        if completed_indices:
            print(
                f"  --resume: {len(completed_indices)} of {len(samples)} samples "
                f"already in history; will skip"
            )

    pending: List[Tuple[int, KrachtVector]] = [
        (i, vec) for i, vec in enumerate(samples) if i not in completed_indices
    ]
    n_skipped = len(samples) - len(pending)

    def _persist_result(
        label: str,
        vector: KrachtVector,
        result: dict,
        *,
        prefix_label: bool = False,
    ) -> None:
        """Common path: write per-case manifest + maybe record in history.

        ``prefix_label`` keeps the parallel-path log lines self-describing
        (chunks complete out of order); the sequential path keeps the
        original ``  Cd = X`` format so its log output is bit-identical
        to the pre-refactor script.
        """
        nonlocal n_succeeded, n_failed
        manifest_path = doe_results_dir / f"{label}.json"
        manifest_path.write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
        cd = result.get("cd")
        tag = f"{label}: " if prefix_label else ""
        if result.get("status") == "succeeded" and cd is not None:
            history_store.record(vector, cd=float(cd), backend="simple_foam")
            cd_values.append(float(cd))
            n_succeeded += 1
            print(f"  {tag}Cd = {cd:.4f}  (recorded in {history_path})")
        else:
            n_failed += 1
            print(
                f"  {tag}FAILED: status={result.get('status')!r} "
                f"reason={result.get('reason')!r}"
            )

    if args.parallel_workers <= 1 or len(pending) <= 1:
        # Sequential path: bit-identical to historical behaviour when
        # ``--parallel-workers`` is omitted (default 1).
        for index, vector in pending:
            label = f"sample_{index + 1:03d}"
            print(f"== {label} ({index + 1}/{len(samples)}) ==")
            result = _run_simple_foam_for_vector(
                label=label,
                vector=vector,
                baseline_mesh=baseline_mesh,
                region=region,
                deformer=deformer,
                builder=builder,
                runner=runner,
                work_root=work_root,
                timeout_seconds=args.timeout,
            )
            _persist_result(label, vector, result)
    else:
        # Parallel path: split pending samples into N chunks, dispatch
        # each to a ProcessPoolExecutor worker, aggregate the manifests
        # back here so history.jsonl writes happen in a single process.
        workers = max(1, min(int(args.parallel_workers), len(pending)))
        # Round-robin chunking so neighbouring LHS indices don't all
        # land on the same worker (better utilisation if some samples
        # take longer than others).
        chunks: List[List[Tuple[int, KrachtVector]]] = [[] for _ in range(workers)]
        for offset, (index, vector) in enumerate(pending):
            chunks[offset % workers].append((index, vector))

        repaired_path = case_dir / "working" / "repaired" / "repaired.stl"

        payloads: List[dict] = []
        for w_index, chunk in enumerate(chunks):
            if not chunk:
                continue
            payloads.append(
                {
                    "case_dir": str(case_dir),
                    "work_root": str(work_root),
                    "doe_results_dir": str(doe_results_dir),
                    "repaired_stl": str(repaired_path),
                    "region": region,
                    "timeout": int(args.timeout),
                    "items": [
                        {
                            "label": f"sample_{i + 1:03d}",
                            "vector": dict(vec.values),
                            "index": i,
                        }
                        for i, vec in chunk
                    ],
                    "worker_id": w_index,
                }
            )

        print(
            f"== dispatching {len(pending)} samples across "
            f"{len(payloads)} parallel workers =="
        )
        with ProcessPoolExecutor(max_workers=len(payloads)) as executor:
            futures = {
                executor.submit(_worker_run_chunk, payload): payload
                for payload in payloads
            }
            for future in as_completed(futures):
                payload = futures[future]
                try:
                    chunk_results = future.result()
                except Exception as exc:  # pragma: no cover - defensive
                    print(
                        f"  worker {payload.get('worker_id')} crashed: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    # Mark every sample in the chunk as failed so we still
                    # write their manifests.
                    for item in payload["items"]:
                        synthetic = {
                            "label": item["label"],
                            "vector": item["vector"],
                            "cd": None,
                            "status": "failed",
                            "reason": f"worker_crashed: {type(exc).__name__}: {exc}",
                        }
                        vec = KrachtVector(
                            values={k: float(v) for k, v in item["vector"].items()}
                        )
                        _persist_result(
                            item["label"], vec, synthetic, prefix_label=True
                        )
                    continue
                # Map back results to (index, vector) by label so we
                # write history rows in the parent and avoid concurrent
                # writers clobbering each other.
                label_to_vec = {
                    item["label"]: KrachtVector(
                        values={k: float(v) for k, v in item["vector"].items()}
                    )
                    for item in payload["items"]
                }
                for result in chunk_results:
                    label = result.get("label")
                    vec = label_to_vec.get(label)
                    if vec is None:
                        # Fall back to reconstructing from result payload.
                        try:
                            vec = KrachtVector(
                                values={
                                    k: float(v)
                                    for k, v in (result.get("vector") or {}).items()
                                }
                            )
                        except Exception:  # pragma: no cover - defensive
                            n_failed += 1
                            continue
                    _persist_result(label, vec, result, prefix_label=True)

    total_seconds = time.monotonic() - start_time

    summary = {
        "n": int(args.n),
        "n_succeeded": int(n_succeeded),
        "n_failed": int(n_failed),
        "n_skipped_resume": int(n_skipped),
        "baseline_cd": float(baseline_cd) if baseline_cd is not None else None,
        "min_cd": float(min(cd_values)) if cd_values else None,
        "max_cd": float(max(cd_values)) if cd_values else None,
        "mean_cd": float(statistics.fmean(cd_values)) if cd_values else None,
        "stdev_cd": (
            float(statistics.stdev(cd_values)) if len(cd_values) >= 2 else None
        ),
        "history_path": str(history_path),
        "total_seconds": float(total_seconds),
        "bounds": args.bounds,
        "seed": int(args.seed),
        "source_stl": str(source_stl),
        "parallel_workers": int(args.parallel_workers),
        "resume": bool(args.resume),
    }
    summary_path = case_dir / "seed_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("=" * 78)
    print(f"DOE seed sweep complete in {total_seconds:.1f}s")
    print(f"  succeeded: {n_succeeded}/{args.n}")
    print(f"  failed:    {n_failed}/{args.n}")
    if n_skipped:
        print(f"  skipped (--resume): {n_skipped}/{args.n}")
    if baseline_cd is not None:
        print(f"  baseline Cd: {baseline_cd:.4f}")
    if cd_values:
        print(
            f"  Cd: min={min(cd_values):.4f} mean={statistics.fmean(cd_values):.4f} "
            f"max={max(cd_values):.4f}"
        )
    print(f"  history file: {history_path}")
    print(f"  summary: {summary_path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
