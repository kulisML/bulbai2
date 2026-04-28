"""``run_night_optimization`` — night-run entry point.

Design reference: 2026-04-22-bulbopt-night-optimization-design.md §11.
Mesh-quality follow-ups: 2026-04-23-bulbopt-mesh-quality-design.md §4
(L1 history + GP warm start, L2 quality objective, L4 validity prefilter,
L6 STL sanity export).

End-to-end flow:

  1. Create a new case and persist the source STL via the existing
     ``create_case`` use case.
  2. Run ``prepare_geometry`` once to detect the bulb region and write
     ``working/repaired/repaired.stl``.
  3. Build a cascade:
       * mid gate = cheap mesh-derived proxy (slenderness vs. volume
         + mesh quality); GP surrogate replaces objective[0] when enough
         history exists; validity prefilter rejects obviously broken
         candidates before the deformer runs.
       * high gate = caller-supplied (defaults to a numeric proxy)
     NSGA-II explores the Kracht space, the scheduler tracks time.
  4. Persist:
       * ``working/night_optimization/pareto_front.json``
       * ``working/night_optimization/high_fidelity_results.json``
       * ``working/night_optimization/budget_trace.json``
  5. Export the top-ranked high-fidelity STL + each Pareto candidate STL
     under ``outputs/top_candidates/candidate-XXX/`` plus a
     ``stl_valid.json`` sanity report (L6).
  6. Update the case to ``completed`` / ``completed_with_warnings`` and
     return a ``CaseSummary`` pointing at the winner.

The default high-fidelity evaluator is a parameter-only proxy (identical
in cost to mid gate) so unit tests run in milliseconds. The real
``simple_foam_runner`` replaces this evaluator in Task 6.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from typing import Callable, List, Sequence

import trimesh

from bulbopt.application.contracts.models import CaseSummary, CreateCaseCommand
from bulbopt.application.use_cases.create_case import create_case
from bulbopt.domain.core.models import CaseStatus
from bulbopt.execution.logging.case_logger import CaseLogger
from bulbopt.infrastructure.adapters.html_report import HtmlReportAdapter
from bulbopt.infrastructure.adapters.openfoam_adapter import (
    OpenFOAMAdapter,
    detect_openfoam_available,
)
from bulbopt.infrastructure.adapters.openfoam_runner import OpenFOAMRunnerAdapter
from bulbopt.infrastructure.adapters.simple_foam_gate import (
    SimpleFoamHighFidelityGate,
    _read_force_coeffs,
)
from bulbopt.infrastructure.adapters.stl_sanity import validate_stl
from bulbopt.infrastructure.adapters.stub_geometry import StubGeometryAdapter
from bulbopt.optimization.learning.gp_surrogate import GPSurrogate, MIN_TRAIN_POINTS
from bulbopt.optimization.learning.cfd_evidence_store import CFDEvidenceStore
from bulbopt.optimization.learning.history_store import (
    HistoryStore,
    default_history_path,
)
from bulbopt.optimization.learning.validity_classifier import (
    MIN_TRAINING_SAMPLES,
    ValidityClassifier,
)
from bulbopt.optimization.quality.mesh_metrics import compute_mesh_quality
from bulbopt.optimization.parametric.ffd_deformer import BulbFFDDeformer
from bulbopt.optimization.parametric.isotropic_remesh import (
    is_isotropic_remesh_available,
    remesh_region_isotropic,
)
from bulbopt.optimization.parametric.kracht_space import (
    KRACHT_PARAMETER_NAMES,
    KrachtDesignSpace,
    KrachtVector,
)
from bulbopt.optimization.scheduler.budget_scheduler import BudgetScheduler
from bulbopt.reporting.fuel_savings import project_fuel_savings
from bulbopt.reporting.pareto_plot import render_pareto_plot_svg
from bulbopt.optimization.strategies.cascade_strategy import (
    CascadeResult,
    CascadeStrategy,
    Gate,
)
from bulbopt.storage.filesystem.json_store import JsonStore
from bulbopt.storage.project_repository.filesystem_repository import (
    FilesystemProjectRepository,
)


HighFidelityEvaluator = Callable[[List[KrachtVector]], List[List[float]]]


# Validity history filename, kept under ``<project_root>/.history`` so a single
# history accumulates across every night-run for the same project root (see
# 2026-04-23 mesh-quality spec §4 L4). One JSON object per line: ``{"vector":
# [..8 floats..], "invalid": 0 | 1}``.
VALIDITY_HISTORY_FILENAME: str = "validity_history.jsonl"
CFD_EVIDENCE_FILENAME: str = "cfd_evidence.jsonl"


@dataclass(slots=True)
class NightOptimizationConfig:
    """All the knobs for a night run that aren't in CreateCaseCommand."""
    population: int = 50
    generations: int = 20
    high_fidelity_budget: int = 10
    runtime_budget_hours: float = 8.0
    seed: int | None = None
    mid_gate_estimated_seconds_per_eval: float = 5.0
    high_gate_estimated_seconds_per_eval: float = 300.0
    # L4: validity classifier prefilter. Enabled by default; reject
    # candidates whose predicted p(invalid) exceeds the threshold. Set
    # ``enable_validity_prefilter=False`` to disable (e.g. when sanity-
    # check costs dominate prediction accuracy).
    enable_validity_prefilter: bool = True
    validity_reject_threshold: float = 0.7
    # L1: number of best historical vectors to seed the initial GA
    # population with. Falls back to pure random sampling if the history
    # store is empty.
    warm_start_top_k: int = 10
    # Keep generation zero diverse: only this fraction of the population may
    # come from historical elites. The rest stays random/exploratory.
    warm_start_ratio: float = 0.3
    # Minimum normalised Euclidean distance between seeded vectors.
    warm_start_dedup_distance: float = 0.02
    # Mutated variants around elites fill part of generation zero without
    # collapsing fully into clones.
    warm_start_mutation_ratio: float = 0.2
    warm_start_mutation_sigma: float = 0.05
    # Hard floor for random generation-zero exploration.
    random_exploration_ratio: float = 0.5
    # L1: minimum history size before the GP surrogate replaces the
    # analytic mid-gate proxy.
    #
    # Lowered from 20 -> 12 by audit 2026-04-26 follow-up: the DOE bootstrap
    # produces 30 simpleFoam rows with some failures (~25-28 valid), and a
    # threshold of 20 left runs blind on the analytic proxy (which is
    # anti-correlated with real Cd) for too long. 12 rows is enough for a
    # rough RBF length-scale fit and gets the GP active sooner while the
    # DOE is still completing.
    gp_surrogate_min_history: int = 12
    # L1: override the history path (default ``~/.bulbopt/history.jsonl``)
    history_path: Path | None = None
    # Spec 2026-04-22 §8: parallel mid-gate fan-out. ``1`` (default) keeps
    # the cascade fully sequential and bit-identical to the legacy path.
    # Setting this to e.g. 4 dispatches each generation's mid-gate
    # evaluation across 4 worker processes — population × generations
    # × seconds_per_eval ÷ workers. Falls back to sequential when the
    # wrapped evaluator is not picklable (typical for closures over
    # trimesh meshes).
    parallel_workers: int = 1
    # Visual-only post-processing flag for the top-candidate STL exports
    # under ``outputs/top_candidates/candidate-XXX/geometry.stl``. When
    # ``True`` the saved STL is run through
    # :func:`bulbopt.optimization.parametric.isotropic_remesh.remesh_region_isotropic`
    # (a meshlib-backed edge-flip / split / collapse remesher) which
    # drops the bulb-region max aspect ratio from ~3650 to ~6 on the
    # production hull while keeping the rest of the upper hull
    # bit-identical to the deformed input. CFD-evaluated geometry is
    # NOT touched — the polish runs only on the STL written to disk
    # for visual review, so the GP training history stays consistent.
    # Default ``False`` to keep behaviour identical to existing callers
    # and CI runs that don't have ``meshlib`` installed.
    apply_isotropic_polish_to_outputs: bool = False


@dataclass(slots=True)
class ResumeState:
    """Internal state passed between ``resume_night_optimization`` and
    ``run_night_optimization`` (spec 2026-04-22 §10.3).

    ``starting_generation`` is the on-disk index of the FIRST generation
    the resumed run should produce (i.e. the killed run's last completed
    generation index + 1). ``population`` is the list of vectors that
    should seed NSGA-II's gen 0 — typically the population from the last
    snapshot the killed run wrote.

    ``case`` and ``case_dir`` are the already-loaded case to mutate (no
    new ``create_case`` call is performed when this state is supplied).
    The geometry is also expected to already be on disk under
    ``case_dir/working/repaired/``.
    """

    starting_generation: int
    population: List[KrachtVector]
    case: object
    case_dir: Path


def run_night_optimization(
    *,
    project_root: Path,
    command: CreateCaseCommand,
    config: NightOptimizationConfig | None = None,
    high_fidelity_evaluator: HighFidelityEvaluator | None = None,
    _resume_state: ResumeState | None = None,
) -> CaseSummary:
    """Spec 2026-04-22 §11. Public signature is ``project_root, command,
    config, high_fidelity_evaluator``. ``_resume_state`` is an internal
    keyword for use by :func:`resume_night_optimization` only — it must
    NOT be invoked by external callers, hence the leading underscore.
    When supplied:

    * ``create_case`` is skipped (the case already exists).
    * ``prepare_geometry`` is skipped (the repaired mesh is already on
      disk; re-running would mutate it).
    * NSGA-II is seeded with ``_resume_state.population`` and the
      generation snapshot writer is offset by
      ``_resume_state.starting_generation`` so new ``gen-NN/`` directories
      pick up where the killed run left off.
    * ``convergence.csv`` is appended to instead of overwritten.
    """
    config = config or NightOptimizationConfig()
    repository = FilesystemProjectRepository(root_dir=project_root)

    if _resume_state is None:
        case = create_case(command=command, repository=repository)
        case_dir = repository.case_dir(case.case_id)
    else:
        case = _resume_state.case  # type: ignore[assignment]
        case_dir = _resume_state.case_dir

    is_resume = _resume_state is not None
    generation_offset = (
        int(_resume_state.starting_generation) if _resume_state is not None else 0
    )

    json_store = JsonStore()
    case_logger = CaseLogger(case_dir / "logs" / "case.log")
    case_logger.log_stage(
        stage="pipeline",
        status="resumed" if is_resume else "started",
        extra={
            "case_id": case.case_id,
            "mode": "night_optimization",
            "resumed": is_resume,
            "starting_generation": generation_offset,
        },
    )

    # Spec 2026-04-22 §10.1: enter the dedicated night state once the case
    # has been created. This lets ``bulbopt list`` show "running_night_*"
    # while the run is in flight, and lets a crashed run that persists in
    # FAILED be told apart from a vertical-slice failure.
    case.status = CaseStatus.RUNNING_NIGHT_OPTIMIZATION
    case.is_recoverable = True
    repository.save_case(case)

    # Per-generation convergence rows accumulate here as the snapshot
    # callback fires; the use case writes the final ``convergence.csv``
    # after the cascade exits.
    convergence_rows: list[dict] = []

    try:
        # Stage 1: prepare base geometry — but only on a fresh run. The
        # repaired mesh and analysis are already on disk for resume, and
        # re-running ``prepare_geometry`` would mutate the repaired STL.
        if is_resume:
            case_logger.log_stage(
                stage="prepare_geometry",
                status="skipped",
                extra={"reason": "resumed"},
            )
            geometry_analysis = json_store.read(
                case_dir / "working" / "repaired" / "geometry_analysis.json"
            )
        else:
            case_logger.log_stage(stage="prepare_geometry", status="started")
            geometry = StubGeometryAdapter()
            geometry_analysis = geometry.prepare_geometry(case_dir, Path(command.source_path))
            case_logger.log_stage(stage="prepare_geometry", status="completed")

        # Stage 2: cascade.
        case_logger.log_stage(stage="night_optimization", status="started")
        hull_fingerprint = _file_sha256(Path(command.source_path))
        space = KrachtDesignSpace()
        deformer = BulbFFDDeformer()
        repaired_mesh = trimesh.load(
            case_dir / "working" / "repaired" / "repaired.stl",
            force="mesh",
        )
        region = geometry_analysis["bulb_region"]

        # L4: load / train validity classifier from prior HF history.
        # When the project has fewer than MIN_TRAINING_SAMPLES examples
        # this becomes a no-op; the wrapper still records new labels so
        # the next run benefits.
        validity_history_path = _validity_history_path(project_root)
        validity_classifier = _load_and_train_validity_classifier(validity_history_path)

        # L1: load historical (vector, cd) pairs for warm-start + GP.
        history_store = HistoryStore(
            path=config.history_path if config.history_path is not None else None
        )
        evidence_store = CFDEvidenceStore(_cfd_evidence_history_path(project_root))
        openfoam_available = (
            high_fidelity_evaluator is None and detect_openfoam_available()
        )
        planned_high_gate_backend = (
            "simple_foam"
            if openfoam_available
            else ("external" if high_fidelity_evaluator is not None else "surrogate")
        )
        # Bug #5 (audit 2026-04-26): older HistoryStore JSONLs hold a mix
        # of "real OpenFOAM Cd" (~0.3-1.5) and "(beam*draft)/axial proxy"
        # (~0.01-0.5) rows. Training one GP on both corrupts the kernel
        # length-scales. Prefer rows tagged with the planned high-gate
        # backend; only fall back to the full set when no clean rows
        # exist (e.g. the very first run after upgrade), and even then
        # only for warm-start, never for GP fit (see below).
        history_rows = history_store.load_all()
        history_rows_for_gp = history_store.load_all(
            backend=planned_high_gate_backend
        )
        settings_hash = _solver_settings_hash(backend=planned_high_gate_backend)
        surrogate_training_rows = (
            evidence_store.surrogate_training_pairs(
                hull_fingerprint=hull_fingerprint,
                settings_hash=settings_hash,
            )
            + history_rows_for_gp
        )
        raw_warm_start_vectors = evidence_store.top_k_safe_warm_start(
            config.warm_start_top_k,
            hull_fingerprint=hull_fingerprint,
            settings_hash=settings_hash,
        )
        evidence_eligibility = evidence_store.warm_start_eligibility_summary(
            hull_fingerprint=hull_fingerprint,
            settings_hash=settings_hash,
        )
        if not raw_warm_start_vectors:
            raw_warm_start_vectors = history_store.top_k(config.warm_start_top_k)
        warm_start_vectors, warm_start_summary = _select_warm_start_vectors(
            raw_vectors=raw_warm_start_vectors,
            space=space,
            population=config.population,
            ratio=config.warm_start_ratio,
            dedup_distance=config.warm_start_dedup_distance,
            mutation_ratio=config.warm_start_mutation_ratio,
            mutation_sigma=config.warm_start_mutation_sigma,
            random_exploration_ratio=config.random_exploration_ratio,
            seed=config.seed,
        )
        warm_start_summary["evidence_eligibility"] = evidence_eligibility
        case_logger.log_stage(
            stage="night_optimization_warm_start",
            status="loaded" if warm_start_vectors else "empty",
            extra={
                "history_size": len(history_rows),
                "evidence_eligibility": evidence_eligibility,
                **warm_start_summary,
            },
        )

        base_mid_evaluator = _mid_gate_evaluator(repaired_mesh, region, deformer)

        # L1: wrap the mid-gate with a GP prediction once the history
        # crosses the threshold; otherwise keep the analytic proxy.
        #
        # Audit 2026-04-26 follow-up: also emit a ``mid_gate_routing`` log
        # entry so an engineer reading ``case.log`` can tell at a glance
        # which mid-gate is in play. The proxy fallback is anti-correlated
        # with real Cd in some regions, so the GA is "running blind" when
        # it fires — the warning makes that obvious.
        if len(surrogate_training_rows) >= config.gp_surrogate_min_history:
            surrogate = GPSurrogate()
            surrogate.fit(
                [vec for vec, _cd in surrogate_training_rows],
                [cd for _vec, cd in surrogate_training_rows],
            )
            mid_evaluator = _wrap_mid_gate_with_gp(base_mid_evaluator, surrogate)
            case_logger.log_stage(
                stage="night_optimization_surrogate",
                status="trained",
                extra={"training_points": len(surrogate_training_rows)},
            )
            case_logger.log_stage(
                stage="mid_gate_routing",
                status="gp",
                extra={
                    "training_points": len(history_rows_for_gp),
                    "threshold": config.gp_surrogate_min_history,
                },
            )
        else:
            mid_evaluator = base_mid_evaluator
            case_logger.log_stage(
                stage="mid_gate_routing",
                status="proxy_fallback_blind",
                extra={
                    "history_size": len(history_rows_for_gp),
                    "threshold": config.gp_surrogate_min_history,
                    "warning": (
                        "Mid-gate using analytic proxy that may "
                        "anti-correlate with real Cd. Increase real-CFD "
                        "history (run scripts/cfd_doe_seed.py) to "
                        "activate GP."
                    ),
                },
            )

        # L4: validity prefilter wraps whichever evaluator we ended up with.
        if config.enable_validity_prefilter:
            mid_evaluator = _with_validity_prefilter(
                mid_evaluator,
                classifier=validity_classifier,
                reject_threshold=config.validity_reject_threshold,
                history_path=validity_history_path,
                baseline_mesh=repaired_mesh,
                region=region,
                deformer=deformer,
                case_logger=case_logger,
            )

        mid_gate = Gate(
            name="mid",
            evaluate=mid_evaluator,
            estimated_seconds_per_eval=config.mid_gate_estimated_seconds_per_eval,
        )
        baseline_cfd_result: dict | None = None

        foam_gate: SimpleFoamHighFidelityGate | None = None
        if high_fidelity_evaluator is not None:
            high_eval = high_fidelity_evaluator
            high_gate_backend = "external"
        elif openfoam_available:
            case_logger.log_stage(
                stage="high_gate",
                status="initialised",
                extra={"backend": "simple_foam"},
            )
            foam_work_root = case_dir / "working" / "night_optimization" / "foam_candidates"
            foam_work_root.mkdir(parents=True, exist_ok=True)
            builder = OpenFOAMAdapter()
            runner = OpenFOAMRunnerAdapter()
            foam_gate = SimpleFoamHighFidelityGate(
                work_root=foam_work_root,
                baseline_mesh=repaired_mesh,
                region=region,
                deformer=deformer,
                build_case=builder.build_case,
                run_case=runner.run_case,
            )
            high_eval = foam_gate.evaluate
            high_gate_backend = "simple_foam"
        else:
            case_logger.log_stage(
                stage="high_gate",
                status="initialised",
                extra={"backend": "surrogate", "reason": "openfoam_unavailable"},
            )
            high_eval = _default_high_evaluator(repaired_mesh, region, deformer)
            high_gate_backend = "surrogate"
        high_gate = Gate(
            name="high",
            evaluate=high_eval,
            estimated_seconds_per_eval=config.high_gate_estimated_seconds_per_eval,
        )

        scheduler = BudgetScheduler(runtime_budget_hours=config.runtime_budget_hours)
        generation_writer = _make_generation_writer(
            case_dir=case_dir,
            repository=repository,
            case=case,
            convergence_rows=convergence_rows,
            case_logger=case_logger,
            generation_offset=generation_offset,
        )
        # Spec 2026-04-22 §10.3: when resuming, NSGA-II is seeded with the
        # exact population from the last snapshot the killed run wrote.
        # This bypasses the history-driven warm-start logic above so the
        # resume behaviour is deterministic and reproduces the trajectory
        # of the original run.
        cascade_warm_start_vectors = (
            list(_resume_state.population)
            if _resume_state is not None
            else warm_start_vectors
        )
        cascade = CascadeStrategy(
            space=space,
            scheduler=scheduler,
            population=config.population,
            generations=config.generations,
            high_fidelity_budget=config.high_fidelity_budget,
            mid_gate=mid_gate,
            high_gate=high_gate,
            seed=config.seed,
            warm_start_vectors=cascade_warm_start_vectors,
            n_objectives=3,
            on_generation_snapshot=generation_writer,
            parallel_workers=config.parallel_workers,
        )
        result: CascadeResult = cascade.run()

        # Persist the run-level convergence series. Empty when no
        # generation snapshot fired (e.g. the test harness skipped the
        # callback) — but the file still gets a header row so consumers
        # don't have to special-case missing files. Append mode (resume)
        # preserves the killed run's earlier rows above the new ones.
        _write_convergence_csv(
            case_dir=case_dir,
            rows=convergence_rows,
            case_logger=case_logger,
            append=is_resume,
        )

        if _should_run_baseline_cfd(
            openfoam_available=openfoam_available,
            high_fidelity_results=result.high_fidelity_results,
        ):
            baseline_cfd_result = _run_baseline_simple_foam(
                baseline_work_dir=case_dir
                / "working"
                / "night_optimization"
                / "baseline_cfd",
                geometry_path=case_dir / "working" / "repaired" / "repaired.stl",
                build_case=builder.build_case,
                run_case=runner.run_case,
            )
            if baseline_cfd_result is not None:
                case_logger.log_stage(
                    stage="baseline_cfd",
                    status="completed",
                    extra={"final_cd": baseline_cfd_result.get("final_cd")},
                )

        # L1: append every high-fidelity (vector, cd) pair to the history
        # JSONL so the next night-run can warm-start from it.
        # Bug #5 fix: tag every row with the high-gate backend so the GP
        # surrogate can later filter to a single Cd scale (real OpenFOAM
        # Cd vs the analytic proxy live in different ranges).
        for hf in _engineering_valid_candidates(result.high_fidelity_results):
            if hf.objectives:
                history_store.record(
                    hf.vector,
                    cd=float(hf.objectives[0]),
                    backend=high_gate_backend,
                )
        case_logger.log_stage(
            stage="night_optimization",
            status="completed",
            extra={
                "pareto_candidates": len(result.pareto_front.candidates),
                "high_fidelity_candidates": len(result.high_fidelity_results),
                "budget_exhausted": result.budget_exhausted,
            },
        )

        # Stage 3: persist outputs.
        night_dir = case_dir / "working" / "night_optimization"
        night_dir.mkdir(parents=True, exist_ok=True)
        json_store.write(
            night_dir / "pareto_front.json",
            {
                "candidates": [
                    {
                        "vector": [
                            candidate.vector.values[name]
                            for name in KRACHT_PARAMETER_NAMES
                        ],
                        "parameters": dict(candidate.vector.values),
                        "objectives": list(candidate.objectives),
                    }
                    for candidate in result.pareto_front.candidates
                ],
            },
        )
        foam_evaluation_records = (
            list(foam_gate.evaluation_records) if foam_gate is not None else []
        )
        high_fidelity_rows = []
        for index, r in enumerate(result.high_fidelity_results):
            foam_record = (
                dict(foam_evaluation_records[index])
                if index < len(foam_evaluation_records)
                else None
            )
            high_fidelity_rows.append(
                {
                    "vector": [r.vector.values[name] for name in KRACHT_PARAMETER_NAMES],
                    "parameters": dict(r.vector.values),
                    "objectives": list(r.objectives),
                    "geometry": _candidate_geometry_evidence(
                        candidate=r,
                        baseline_mesh=repaired_mesh,
                        region=region,
                        deformer=deformer,
                        foam_record=foam_record,
                    ),
                }
            )
        baseline_cd = (
            float(baseline_cfd_result["final_cd"])
            if baseline_cfd_result is not None
            and baseline_cfd_result.get("final_cd") is not None
            else None
        )
        engineering_summary = _baseline_improvement_summary(
            baseline_cd=baseline_cd,
            high_fidelity_rows=high_fidelity_rows,
        )
        json_store.write(
            night_dir / "high_fidelity_results.json",
            {
                "baseline_cfd": baseline_cfd_result,
                "engineering_summary": engineering_summary,
                "results": high_fidelity_rows,
            },
        )
        json_store.write(
            night_dir / "budget_trace.json",
            {
                "trace": scheduler.trace(),
                "gate_timings": scheduler.gate_timings(),
                "runtime_budget_hours": config.runtime_budget_hours,
                "remaining_seconds": scheduler.remaining_seconds,
                "budget_exhausted": result.budget_exhausted,
            },
        )

        # Stage 4: write STL per top candidate + pick winner.
        winner_id, stl_invalid_candidates = _persist_top_candidate_meshes(
            case_dir=case_dir,
            result=result,
            repaired_mesh=repaired_mesh,
            region=region,
            deformer=deformer,
            apply_isotropic_polish=config.apply_isotropic_polish_to_outputs,
        )
        rejected_candidates = _persist_rejected_candidate_meshes(
            case_dir=case_dir,
            candidates=result.high_fidelity_results,
            repaired_mesh=repaired_mesh,
            region=region,
            deformer=deformer,
        )
        winner_id = _winner_after_baseline_check(
            winner_id=winner_id,
            engineering_summary=engineering_summary,
        )
        engineering_outcome = _engineering_outcome(
            winner_id=winner_id,
            engineering_summary=engineering_summary,
        )
        high_fidelity_payload = json_store.read(night_dir / "high_fidelity_results.json")
        high_fidelity_payload["engineering_outcome"] = engineering_outcome
        high_fidelity_payload["rejected_candidates"] = rejected_candidates
        json_store.write(night_dir / "high_fidelity_results.json", high_fidelity_payload)

        evidence_rows = _cfd_evidence_rows(
            case_id=case.case_id,
            case_name=case.case_name,
            source_path=command.source_path,
            hull_fingerprint=hull_fingerprint,
            settings_hash=settings_hash,
            high_fidelity_results=result.high_fidelity_results,
            backend=high_gate_backend,
            baseline_cfd_result=baseline_cfd_result,
            engineering_outcome=engineering_outcome,
            rejected_candidates=rejected_candidates,
            foam_evaluation_records=foam_evaluation_records,
            baseline_mesh=repaired_mesh,
            region=region,
            deformer=deformer,
        )
        CFDEvidenceStore(night_dir / CFD_EVIDENCE_FILENAME).write_all(evidence_rows)
        CFDEvidenceStore(_cfd_evidence_history_path(project_root)).append_many(
            evidence_rows
        )
        case_logger.log_stage(
            stage="night_optimization_outputs",
            status="completed",
            extra={
                "winner": winner_id or "n/a",
                "stl_invalid_count": len(stl_invalid_candidates),
                "rejected_count": len(rejected_candidates),
                "cfd_evidence_rows": len(evidence_rows),
            },
        )

        # Stage 5: render the HTML night-run report.
        try:
            _render_night_report(
                case_dir=case_dir,
                command=command,
                config=config,
                result=result,
                scheduler=scheduler,
                winner_id=winner_id,
                high_gate_backend=high_gate_backend,
                stl_invalid_candidates=stl_invalid_candidates,
                rejected_candidates=rejected_candidates,
                engineering_summary=engineering_summary,
                engineering_outcome=engineering_outcome,
                warm_start_summary=warm_start_summary,
                repaired_mesh=repaired_mesh,
                region=region,
            )
            case_logger.log_stage(
                stage="build_night_report",
                status="completed",
                extra={"path": str(case_dir / "outputs" / "reports" / "night_report.html")},
            )
        except Exception as report_error:
            case_logger.log_stage(
                stage="build_night_report",
                status="failed",
                extra={"error": str(report_error)},
            )

        case.status = (
            CaseStatus.COMPLETED_WITH_WARNINGS
            if result.budget_exhausted or winner_id is None
            else CaseStatus.COMPLETED
        )
        case.is_recoverable = False
        case.summary_metrics = {
            "night_optimization": {
                "pareto_size": len(result.pareto_front.candidates),
                "high_fidelity_size": len(result.high_fidelity_results),
                "budget_exhausted": result.budget_exhausted,
                "winner_id": winner_id,
                "warm_start": warm_start_summary,
                "gate_timings": scheduler.gate_timings(),
                "engineering_summary": engineering_summary,
                "engineering_outcome": engineering_outcome,
                # Spec 2026-04-22 §10.3: distinguish a fresh run from a
                # resumed one so downstream tooling (and tests) can
                # correlate the on-disk gen-NN sequence with how it was
                # produced.
                "resumed": is_resume,
                "starting_generation": int(generation_offset),
            }
        }
        repository.save_case(case)
        case_logger.log_stage(
            stage="pipeline",
            status=case.status.value,
            extra={"winner": winner_id or "n/a"},
        )
    except Exception as exc:
        case.status = CaseStatus.FAILED
        case.is_recoverable = True
        repository.save_case(case)
        case_logger.log_stage(
            stage="pipeline",
            status="failed",
            extra={"error": str(exc), "is_recoverable": True},
        )
        raise

    return CaseSummary(
        case_id=case.case_id,
        case_name=case.case_name,
        status=case.status.value,
        best_candidate_id=winner_id,
    )


# --- gates ---------------------------------------------------------------


def _mid_gate_evaluator(
    baseline_mesh: trimesh.Trimesh,
    region: dict,
    deformer: BulbFFDDeformer,
) -> Callable[[Sequence[KrachtVector]], List[List[float]]]:
    """Mid-fidelity proxy: deform mesh, read slenderness + volume delta
    + mesh quality (L2). All three objectives minimised."""

    baseline_volume = _mesh_volume(baseline_mesh)
    design_space = KrachtDesignSpace()

    def evaluate(vectors: Sequence[KrachtVector]) -> List[List[float]]:
        objectives: List[List[float]] = []
        for vector in vectors:
            if design_space.constraint_violations(vector):
                objectives.append([1e9, 1e9, 1e9])
                continue
            deformed = deformer.deform(baseline_mesh, region, vector)
            stl_report = validate_stl(deformed)
            if not stl_report["checks_passed"]:
                objectives.append([1e9, 1e9, 1e9])
                continue
            extents = deformed.extents.astype(float)
            primary = int(region.get("axis_index", int(extents.argmax())))
            secondary = [i for i in range(3) if i != primary]
            axial = max(extents[primary], 1e-9)
            beam = max(extents[secondary[0]], 1e-9)
            draft = max(extents[secondary[1]], 1e-9)
            # Resistance proxy (smaller is better): wider & deeper bulbs
            # trade drag for volume.
            resistance_proxy = (beam * draft) / axial
            # Volume delta (preserve displacement). Smaller abs is better.
            deformed_volume = _mesh_volume(deformed)
            volume_delta = abs(deformed_volume - baseline_volume) / max(baseline_volume, 1e-6)
            # L2: mesh quality — broken meshes get >= 100 and dominated.
            # Pass the region's resolved beam_axis so the symmetry metric
            # mirrors around the actual port-starboard axis of the hull
            # (audit 2026-04-26: the default argmin(extents) heuristic
            # picks the wrong axis on real ship hulls). Pass ``region``
            # so the new bulb/aft self-intersection penalty (audit
            # 2026-04-26 Add #3) can identify the bulb sub-mesh.
            mesh_quality = compute_mesh_quality(
                deformed,
                beam_axis=region.get("beam_axis"),
                region=region,
            )
            manufacturability_penalty = _manufacturability_penalty(
                design_space.manufacturability_warnings(vector)
            )
            resistance_proxy *= 1.0 + manufacturability_penalty
            mesh_quality += manufacturability_penalty
            objectives.append(
                [float(resistance_proxy), float(volume_delta), float(mesh_quality)]
            )
        return objectives

    return evaluate


def _manufacturability_penalty(warnings: Sequence[str]) -> float:
    """Small soft penalty for valid candidates near risky parameter limits."""
    if not warnings:
        return 0.0
    return min(0.30, 0.03 * float(len(warnings)))


def _default_high_evaluator(
    baseline_mesh: trimesh.Trimesh,
    region: dict,
    deformer: BulbFFDDeformer,
) -> Callable[[Sequence[KrachtVector]], List[List[float]]]:
    """Placeholder high-fidelity gate: same proxy as mid for fast tests.
    Task 6 replaces this with a simpleFoam runner."""
    return _mid_gate_evaluator(baseline_mesh, region, deformer)


def _run_baseline_simple_foam(
    *,
    baseline_work_dir: Path,
    geometry_path: Path,
    build_case: Callable[..., dict],
    run_case: Callable[..., dict],
) -> dict | None:
    """Run the undeformed baseline through the same OpenFOAM adapter.

    Returns the parsed final Cd report, or ``None`` when the solver did
    not complete or force coefficients were unavailable.
    """
    try:
        manifest = build_case(
            baseline_work_dir,
            best_candidate_id="baseline",
            best_candidate_geometry_path=geometry_path,
        )
        foam_case_dir = baseline_work_dir / "working" / "openfoam_case"
        run_manifest = run_case(
            foam_case_dir,
            case_manifest=manifest,
            execute=True,
        )
    except Exception:
        return None

    if run_manifest.get("status") != "executed_ok":
        return None

    report = _read_force_coeffs(
        foam_case_dir,
        reference_velocity=None,
        reference_area=None,
        fluid_density=None,
    )
    if report is None:
        return None
    report["backend"] = "simple_foam"
    return report


def _should_run_baseline_cfd(
    *,
    openfoam_available: bool,
    high_fidelity_results: Sequence,
) -> bool:
    """Run baseline CFD only when it can be compared to a valid candidate."""
    return bool(
        openfoam_available
        and _engineering_valid_candidates(high_fidelity_results)
    )


def _baseline_improvement_summary(
    *,
    baseline_cd: float | None,
    high_fidelity_rows: Sequence[dict],
) -> dict | None:
    """Compare the best high-fidelity Cd against the baseline Cd."""
    if baseline_cd is None or baseline_cd <= 0:
        return None

    winner_cd: float | None = None
    for row in high_fidelity_rows:
        parameters = row.get("parameters")
        if parameters is not None:
            vector = KrachtVector(values=parameters)
            if KrachtDesignSpace().constraint_violations(vector):
                continue
        objectives = row.get("objectives") or []
        if not objectives:
            continue
        cd = float(objectives[0])
        if cd >= 1e8:
            continue
        winner_cd = cd if winner_cd is None else min(winner_cd, cd)

    if winner_cd is None:
        return None

    improvement = (float(baseline_cd) - winner_cd) / float(baseline_cd) * 100.0
    return {
        "baseline_cd": float(baseline_cd),
        "winner_cd": float(winner_cd),
        "improvement_percent": float(improvement),
    }


def _engineering_valid_candidates(candidates: Sequence) -> list:
    """Return candidates that can be treated as engineering winners.

    Penalty objectives and parameter constraint violations are not valid
    engineering results even if the optimizer promoted them.
    """
    space = KrachtDesignSpace()
    valid: list = []
    for candidate in candidates:
        objectives = list(getattr(candidate, "objectives", []) or [])
        if not objectives:
            continue
        if float(objectives[0]) >= 1e8:
            continue
        vector = getattr(candidate, "vector", None)
        if vector is None or space.constraint_violations(vector):
            continue
        valid.append(candidate)
    return valid


def _candidate_rejection_reasons(candidate) -> list[str]:
    reasons: list[str] = []
    objectives = list(getattr(candidate, "objectives", []) or [])
    if not objectives:
        reasons.append("missing_objectives")
    elif float(objectives[0]) >= 1e8:
        reasons.append("penalty_objective")

    vector = getattr(candidate, "vector", None)
    if vector is None:
        reasons.append("missing_vector")
    else:
        reasons.extend(KrachtDesignSpace().constraint_violations(vector))
    return reasons


def _winner_after_baseline_check(
    *,
    winner_id: str | None,
    engineering_summary: dict | None,
) -> str | None:
    """Only keep a winner when it improves over a measured baseline."""
    if winner_id is None or engineering_summary is None:
        return winner_id
    improvement = float(engineering_summary.get("improvement_percent", 0.0))
    if improvement <= 0.0:
        return None
    return winner_id


def _engineering_outcome(
    *,
    winner_id: str | None,
    engineering_summary: dict | None,
) -> dict:
    """Return machine-readable engineering decision status."""
    if engineering_summary is None:
        if winner_id is None:
            return {
                "status": "no_engineering_winner",
                "reason": "no_valid_candidate",
                "message": "No valid high-fidelity candidate is available.",
            }
        return {
            "status": "unverified_winner",
            "reason": "baseline_unavailable",
            "winner_id": winner_id,
            "message": "Candidate exists, but no baseline CFD comparison is available.",
        }

    improvement = float(engineering_summary.get("improvement_percent", 0.0))
    if winner_id is None or improvement <= 0.0:
        return {
            "status": "no_engineering_winner",
            "reason": "candidate_worse_than_baseline",
            "message": "Best CFD candidate is worse than the baseline.",
        }

    return {
        "status": "engineering_winner",
        "reason": "improves_baseline",
        "winner_id": winner_id,
        "message": "Best CFD candidate improves over the baseline.",
    }


def _cfd_evidence_history_path(project_root: Path) -> Path:
    return Path(project_root) / ".history" / CFD_EVIDENCE_FILENAME


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _solver_settings_hash(*, backend: str) -> str:
    payload = {
        "backend": backend,
        "parameter_schema": list(KRACHT_PARAMETER_NAMES),
        "geometry_gate_version": 1,
        "objective_schema": "cd_volume_mesh_quality_v1",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _select_warm_start_vectors(
    *,
    raw_vectors: Sequence[KrachtVector],
    space: KrachtDesignSpace,
    population: int,
    ratio: float,
    dedup_distance: float,
    mutation_ratio: float = 0.0,
    mutation_sigma: float = 0.05,
    random_exploration_ratio: float = 0.5,
    seed: int | None = None,
) -> tuple[list[KrachtVector], dict]:
    """Select a bounded, de-duplicated seed set for generation zero."""
    raw_vectors = list(raw_vectors)
    population = int(population)
    non_random_limit = _non_random_warm_start_limit(
        population=population,
        random_exploration_ratio=random_exploration_ratio,
    )
    max_seeded = min(
        _warm_start_seed_limit(population=population, ratio=ratio),
        non_random_limit,
    )
    selected: list[KrachtVector] = []
    rejected_by_constraints = 0
    deduplicated = 0
    skipped_by_ratio = 0

    for vector in raw_vectors:
        if not space.validate(vector):
            rejected_by_constraints += 1
            continue
        if _is_near_existing_seed(
            vector=vector,
            selected=selected,
            space=space,
            dedup_distance=dedup_distance,
        ):
            deduplicated += 1
            continue
        if len(selected) >= max_seeded:
            skipped_by_ratio += 1
            continue
        selected.append(vector)

    mutation_limit = min(
        _warm_start_seed_limit(population=population, ratio=mutation_ratio),
        max(non_random_limit - len(selected), 0),
    )
    mutated_vectors, mutation_skipped = _mutate_elite_vectors(
        elites=selected,
        space=space,
        count=mutation_limit,
        mutation_sigma=mutation_sigma,
        dedup_distance=dedup_distance,
        seed=seed,
    )
    warm_start_vectors = selected + mutated_vectors
    random_count = max(population - len(warm_start_vectors), 0)
    summary = {
        "requested": len(raw_vectors),
        "selected": len(selected),
        "warm_start": len(warm_start_vectors),
        "seeded": len(selected),
        "mutated": len(mutated_vectors),
        "random": random_count,
        "warm_start_ratio": float(ratio),
        "warm_start_mutation_ratio": float(mutation_ratio),
        "warm_start_mutation_sigma": float(mutation_sigma),
        "random_exploration_ratio": float(random_exploration_ratio),
        "warm_start_limit": int(max_seeded),
        "non_random_limit": int(non_random_limit),
        "deduplicated": deduplicated,
        "dedup_distance": float(dedup_distance),
        "rejected_by_constraints": rejected_by_constraints,
        "skipped_by_ratio": skipped_by_ratio,
        "mutation_skipped": mutation_skipped,
    }
    return warm_start_vectors, summary


def _warm_start_seed_limit(*, population: int, ratio: float) -> int:
    if population <= 0 or ratio <= 0.0:
        return 0
    return max(1, min(int(population), int(population * min(float(ratio), 1.0))))


def _non_random_warm_start_limit(
    *,
    population: int,
    random_exploration_ratio: float,
) -> int:
    if population <= 0:
        return 0
    random_floor = int(population * max(min(float(random_exploration_ratio), 1.0), 0.0))
    return max(int(population) - random_floor, 0)


def _mutate_elite_vectors(
    *,
    elites: Sequence[KrachtVector],
    space: KrachtDesignSpace,
    count: int,
    mutation_sigma: float,
    dedup_distance: float,
    seed: int | None,
) -> tuple[list[KrachtVector], int]:
    if not elites or count <= 0 or mutation_sigma <= 0.0:
        return [], 0
    rng = random.Random(seed)
    mutated: list[KrachtVector] = []
    skipped = 0
    attempts = 0
    max_attempts = max(count * 50, 50)
    while len(mutated) < count and attempts < max_attempts:
        attempts += 1
        elite = elites[(attempts - 1) % len(elites)]
        values: dict[str, float] = {}
        for name in KRACHT_PARAMETER_NAMES:
            lo, hi = space.bounds[name]
            width = float(hi) - float(lo)
            value = float(elite.values[name]) + rng.gauss(0.0, mutation_sigma) * width
            values[name] = min(max(value, float(lo)), float(hi))
        candidate = KrachtVector(values=values)
        if not space.validate(candidate):
            skipped += 1
            continue
        if _is_near_existing_seed(
            vector=candidate,
            selected=[*elites, *mutated],
            space=space,
            dedup_distance=dedup_distance,
        ):
            skipped += 1
            continue
        mutated.append(candidate)
    return mutated, skipped


def _is_near_existing_seed(
    *,
    vector: KrachtVector,
    selected: Sequence[KrachtVector],
    space: KrachtDesignSpace,
    dedup_distance: float,
) -> bool:
    if dedup_distance <= 0.0:
        return False
    return any(
        _normalised_parameter_distance(vector, existing, space) < dedup_distance
        for existing in selected
    )


def _normalised_parameter_distance(
    a: KrachtVector,
    b: KrachtVector,
    space: KrachtDesignSpace,
) -> float:
    total = 0.0
    for name in KRACHT_PARAMETER_NAMES:
        lo, hi = space.bounds[name]
        width = max(float(hi) - float(lo), 1e-12)
        delta = (float(a.values[name]) - float(b.values[name])) / width
        total += delta * delta
    return total ** 0.5


def _cfd_evidence_rows(
    *,
    case_id: str,
    case_name: str,
    source_path: str,
    hull_fingerprint: str,
    settings_hash: str,
    high_fidelity_results: Sequence,
    backend: str,
    baseline_cfd_result: dict | None,
    engineering_outcome: dict,
    rejected_candidates: Sequence[dict],
    foam_evaluation_records: Sequence[dict],
    baseline_mesh: trimesh.Trimesh,
    region: dict,
    deformer: BulbFFDDeformer,
) -> list[dict]:
    """Build durable high-fidelity evidence rows for audit and ML reuse."""
    created_at = datetime.now(timezone.utc).isoformat()
    baseline_cd = (
        float(baseline_cfd_result["final_cd"])
        if baseline_cfd_result is not None
        and baseline_cfd_result.get("final_cd") is not None
        else None
    )
    valid_candidate_ids = {
        id(candidate) for candidate in _engineering_valid_candidates(high_fidelity_results)
    }
    rejection_by_id = {
        str(row.get("candidate_id")): list(row.get("reasons") or [])
        for row in rejected_candidates
    }
    rows: list[dict] = []

    for index, candidate in enumerate(high_fidelity_results, start=1):
        candidate_id = f"candidate-{index:03d}"
        vector = getattr(candidate, "vector")
        objectives = [float(value) for value in getattr(candidate, "objectives", []) or []]
        final_cd = float(objectives[0]) if objectives and objectives[0] < 1e8 else None
        foam_record = (
            dict(foam_evaluation_records[index - 1])
            if index - 1 < len(foam_evaluation_records)
            else None
        )
        geometry = _candidate_geometry_evidence(
            candidate=candidate,
            baseline_mesh=baseline_mesh,
            region=region,
            deformer=deformer,
            foam_record=foam_record,
        )
        rows.append(
            {
                "schema_version": 1,
                "record_type": "candidate",
                "created_at": created_at,
                "case_id": case_id,
                "case_name": case_name,
                "source_path": str(source_path),
                "hull_fingerprint": hull_fingerprint,
                "settings_hash": settings_hash,
                "candidate_id": candidate_id,
                "backend": backend,
                "parameters": dict(vector.values),
                "objectives": objectives,
                "final_cd": final_cd,
                "baseline_cd": baseline_cd,
                "improvement_percent": _candidate_improvement_percent(
                    baseline_cd=baseline_cd,
                    candidate_cd=final_cd,
                ),
                "engineering_valid": id(candidate) in valid_candidate_ids,
                "engineering_outcome": dict(engineering_outcome),
                "rejection_reasons": rejection_by_id.get(candidate_id, []),
                "geometry": geometry,
                "solver": _solver_evidence(foam_record),
                "force_coeffs": (
                    foam_record.get("force_coeffs")
                    if foam_record is not None
                    else None
                ),
                "artifact_paths": _artifact_paths(foam_record),
            }
        )

    if baseline_cfd_result is not None:
        rows.append(
            {
                "schema_version": 1,
                "record_type": "baseline",
                "created_at": created_at,
                "case_id": case_id,
                "case_name": case_name,
                "source_path": str(source_path),
                "hull_fingerprint": hull_fingerprint,
                "settings_hash": settings_hash,
                "candidate_id": "baseline",
                "backend": str(baseline_cfd_result.get("backend", "unknown")),
                "parameters": {},
                "objectives": [float(baseline_cd)] if baseline_cd is not None else [],
                "final_cd": baseline_cd,
                "baseline_cd": baseline_cd,
                "improvement_percent": 0.0 if baseline_cd is not None else None,
                "engineering_valid": True,
                "engineering_outcome": dict(engineering_outcome),
                "rejection_reasons": [],
                "solver": {"status": "executed_ok"},
                "force_coeffs": dict(baseline_cfd_result),
                "artifact_paths": {},
            }
        )
    return rows


def _candidate_geometry_evidence(
    *,
    candidate,
    baseline_mesh: trimesh.Trimesh,
    region: dict,
    deformer: BulbFFDDeformer,
    foam_record: dict | None,
) -> dict:
    """Build geometry/manufacturability labels for evidence rows."""
    vector = getattr(candidate, "vector", None)
    if vector is None:
        return {
            "stl_report": None,
            "parameter_warnings": [],
            "constraint_violations": ["missing_vector"],
            "geometry_risk": "high",
            "manufacturability_risk": "warning",
        }

    design_space = KrachtDesignSpace()
    parameter_warnings = design_space.manufacturability_warnings(vector)
    constraint_violations = design_space.constraint_violations(vector)
    stl_report = (
        foam_record.get("stl_report")
        if foam_record is not None and foam_record.get("stl_report") is not None
        else None
    )
    if stl_report is None:
        try:
            deformed = deformer.deform(baseline_mesh, region, vector)
            stl_report = validate_stl(deformed)
        except Exception as exc:
            stl_report = {
                "checks_passed": False,
                "geometry_risk": "high",
                "failure_reasons": [f"geometry_evidence_failed:{exc}"],
            }

    geometry_risk = str(stl_report.get("geometry_risk", "high"))
    if parameter_warnings and geometry_risk == "low":
        geometry_risk = "medium"

    return {
        "stl_report": stl_report,
        "parameter_warnings": parameter_warnings,
        "constraint_violations": constraint_violations,
        "geometry_risk": geometry_risk,
        "manufacturability_risk": "warning" if parameter_warnings else "clear",
    }


def _candidate_improvement_percent(
    *,
    baseline_cd: float | None,
    candidate_cd: float | None,
) -> float | None:
    if baseline_cd is None or baseline_cd <= 0 or candidate_cd is None:
        return None
    return float((baseline_cd - candidate_cd) / baseline_cd * 100.0)


def _solver_evidence(foam_record: dict | None) -> dict | None:
    if foam_record is None:
        return None
    run_manifest = foam_record.get("run_manifest") or {}
    check_mesh_reports = [
        step.get("check_mesh_report")
        for step in run_manifest.get("executed_steps", [])
        if step.get("check_mesh_report") is not None
    ]
    solver_reports = [
        step.get("solver_report")
        for step in run_manifest.get("executed_steps", [])
        if step.get("solver_report") is not None
    ]
    return {
        "status": foam_record.get("solver_status"),
        "reason": foam_record.get("solver_reason"),
        "foam_candidate_id": foam_record.get("foam_candidate_id"),
        "high_fidelity_used": run_manifest.get("high_fidelity_used"),
        "check_mesh": check_mesh_reports[-1] if check_mesh_reports else None,
        "solver_report": solver_reports[-1] if solver_reports else None,
    }


def _artifact_paths(foam_record: dict | None) -> dict:
    if foam_record is None:
        return {}
    paths: dict[str, str] = {}
    for source_key, target_key in (
        ("candidate_work_dir", "work_dir"),
        ("input_geometry_path", "input_geometry"),
    ):
        value = foam_record.get(source_key)
        if value is not None:
            paths[target_key] = str(value)
    return paths


# --- validity classifier glue (spec 2026-04-23 §4 L4) -------------------


def _validity_history_path(project_root: Path) -> Path:
    """Return the JSONL file accumulating ``(vector, invalid)`` labels.

    One file per project root so runs share a common training history.
    The directory is created lazily by :func:`_append_validity_record`
    to avoid polluting an empty ``project_root`` with a ``.history`` dir
    on dry-run flows.
    """
    return Path(project_root) / ".history" / VALIDITY_HISTORY_FILENAME


def _load_and_train_validity_classifier(history_path: Path) -> ValidityClassifier:
    """Train a classifier on every row of the JSONL history.

    Missing or empty history → returns a cold classifier whose
    ``predict_invalid_probability`` returns ``None`` (prefilter becomes a
    no-op).
    """
    import json as _json

    classifier = ValidityClassifier()
    if not history_path.exists():
        return classifier

    vectors: List[List[float]] = []
    labels: List[int] = []
    try:
        with history_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                entry = _json.loads(line)
                vector = entry.get("vector")
                invalid = entry.get("invalid")
                if vector is None or invalid is None:
                    continue
                if len(vector) != len(KRACHT_PARAMETER_NAMES):
                    continue
                vectors.append([float(v) for v in vector])
                labels.append(int(invalid))
    except (OSError, ValueError):
        # Corrupt history is not fatal — degrade to cold classifier.
        return classifier

    if len(vectors) < MIN_TRAINING_SAMPLES:
        # Fit anyway so ``n_training_samples`` reflects reality, but the
        # predictor will return None until the threshold is reached.
        classifier.fit(vectors, labels)
        return classifier

    classifier.fit(vectors, labels)
    return classifier


def _append_validity_record(
    history_path: Path,
    vector: KrachtVector,
    invalid: bool,
) -> None:
    import json as _json

    row = {
        "vector": [float(vector.values[name]) for name in KRACHT_PARAMETER_NAMES],
        "invalid": int(bool(invalid)),
    }
    try:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(_json.dumps(row) + "\n")
    except OSError:
        # Do not let persistence errors derail the optimisation run.
        pass


def _is_mesh_invalid(mesh: trimesh.Trimesh) -> bool:
    """Sanity check for a deformed mesh.

    A mesh is "invalid" if it fails any of:
    * not watertight (topology broken),
    * winding not consistent (normals inverted somewhere),
    * volume is not strictly positive (folded / self-intersected).

    Trimesh's winding check doesn't assert there are zero self-
    intersections, but together the three signals are a reasonable proxy.
    """
    try:
        if not bool(mesh.is_watertight):
            return True
        if not bool(mesh.is_winding_consistent):
            return True
    except Exception:
        return True
    try:
        if mesh.is_volume:
            volume = float(mesh.volume)
        else:
            extents = mesh.extents.astype(float)
            volume = float(extents[0] * extents[1] * extents[2])
    except Exception:
        return True
    return volume <= 0.0


def _with_validity_prefilter(
    base_evaluator: Callable[[Sequence[KrachtVector]], List[List[float]]],
    *,
    classifier: ValidityClassifier,
    reject_threshold: float,
    history_path: Path,
    baseline_mesh: trimesh.Trimesh,
    region: dict,
    deformer: BulbFFDDeformer,
    case_logger: CaseLogger,
) -> Callable[[Sequence[KrachtVector]], List[List[float]]]:
    """Wrap the mid-gate with a validity prefilter + label recorder.

    Behaviour per vector:

    * If the classifier predicts ``p(invalid) > reject_threshold``, skip
      the deformer entirely and assign a hard penalty objective vector.
    * Otherwise run the base evaluator and, after deforming for the
      sanity check, append a label row to ``history_path`` for the next
      run's training data.
    """
    # Spec 2026-04-23 §4 L4 calls out a [1e9, 1e9, 1e9] penalty vector,
    # but the NSGA-II problem is built with the mid-gate's natural
    # objective count (3 today with L2 quality objective). We size the
    # penalty row off the first observed base-evaluator output so this
    # prefilter stays compatible whenever the mid-gate grows extra
    # objectives.
    penalty_value: float = 1e9
    observed_n_objectives: list[int] = []

    def evaluate(vectors: Sequence[KrachtVector]) -> List[List[float]]:
        vectors = list(vectors)
        results: List[List[float] | None] = [None] * len(vectors)
        rerun_indices: List[int] = []
        rerun_vectors: List[KrachtVector] = []
        reject_count = 0

        for index, vector in enumerate(vectors):
            prob = classifier.predict_invalid_probability(vector)
            if prob is not None and prob > reject_threshold:
                # Predicted invalid → skip deformer, write label, penalty.
                results[index] = None  # fill below when we know the width
                rerun_indices.append(-1)  # placeholder; not re-run
                _append_validity_record(history_path, vector, invalid=True)
                reject_count += 1
                continue
            rerun_indices.append(index)
            rerun_vectors.append(vector)

        if rerun_vectors:
            rerun_rows = base_evaluator(rerun_vectors)
            if rerun_rows:
                observed_n_objectives.append(len(rerun_rows[0]))
            j = 0
            for target_index in rerun_indices:
                if target_index == -1:
                    continue
                vector = vectors[target_index]
                row = rerun_rows[j]
                j += 1
                # Sanity check via a fresh deform so we record the real label.
                try:
                    deformed = deformer.deform(baseline_mesh, region, vector)
                    invalid_label = _is_mesh_invalid(deformed)
                except Exception:
                    invalid_label = True
                _append_validity_record(history_path, vector, invalid=invalid_label)
                results[target_index] = [float(value) for value in row]

        # Now fill in penalty rows with the correct width.
        n_objectives = observed_n_objectives[0] if observed_n_objectives else 3
        penalty_row = [penalty_value] * n_objectives
        filled: List[List[float]] = []
        for row in results:
            filled.append(row if row is not None else list(penalty_row))

        if reject_count > 0:
            case_logger.log_stage(
                stage="validity_prefilter",
                status="applied",
                extra={
                    "rejected": reject_count,
                    "total": len(vectors),
                    "training_samples": classifier.n_training_samples,
                    "reject_threshold": float(reject_threshold),
                },
            )

        return filled

    return evaluate


def _wrap_mid_gate_with_gp(
    base_evaluator: Callable[[Sequence[KrachtVector]], List[List[float]]],
    surrogate: GPSurrogate,
) -> Callable[[Sequence[KrachtVector]], List[List[float]]]:
    """L1 — replace the drag proxy with a GP mean when the surrogate is
    confident; fall back to the analytic proxy otherwise.

    Strategy:
    * Compute the proxy once per batch so we keep the second objective
      (volume delta / etc.) unchanged.
    * Ask the GP for (mean, std) per vector. If ``predict`` returns
      ``None`` (too few training points) we return the proxy untouched.
    * Otherwise replace objective[0] with the GP mean for every vector.
      We don't yet gate on std because the GP naturally assigns high
      std to extrapolated points and the GA will penalise them via the
      proxy's implicit volume term — keeping the code simple.
    """

    def wrapped(vectors: Sequence[KrachtVector]) -> List[List[float]]:
        proxy_objectives = base_evaluator(vectors)
        prediction = surrogate.predict(vectors)
        if prediction is None:
            return proxy_objectives
        means, _stds = prediction
        merged: List[List[float]] = []
        for row, mean in zip(proxy_objectives, means):
            new_row = list(row)
            if new_row:
                new_row[0] = float(mean)
            merged.append(new_row)
        return merged

    return wrapped


# --- per-generation snapshots (spec 2026-04-22 §10.2) -----------------


def _make_generation_writer(
    *,
    case_dir: Path,
    repository: FilesystemProjectRepository,
    case,
    convergence_rows: list[dict],
    case_logger: CaseLogger,
    generation_offset: int = 0,
):
    """Return a callable that persists one ``gen-NN/`` directory per call.

    The returned closure matches the snapshot signature accepted by
    :class:`NSGA2Strategy`:

        ``(generation_index, population, objectives, pareto) -> None``

    For each generation we write::

        case_dir/working/night_optimization/generations/gen-NN/
            population.json   # vector + objectives per individual
            pareto.json       # current non-dominated set

    plus we accumulate one row in ``convergence_rows`` (the run-level
    ``convergence.csv`` is flushed after the cascade exits, not per
    generation, so a half-written CSV doesn't outlive a crashed run).

    The optional ``generation_offset`` shifts NSGA-II's internal 0-based
    generation index when persisting (spec 2026-04-22 §10.3 — resume).
    Used by ``resume_night_optimization``: the resumed cascade always
    starts at NSGA-II generation 0 internally, but if the killed run
    had already written ``gen-04`` we want the new snapshots on disk
    as ``gen-05`` and onwards. Existing snapshots from the prior run
    are NOT overwritten because the writer never targets indices below
    the offset.

    Failures inside the writer are logged and swallowed: a bad disk
    write must NOT abort the optimisation. The rest of the run keeps
    going and we still get the final outputs.
    """
    generations_root = case_dir / "working" / "night_optimization" / "generations"
    offset = int(generation_offset)

    def write_generation(
        generation_index: int,
        population: List[KrachtVector],
        objectives: List[List[float]],
        pareto: List,
    ) -> None:
        try:
            on_disk_index = int(generation_index) + offset
            gen_dir = generations_root / f"gen-{on_disk_index:02d}"
            gen_dir.mkdir(parents=True, exist_ok=True)

            population_payload = {
                "generation": int(on_disk_index),
                "individuals": [
                    {
                        "vector": [
                            float(vector.values[name])
                            for name in KRACHT_PARAMETER_NAMES
                        ],
                        "parameters": {
                            name: float(vector.values[name])
                            for name in KRACHT_PARAMETER_NAMES
                        },
                        "objectives": [float(value) for value in obj_row],
                    }
                    for vector, obj_row in zip(population, objectives)
                ],
            }
            (gen_dir / "population.json").write_text(
                json.dumps(population_payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )

            pareto_payload = {
                "generation": int(on_disk_index),
                "candidates": [
                    {
                        "vector": [
                            float(candidate.vector.values[name])
                            for name in KRACHT_PARAMETER_NAMES
                        ],
                        "parameters": {
                            name: float(candidate.vector.values[name])
                            for name in KRACHT_PARAMETER_NAMES
                        },
                        "objectives": [
                            float(value) for value in candidate.objectives
                        ],
                    }
                    for candidate in pareto
                ],
            }
            (gen_dir / "pareto.json").write_text(
                json.dumps(pareto_payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )

            convergence_rows.append(
                _convergence_row(
                    generation_index=int(on_disk_index),
                    objectives=objectives,
                )
            )

            # Touch the case so an external "bulbopt list" run sees the
            # case is still active. This also creates a natural hook for
            # tests that want to confirm the case sat in
            # RUNNING_NIGHT_OPTIMIZATION during the run.
            try:
                repository.save_case(case)
            except Exception:
                # save_case is best-effort here; the actual lifecycle
                # transitions remain in run_night_optimization itself.
                pass

        except Exception as exc:
            # Spec brief: do NOT abort the optimisation if the writer
            # fails. Log and continue.
            case_logger.log_stage(
                stage="generation_snapshot",
                status="failed",
                extra={
                    "generation": int(generation_index),
                    "error": str(exc),
                },
            )

    return write_generation


def _convergence_row(
    *,
    generation_index: int,
    objectives: List[List[float]],
) -> dict:
    """Compute the per-generation convergence summary."""
    n = len(objectives)
    n_obj = len(objectives[0]) if objectives else 3
    # Pad missing columns so a 2-objective evaluator (legacy tests) still
    # produces 3-column convergence rows.
    width = max(n_obj, 3)

    bests = [float("inf")] * width
    sums = [0.0] * width
    for row in objectives:
        for index in range(width):
            if index < len(row):
                value = float(row[index])
                if value < bests[index]:
                    bests[index] = value
                sums[index] += value
    means = [sums[index] / n if n else 0.0 for index in range(width)]
    # If a column never received a value (zero-row cohort) ``best`` stays
    # ``inf`` — replace with NaN-equivalent string for the CSV later.
    return {
        "generation": int(generation_index),
        "n_individuals": int(n),
        "best_obj0": bests[0] if bests[0] != float("inf") else float("nan"),
        "best_obj1": bests[1] if bests[1] != float("inf") else float("nan"),
        "best_obj2": bests[2] if bests[2] != float("inf") else float("nan"),
        "mean_obj0": float(means[0]),
    }


def _write_convergence_csv(
    *,
    case_dir: Path,
    rows: Sequence[dict],
    case_logger: CaseLogger,
    append: bool = False,
) -> None:
    """Write ``working/night_optimization/convergence.csv``.

    Always writes a header row so downstream tooling doesn't trip on a
    missing file. Failures (disk full, etc.) are logged and swallowed —
    the rest of the night-run should still finalise.

    When ``append=True`` (used by ``resume_night_optimization``) and the
    target file already exists with a matching header, the new rows are
    appended after the existing ones instead of overwriting the file.
    Mismatched/missing header → behave like a fresh write so a corrupt
    older CSV does not survive a resume.
    """
    night_dir = case_dir / "working" / "night_optimization"
    night_dir.mkdir(parents=True, exist_ok=True)
    path = night_dir / "convergence.csv"
    columns = [
        "generation",
        "n_individuals",
        "best_obj0",
        "best_obj1",
        "best_obj2",
        "mean_obj0",
    ]
    header = ",".join(columns)
    try:
        new_data_lines: list[str] = []
        for row in rows:
            cells: list[str] = []
            for column in columns:
                value = row.get(column)
                if isinstance(value, float):
                    # Match repr of NaN / regular floats; sufficient for
                    # downstream pandas / numpy readers.
                    cells.append(repr(value))
                else:
                    cells.append(str(value))
            new_data_lines.append(",".join(cells))

        if append and path.exists():
            existing = path.read_text(encoding="utf-8")
            existing_lines = [
                line for line in existing.splitlines() if line.strip()
            ]
            if existing_lines and existing_lines[0] == header:
                merged = existing_lines + new_data_lines
                path.write_text("\n".join(merged) + "\n", encoding="utf-8")
                return

        lines = [header, *new_data_lines]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as exc:
        case_logger.log_stage(
            stage="convergence_csv",
            status="failed",
            extra={"error": str(exc)},
        )


# --- persistence helpers ------------------------------------------------


def _persist_top_candidate_meshes(
    *,
    case_dir: Path,
    result: CascadeResult,
    repaired_mesh: trimesh.Trimesh,
    region: dict,
    deformer: BulbFFDDeformer,
    apply_isotropic_polish: bool = False,
) -> tuple[str | None, List[dict]]:
    """Write deformed STL and companion ``stl_valid.json`` for each top
    candidate. Returns ``(winner_id, invalid_candidates)`` where
    ``invalid_candidates`` is a list of ``{candidate_id, report}`` for
    any STL that failed sanity checks so the caller can pass them to the
    HTML template.

    When ``apply_isotropic_polish`` is ``True`` the STL written to disk
    is also passed through
    :func:`bulbopt.optimization.parametric.isotropic_remesh.remesh_region_isotropic`
    — a visual-only polish that re-triangulates the bulb region into
    near-equilateral triangles. The validity report is computed on the
    *polished* mesh so the engineer sees what they're shipping.
    The deformed (un-polished) mesh is what CFD operates on upstream
    of this call, so this polish never affects optimisation or GP
    training history.
    """
    output_root = case_dir / "outputs" / "top_candidates"
    output_root.mkdir(parents=True, exist_ok=True)

    if result.high_fidelity_results:
        ranked = _engineering_valid_candidates(result.high_fidelity_results)
    else:
        ranked = _engineering_valid_candidates([
        type(
            "_CandidateAsResult",
            (),
            {"vector": candidate.vector, "objectives": candidate.objectives},
        )()
        for candidate in result.pareto_front.candidates
    ])
    winner_id: str | None = None
    invalid_candidates: List[dict] = []
    polish_active = apply_isotropic_polish and is_isotropic_remesh_available()
    for index, candidate in enumerate(ranked[:10], start=1):
        candidate_id = f"candidate-{index:03d}"
        if winner_id is None:
            winner_id = candidate_id
        candidate_dir = output_root / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        deformed = deformer.deform(repaired_mesh, region, candidate.vector)
        if polish_active:
            # Visual-only polish: shift the saved STL onto a clean
            # isotropic triangulation in the bulb region. The deformed
            # mesh that CFD evaluated upstream stays unchanged.
            deformed = remesh_region_isotropic(deformed, region)
        (candidate_dir / "geometry.stl").write_bytes(
            trimesh.exchange.stl.export_stl(deformed)
        )
        # L6: write stl_valid.json next to geometry.stl and remember
        # any failing candidates so the report can warn the engineer.
        report = validate_stl(deformed)
        parameter_warnings = KrachtDesignSpace().manufacturability_warnings(
            candidate.vector
        )
        report["parameter_warnings"] = parameter_warnings
        report["manufacturability_risk"] = (
            "warning" if parameter_warnings else "clear"
        )
        if parameter_warnings and report.get("geometry_risk") == "low":
            report["geometry_risk"] = "medium"
        import json as _json

        (candidate_dir / "stl_valid.json").write_text(
            _json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if not report["checks_passed"]:
            invalid_candidates.append(
                {"candidate_id": candidate_id, "report": report}
            )
    return winner_id, invalid_candidates


def _persist_rejected_candidate_meshes(
    *,
    case_dir: Path,
    candidates: Sequence,
    repaired_mesh: trimesh.Trimesh,
    region: dict,
    deformer: BulbFFDDeformer,
) -> list[dict]:
    """Write rejected high-fidelity candidates into a quarantine folder."""
    output_root = case_dir / "outputs" / "rejected_candidates"
    rejected: list[dict] = []
    for index, candidate in enumerate(candidates, start=1):
        reasons = _candidate_rejection_reasons(candidate)
        if not reasons:
            continue

        candidate_id = f"candidate-{index:03d}"
        candidate_dir = output_root / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)

        stl_path: str | None = None
        stl_report: dict | None = None
        vector = getattr(candidate, "vector", None)
        if vector is not None:
            try:
                deformed = deformer.deform(repaired_mesh, region, vector)
                geometry_path = candidate_dir / "geometry.stl"
                geometry_path.write_bytes(trimesh.exchange.stl.export_stl(deformed))
                stl_path = str(geometry_path.relative_to(case_dir))
                stl_report = validate_stl(deformed)
                (candidate_dir / "stl_valid.json").write_text(
                    json.dumps(stl_report, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                for reason in stl_report.get("failure_reasons", []):
                    if reason not in reasons:
                        reasons.append(str(reason))
            except Exception as exc:
                reasons.append(f"geometry_export_failed:{exc}")

        payload = {
            "candidate_id": candidate_id,
            "reasons": reasons,
            "objectives": list(getattr(candidate, "objectives", []) or []),
            "parameters": dict(vector.values) if vector is not None else {},
            "stl_path": stl_path,
            "stl_report": stl_report,
        }
        (candidate_dir / "rejection.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        rejected.append(payload)
    return rejected


def _render_night_report(
    *,
    case_dir: Path,
    command: CreateCaseCommand,
    config: NightOptimizationConfig,
    result,
    scheduler,
    winner_id: str | None,
    high_gate_backend: str,
    stl_invalid_candidates: List[dict] | None = None,
    rejected_candidates: List[dict] | None = None,
    engineering_summary: dict | None = None,
    engineering_outcome: dict | None = None,
    warm_start_summary: dict | None = None,
    repaired_mesh: trimesh.Trimesh | None = None,
    region: dict | None = None,
) -> Path:
    template_root = Path(__file__).resolve().parents[2] / "reporting" / "templates"
    report = HtmlReportAdapter(template_root=template_root)
    reports_dir = case_dir / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    pareto_candidates = []
    for index, c in enumerate(result.pareto_front.candidates, start=1):
        pareto_candidates.append(
            {
                "candidate_id": f"pareto-{index:03d}",
                "parameters": dict(c.vector.values),
                "objectives": list(c.objectives),
            }
        )

    high_fidelity = []
    parameter_warning_candidates = []
    design_space = KrachtDesignSpace()
    for index, r in enumerate(result.high_fidelity_results, start=1):
        candidate_id = f"candidate-{index:03d}"
        stl_path = case_dir / "outputs" / "top_candidates" / candidate_id / "geometry.stl"
        parameter_warnings = design_space.manufacturability_warnings(r.vector)
        if parameter_warnings:
            parameter_warning_candidates.append(
                {
                    "candidate_id": candidate_id,
                    "warnings": parameter_warnings,
                    "parameters": dict(r.vector.values),
                }
            )
        high_fidelity.append(
            {
                "candidate_id": candidate_id,
                "parameters": dict(r.vector.values),
                "objectives": list(r.objectives),
                "stl_path": str(stl_path.relative_to(case_dir)),
                "parameter_warnings": parameter_warnings,
            }
        )

    # Pareto SVG: feeds the new pareto-plot section. We pass the
    # high-fidelity candidates first (so the engineering winner ends up
    # at the front and is highlighted), then the rest of the Pareto
    # front. Penalty entries are filtered inside render_pareto_plot_svg.
    plot_candidates: list[dict] = []
    seen_ids: set[str] = set()
    for hf in high_fidelity:
        plot_candidates.append(
            {
                "candidate_id": hf["candidate_id"],
                "objectives": hf["objectives"],
            }
        )
        seen_ids.add(hf["candidate_id"])
    for pc in pareto_candidates:
        if pc["candidate_id"] in seen_ids:
            continue
        plot_candidates.append(
            {
                "candidate_id": pc["candidate_id"],
                "objectives": pc["objectives"],
            }
        )
    pareto_plot_svg = render_pareto_plot_svg(plot_candidates)

    # Fuel-savings projection: derive the baseline resistance proxy
    # directly from the repaired baseline mesh using the same formula
    # as ``_mid_gate_evaluator`` ((beam * draft) / axial). The winner's
    # proxy is taken from the first high-fidelity result, falling back
    # to the first non-penalty Pareto candidate when no HF run was
    # promoted. When neither is available the fuel-savings section is
    # left out of the report.
    fuel_savings: dict | None = None
    if repaired_mesh is not None and region is not None:
        try:
            extents = repaired_mesh.extents.astype(float)
            primary = int(region.get("axis_index", int(extents.argmax())))
            secondary = [i for i in range(3) if i != primary]
            axial = max(extents[primary], 1e-9)
            beam = max(extents[secondary[0]], 1e-9)
            draft = max(extents[secondary[1]], 1e-9)
            baseline_resistance = float((beam * draft) / axial)
        except Exception:
            baseline_resistance = None  # type: ignore[assignment]
        else:
            winner_resistance: float | None = None
            if result.high_fidelity_results:
                first_hf_obj = list(result.high_fidelity_results[0].objectives)
                if first_hf_obj and first_hf_obj[0] < 1e8:
                    winner_resistance = float(first_hf_obj[0])
            if winner_resistance is None:
                for c in result.pareto_front.candidates:
                    objs = list(c.objectives)
                    if objs and objs[0] < 1e8:
                        winner_resistance = float(objs[0])
                        break
            if winner_resistance is not None:
                fuel_savings = project_fuel_savings(
                    baseline_resistance=baseline_resistance,
                    winner_resistance=winner_resistance,
                )
                # Pre-format thousands-separated USD/year for the
                # template. Jinja's ``|format`` filter is %-style and
                # cannot handle ``{:,.0f}`` directly.
                fuel_savings["fuel_savings_usd_per_year_formatted"] = (
                    f"{fuel_savings['fuel_savings_usd_per_year']:,.0f}"
                )

    status = (
        "completed_with_warnings" if result.budget_exhausted else "completed"
    )
    context = {
        "case_name": command.case_name,
        "status": status,
        "winner_id": winner_id,
        "runtime_budget_hours": config.runtime_budget_hours,
        "population": config.population,
        "generations": config.generations,
        "high_fidelity_budget": config.high_fidelity_budget,
        "pareto_size": len(pareto_candidates),
        "high_fidelity_size": len(high_fidelity),
        "budget_exhausted": result.budget_exhausted,
        "gate_timings": scheduler.gate_timings(),
        "pareto_candidates": pareto_candidates,
        "high_fidelity": high_fidelity,
        "budget_trace": scheduler.trace(),
        "high_gate_backend": high_gate_backend,
        "parameter_names": list(KRACHT_PARAMETER_NAMES),
        "stl_invalid_candidates": list(stl_invalid_candidates or []),
        "parameter_warning_candidates": parameter_warning_candidates,
        "rejected_candidates": list(rejected_candidates or []),
        "engineering_summary": engineering_summary,
        "engineering_outcome": engineering_outcome,
        "warm_start_summary": warm_start_summary,
        "pareto_plot_svg": pareto_plot_svg,
        "fuel_savings": fuel_savings,
    }

    # The HtmlReportAdapter writes report.html via a hard-coded template
    # name; for the night report we render via its raw Jinja environment
    # and write to night_report.html beside it.
    template = report.environment.get_template("night_report.html.j2")
    output_path = reports_dir / "night_report.html"
    output_path.write_text(template.render(**context), encoding="utf-8")
    return output_path


def _mesh_volume(mesh: trimesh.Trimesh) -> float:
    if mesh.is_volume:
        return float(abs(mesh.volume))
    # fall back to AABB volume so non-watertight meshes still produce a
    # finite "volume proxy" without crashing.
    extents = mesh.extents.astype(float)
    return float(abs(extents[0] * extents[1] * extents[2]))
