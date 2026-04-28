"""``resume_night_optimization`` — recover a killed night-run.

Design reference: 2026-04-22-bulbopt-night-optimization-design.md §10.3.

If a night-run process is killed at gen-XX of YY (e.g. the engineer's
laptop went into sleep mode at gen-18/20, or the WSL VM panicked), today
the only option is to start a fresh case and burn another 8-hour budget.
This use case lets the engineer resume from the latest per-generation
snapshot the killed run wrote.

Strategy:

1. Open the case via :class:`FilesystemProjectRepository.load_case`. If
   the case already terminated (``COMPLETED`` or
   ``COMPLETED_WITH_WARNINGS``), return a no-op summary so the operator
   gets a clear "nothing to do" instead of accidentally running a second
   pass over a finished case.
2. Walk ``case_dir/working/night_optimization/generations/`` and pick
   the highest-numbered ``gen-NN/`` directory. Read its
   ``population.json``.
3. Translate the persisted population back into ``KrachtVector``
   instances using the parameters dict.
4. Compute the remaining generations (``total_generations -
   (last_gen_index + 1)``) and either:
   * call :func:`run_night_optimization` with an internal ``ResumeState``
     (skips ``create_case`` and ``prepare_geometry``, seeds NSGA-II with
     the persisted population, offsets new ``gen-NN`` writes so the
     killed run's snapshots are NEVER overwritten); or
   * if there are no remaining generations, mark the case completed
     immediately.
5. If ``generations/`` is empty/missing, fall through to a normal
   night-run on the existing case (still skipping ``create_case`` and
   ``prepare_geometry`` because the case already exists).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from bulbopt.application.contracts.models import CaseSummary
from bulbopt.application.use_cases.run_night_optimization import (
    HighFidelityEvaluator,
    NightOptimizationConfig,
    ResumeState,
    run_night_optimization,
)
from bulbopt.domain.core.models import CaseStatus
from bulbopt.execution.logging.case_logger import CaseLogger
from bulbopt.optimization.parametric.kracht_space import (
    KRACHT_PARAMETER_NAMES,
    KrachtVector,
)
from bulbopt.storage.project_repository.filesystem_repository import (
    FilesystemProjectRepository,
)


# Terminal statuses that disqualify a case from being resumed.
_TERMINAL_STATUSES: frozenset[CaseStatus] = frozenset(
    {CaseStatus.COMPLETED, CaseStatus.COMPLETED_WITH_WARNINGS}
)


def resume_night_optimization(
    *,
    project_root: Path,
    case_id: str,
    config: NightOptimizationConfig | None = None,
    high_fidelity_evaluator: HighFidelityEvaluator | None = None,
) -> CaseSummary:
    """Resume a previously-killed night run from the latest generation
    snapshot.

    Locates the highest-numbered ``gen-NN/`` directory under
    ``case_dir/working/night_optimization/generations/``, loads its
    ``population.json``, and continues NSGA-II from that population for
    the remaining generations.

    If no snapshot exists, behaves as a brand-new run on the already-
    created case (no second ``create_case`` call, no second
    ``prepare_geometry``).
    """
    config = config or NightOptimizationConfig()
    repository = FilesystemProjectRepository(root_dir=project_root)
    case = repository.load_case(case_id)
    case_dir = repository.case_dir(case_id)
    case_logger = CaseLogger(case_dir / "logs" / "case.log")

    # Guard 1: nothing to resume on a terminated case.
    if case.status in _TERMINAL_STATUSES:
        case_logger.log_stage(
            stage="resume_night_optimization",
            status="skipped",
            extra={
                "case_id": case_id,
                "case_status": case.status.value,
                "reason": "case_already_terminated",
            },
        )
        # Annotate the summary metrics so callers can distinguish a no-op
        # resume from a real one. We do NOT change the case status —
        # leaving the existing terminal value intact preserves the prior
        # winner, gate timings, etc.
        night_metrics = dict(
            case.summary_metrics.get("night_optimization", {}) or {}
        )
        night_metrics["resumed"] = False
        night_metrics["resume_skipped_reason"] = "case_already_terminated"
        case.summary_metrics = {
            **(case.summary_metrics or {}),
            "night_optimization": night_metrics,
        }
        repository.save_case(case)
        return CaseSummary(
            case_id=case.case_id,
            case_name=case.case_name,
            status=case.status.value,
            best_candidate_id=night_metrics.get("winner_id"),
        )

    # Guard 2: load the original CreateCaseCommand so we have the source
    # path / vessel metadata. The repository persisted this on the
    # initial run — it would be a programming error to call resume on a
    # case where the original metadata is missing, but we surface the
    # repository's FileNotFoundError verbatim if so.
    command = repository.load_create_case_command(case_id)

    # Guard 3: locate the latest gen-NN snapshot, if any.
    population_vectors, last_gen_index = _load_latest_generation_population(case_dir)

    if last_gen_index < 0:
        # No prior snapshot found — equivalent to a fresh night-run on
        # the existing case. The pipeline still runs through the resume
        # branch so create_case / prepare_geometry are skipped (the case
        # already has a repaired mesh from the first attempt).
        case_logger.log_stage(
            stage="resume_night_optimization",
            status="no_snapshot",
            extra={"case_id": case_id, "fallback": "fresh_run_on_existing_case"},
        )
        starting_generation = 0
        # Empty population → fresh sampling inside NSGA-II.
        resume_state = ResumeState(
            starting_generation=starting_generation,
            population=[],
            case=case,
            case_dir=case_dir,
        )
        return run_night_optimization(
            project_root=project_root,
            command=command,
            config=config,
            high_fidelity_evaluator=high_fidelity_evaluator,
            _resume_state=resume_state,
        )

    starting_generation = last_gen_index + 1
    remaining_generations = max(int(config.generations) - starting_generation, 0)

    case_logger.log_stage(
        stage="resume_night_optimization",
        status="resuming",
        extra={
            "case_id": case_id,
            "last_gen_index": last_gen_index,
            "starting_generation": starting_generation,
            "remaining_generations": remaining_generations,
            "population_size": len(population_vectors),
        },
    )

    # Guard 4: nothing to do — the killed run already ran every
    # generation. Mark the case completed and return without touching
    # any of the optimisation infrastructure.
    if remaining_generations <= 0:
        case_logger.log_stage(
            stage="resume_night_optimization",
            status="no_remaining_generations",
            extra={
                "starting_generation": starting_generation,
                "configured_generations": int(config.generations),
            },
        )
        case.status = CaseStatus.COMPLETED
        case.is_recoverable = False
        night_metrics = dict(
            case.summary_metrics.get("night_optimization", {}) or {}
        )
        night_metrics["resumed"] = True
        night_metrics["resume_skipped_reason"] = "no_remaining_generations"
        night_metrics["starting_generation"] = starting_generation
        case.summary_metrics = {
            **(case.summary_metrics or {}),
            "night_optimization": night_metrics,
        }
        repository.save_case(case)
        return CaseSummary(
            case_id=case.case_id,
            case_name=case.case_name,
            status=case.status.value,
            best_candidate_id=night_metrics.get("winner_id"),
        )

    # Build a config that covers ONLY the remaining generations. This
    # keeps the budget tracker honest — we don't want to fail because
    # the killed run already consumed (in wall-clock terms) most of the
    # original 8-hour window. The resumed budget covers the resumed
    # work only.
    resume_config = _config_for_remaining_generations(
        config=config,
        remaining_generations=remaining_generations,
    )

    resume_state = ResumeState(
        starting_generation=starting_generation,
        population=population_vectors,
        case=case,
        case_dir=case_dir,
    )

    return run_night_optimization(
        project_root=project_root,
        command=command,
        config=resume_config,
        high_fidelity_evaluator=high_fidelity_evaluator,
        _resume_state=resume_state,
    )


# --- helpers ---------------------------------------------------------------


def _load_latest_generation_population(
    case_dir: Path,
) -> tuple[List[KrachtVector], int]:
    """Walk ``working/night_optimization/generations/`` and return
    ``(population_vectors, last_gen_index)``.

    ``last_gen_index`` is ``-1`` when no readable gen-NN snapshot exists
    (empty/missing tree, or every directory is corrupt).
    """
    generations_root = case_dir / "working" / "night_optimization" / "generations"
    if not generations_root.exists():
        return [], -1

    candidate_dirs: list[tuple[int, Path]] = []
    for child in generations_root.iterdir():
        if not child.is_dir():
            continue
        index = _parse_generation_index(child.name)
        if index is None:
            continue
        candidate_dirs.append((index, child))

    if not candidate_dirs:
        return [], -1

    candidate_dirs.sort(key=lambda item: item[0])
    # Walk from the highest-numbered downward so a partially-written
    # newest directory degrades gracefully to the previous good one.
    for index, gen_dir in reversed(candidate_dirs):
        population_path = gen_dir / "population.json"
        if not population_path.exists():
            continue
        try:
            payload = json.loads(population_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        vectors = _population_payload_to_vectors(payload)
        if not vectors:
            continue
        return vectors, index

    return [], -1


def _parse_generation_index(name: str) -> int | None:
    """Parse ``gen-NN`` directory names into an integer index."""
    if not name.startswith("gen-"):
        return None
    suffix = name[len("gen-"):]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _population_payload_to_vectors(payload: dict) -> List[KrachtVector]:
    """Translate the JSON snapshot ``population.json`` back into a list
    of :class:`KrachtVector`.

    Tolerant of two payload variants:
    * ``{"individuals": [{"parameters": {...}}, ...]}``
    * ``{"individuals": [{"vector": [v0, v1, ...]}, ...]}``

    Rows missing both fields are skipped silently — the resume seed only
    needs *some* well-formed individuals to bootstrap NSGA-II.
    """
    individuals = payload.get("individuals") if isinstance(payload, dict) else None
    if not isinstance(individuals, list):
        return []

    vectors: List[KrachtVector] = []
    for entry in individuals:
        if not isinstance(entry, dict):
            continue
        parameters = entry.get("parameters")
        if isinstance(parameters, dict) and all(
            name in parameters for name in KRACHT_PARAMETER_NAMES
        ):
            try:
                values = {
                    name: float(parameters[name]) for name in KRACHT_PARAMETER_NAMES
                }
            except (TypeError, ValueError):
                continue
            vectors.append(KrachtVector(values=values))
            continue

        raw_vector = entry.get("vector")
        if (
            isinstance(raw_vector, list)
            and len(raw_vector) == len(KRACHT_PARAMETER_NAMES)
        ):
            try:
                values = {
                    name: float(raw_vector[index])
                    for index, name in enumerate(KRACHT_PARAMETER_NAMES)
                }
            except (TypeError, ValueError):
                continue
            vectors.append(KrachtVector(values=values))

    return vectors


def _config_for_remaining_generations(
    *,
    config: NightOptimizationConfig,
    remaining_generations: int,
) -> NightOptimizationConfig:
    """Return a copy of ``config`` whose ``generations`` field reflects
    only the remaining generations (the resumed pass).

    All other knobs are preserved. The runtime budget represents only
    the resumed portion (budget tracking on resume is local to this run
    — see spec 2026-04-22 §10.3).
    """
    return NightOptimizationConfig(
        population=int(config.population),
        generations=int(remaining_generations),
        high_fidelity_budget=int(config.high_fidelity_budget),
        runtime_budget_hours=float(config.runtime_budget_hours),
        seed=config.seed,
        mid_gate_estimated_seconds_per_eval=float(
            config.mid_gate_estimated_seconds_per_eval
        ),
        high_gate_estimated_seconds_per_eval=float(
            config.high_gate_estimated_seconds_per_eval
        ),
        enable_validity_prefilter=bool(config.enable_validity_prefilter),
        validity_reject_threshold=float(config.validity_reject_threshold),
        warm_start_top_k=int(config.warm_start_top_k),
        warm_start_ratio=float(config.warm_start_ratio),
        warm_start_dedup_distance=float(config.warm_start_dedup_distance),
        warm_start_mutation_ratio=float(config.warm_start_mutation_ratio),
        warm_start_mutation_sigma=float(config.warm_start_mutation_sigma),
        random_exploration_ratio=float(config.random_exploration_ratio),
        gp_surrogate_min_history=int(config.gp_surrogate_min_history),
        history_path=config.history_path,
        parallel_workers=int(config.parallel_workers),
    )
