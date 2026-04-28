from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def build_cli_banner() -> str:
    return "BulbOpt Desktop | STL-first vertical slice"


def default_project_root() -> Path:
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        return Path(local_appdata) / "BulbOpt" / "projects"
    return Path.home() / ".bulbopt" / "projects"


def run_desktop() -> int:
    from PySide6.QtWidgets import QApplication

    from bulbopt.ui.desktop.main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(project_root=default_project_root())
    window.show()
    return app.exec()


def run_cli(argv: list[str]) -> int:
    """Headless entrypoint for overnight/SSH/CI runs (spec §1).

    Subcommands:
      * ``run``           -- execute a new vertical slice case
      * ``list``          -- list persisted cases
      * ``resume``        -- continue a recoverable case from its checkpoints
      * ``night-run``     -- run multi-objective NSGA-II night optimisation
      * ``resume-night``  -- resume a killed night-run from the latest
                             per-generation snapshot (spec 2026-04-22 §10.3)
      * ``evidence``      -- list compatible historical CFD evidence rows
    """

    parser = argparse.ArgumentParser(
        prog="bulbopt",
        description="BulbOpt Desktop headless CLI for STL-first bulbous bow generation.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run a new vertical slice case")
    run_parser.add_argument("--source", required=True, help="Source STL path")
    run_parser.add_argument("--project", required=True, help="Project root directory")
    run_parser.add_argument("--case-name", default="cli-case", help="Case name")
    run_parser.add_argument("--candidate-count", type=int, default=3)
    run_parser.add_argument("--runtime-budget-hours", type=int, default=8)
    run_parser.add_argument(
        "--optimization-mode",
        choices=["generate_new_bulb", "local_optimize"],
        default="generate_new_bulb",
    )

    list_parser = subparsers.add_parser("list", help="List persisted cases")
    list_parser.add_argument("--project", required=True)

    resume_parser = subparsers.add_parser(
        "resume", help="Resume a recoverable case from its checkpoints"
    )
    resume_parser.add_argument("--project", required=True)
    resume_parser.add_argument("--case", required=True)

    night_parser = subparsers.add_parser(
        "night-run",
        help="Run multi-objective NSGA-II night optimization across a Kracht bulb space",
    )
    night_parser.add_argument("--source", required=True, help="Source STL path")
    night_parser.add_argument("--project", required=True, help="Project root directory")
    night_parser.add_argument("--case-name", default="night-run", help="Case name")
    night_parser.add_argument("--budget-hours", type=float, default=8.0)
    night_parser.add_argument("--population", type=int, default=50)
    night_parser.add_argument("--generations", type=int, default=20)
    night_parser.add_argument("--high-fidelity-budget", type=int, default=10)
    night_parser.add_argument("--seed", type=int, default=None)

    resume_night_parser = subparsers.add_parser(
        "resume-night",
        help=(
            "Resume a killed night-run from the latest per-generation snapshot "
            "(spec 2026-04-22 §10.3)"
        ),
    )
    resume_night_parser.add_argument("--project", required=True)
    resume_night_parser.add_argument("--case", required=True)
    resume_night_parser.add_argument("--budget-hours", type=float, default=8.0)
    resume_night_parser.add_argument("--population", type=int, default=50)
    resume_night_parser.add_argument("--generations", type=int, default=20)
    resume_night_parser.add_argument("--high-fidelity-budget", type=int, default=10)
    resume_night_parser.add_argument("--seed", type=int, default=None)

    evidence_parser = subparsers.add_parser(
        "evidence",
        help="List best compatible historical CFD evidence rows",
    )
    evidence_parser.add_argument("--project", required=True)
    evidence_parser.add_argument("--source", required=True, help="Source STL path")
    evidence_parser.add_argument(
        "--backend",
        choices=["surrogate", "external", "simple_foam"],
        default="surrogate",
    )
    evidence_parser.add_argument("--limit", type=int, default=10)
    evidence_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of text",
    )
    evidence_parser.add_argument(
        "--min-eligible",
        type=int,
        default=0,
        help="Return non-zero unless at least this many compatible rows are eligible",
    )

    if not argv:
        parser.print_help(sys.stderr)
        return 2

    namespace = parser.parse_args(argv)

    if namespace.command == "run":
        from bulbopt.application.contracts.models import CreateCaseCommand
        from bulbopt.application.use_cases.run_vertical_slice import run_vertical_slice

        project_root = Path(namespace.project)
        command = CreateCaseCommand(
            case_name=namespace.case_name,
            source_path=str(Path(namespace.source).resolve()),
            vessel_length_m=142.0,
            vessel_beam_m=19.1,
            vessel_draft_m=6.0,
            displacement_t=8420.0,
            speed_knots=[18.0, 20.0],
            candidate_count=int(namespace.candidate_count),
            runtime_budget_hours=int(namespace.runtime_budget_hours),
            optimization_mode=str(namespace.optimization_mode),
        )
        try:
            summary = run_vertical_slice(project_root=project_root, command=command)
        except Exception as error:
            print(f"Run failed: {error}", file=sys.stderr)
            return 1
        print(
            f"{summary.status}: case_id={summary.case_id} "
            f"case_name={summary.case_name} best={summary.best_candidate_id or 'n/a'}"
        )
        return 0

    if namespace.command == "list":
        from bulbopt.storage.project_repository.filesystem_repository import (
            FilesystemProjectRepository,
        )

        repository = FilesystemProjectRepository(root_dir=Path(namespace.project))
        summaries = repository.list_cases()
        if not summaries:
            print("No cases found")
            return 0
        for summary in summaries:
            recoverable = " [recoverable]" if summary.get("is_recoverable") else ""
            print(
                f"{summary.get('case_id', 'n/a')} | {summary.get('case_name', 'n/a')} "
                f"status={summary.get('status', 'n/a')}{recoverable} "
                f"updated={summary.get('updated_at', 'n/a')}"
            )
        return 0

    if namespace.command == "resume":
        from bulbopt.application.use_cases.run_vertical_slice import resume_vertical_slice

        try:
            summary = resume_vertical_slice(
                project_root=Path(namespace.project),
                case_id=namespace.case,
            )
        except Exception as error:
            print(f"Resume failed: {error}", file=sys.stderr)
            return 1
        print(
            f"{summary.status}: case_id={summary.case_id} "
            f"best={summary.best_candidate_id or 'n/a'}"
        )
        return 0

    if namespace.command == "night-run":
        from bulbopt.application.contracts.models import CreateCaseCommand
        from bulbopt.application.use_cases.run_night_optimization import (
            NightOptimizationConfig,
            run_night_optimization,
        )

        project_root = Path(namespace.project)
        command = CreateCaseCommand(
            case_name=namespace.case_name,
            source_path=str(Path(namespace.source).resolve()),
            vessel_length_m=142.0,
            vessel_beam_m=19.1,
            vessel_draft_m=6.0,
            displacement_t=8420.0,
            speed_knots=[18.0, 20.0],
        )
        config = NightOptimizationConfig(
            population=int(namespace.population),
            generations=int(namespace.generations),
            high_fidelity_budget=int(namespace.high_fidelity_budget),
            runtime_budget_hours=float(namespace.budget_hours),
            seed=namespace.seed,
        )
        try:
            summary = run_night_optimization(
                project_root=project_root,
                command=command,
                config=config,
            )
        except Exception as error:
            print(f"Night run failed: {error}", file=sys.stderr)
            return 1
        print(
            f"{summary.status}: case_id={summary.case_id} "
            f"winner={summary.best_candidate_id or 'n/a'}"
        )
        return 0

    if namespace.command == "resume-night":
        # Spec 2026-04-22 §10.3: resume a killed night-run from the
        # latest per-generation snapshot. The CLI mirrors ``night-run``'s
        # population/generations/budget knobs because the stored case
        # only persists the original CreateCaseCommand, not the
        # optimisation config. Pass them again here to control the
        # remaining work.
        from bulbopt.application.use_cases.resume_night_optimization import (
            resume_night_optimization,
        )
        from bulbopt.application.use_cases.run_night_optimization import (
            NightOptimizationConfig,
        )

        project_root = Path(namespace.project)
        config = NightOptimizationConfig(
            population=int(namespace.population),
            generations=int(namespace.generations),
            high_fidelity_budget=int(namespace.high_fidelity_budget),
            runtime_budget_hours=float(namespace.budget_hours),
            seed=namespace.seed,
        )
        try:
            summary = resume_night_optimization(
                project_root=project_root,
                case_id=namespace.case,
                config=config,
            )
        except Exception as error:
            print(f"Night resume failed: {error}", file=sys.stderr)
            return 1
        print(
            f"{summary.status}: case_id={summary.case_id} "
            f"winner={summary.best_candidate_id or 'n/a'}"
        )
        return 0

    if namespace.command == "evidence":
        from bulbopt.application.use_cases.run_night_optimization import (
            _file_sha256,
            _solver_settings_hash,
        )
        from bulbopt.optimization.learning.cfd_evidence_store import CFDEvidenceStore

        project_root = Path(namespace.project)
        source_path = Path(namespace.source)
        evidence_store = CFDEvidenceStore(
            project_root / ".history" / "cfd_evidence.jsonl"
        )
        hull_fingerprint = _file_sha256(source_path)
        settings_hash = _solver_settings_hash(backend=str(namespace.backend))
        summary = evidence_store.warm_start_eligibility_summary(
            hull_fingerprint=hull_fingerprint,
            settings_hash=settings_hash,
        )
        rows = evidence_store.best_candidate_rows(
            int(namespace.limit),
            hull_fingerprint=hull_fingerprint,
            settings_hash=settings_hash,
        )
        payload = {
            "hull_fingerprint": hull_fingerprint,
            "settings_hash": settings_hash,
            "backend": str(namespace.backend),
            "required_eligible": int(namespace.min_eligible),
            "ready_for_warm_start": (
                int(summary["eligible"]) >= int(namespace.min_eligible)
            ),
            "summary": summary,
            "rows": rows,
        }
        if namespace.json:
            print(json.dumps(payload, sort_keys=True))
            return 0 if payload["ready_for_warm_start"] else 1
        print(
            "Evidence eligibility: "
            f"candidate_rows={summary['candidate_rows']} "
            f"eligible={summary['eligible']} "
            f"hull_mismatch={summary['hull_mismatch']} "
            f"settings_mismatch={summary['settings_mismatch']} "
            f"not_engineering_valid={summary['not_engineering_valid']} "
            f"not_improving={summary['not_improving']}"
        )
        if not payload["ready_for_warm_start"]:
            print(
                "Warm-start gate failed: "
                f"eligible={summary['eligible']} "
                f"required={namespace.min_eligible}",
                file=sys.stderr,
            )
            return 1
        if not rows:
            print("No compatible CFD evidence found")
            return 0
        for index, row in enumerate(rows, start=1):
            baseline = row.get("baseline_cd")
            baseline_text = (
                f" baseline={float(baseline):.6f}" if baseline is not None else ""
            )
            improvement = row.get("improvement_percent")
            improvement_text = (
                f" improvement={float(improvement):.2f}%"
                if improvement is not None
                else ""
            )
            print(
                f"{index}. {row.get('case_id', 'n/a')} "
                f"{row.get('candidate_id', 'n/a')} "
                f"Cd={float(row['final_cd']):.6f}"
                f"{baseline_text}{improvement_text}"
            )
        return 0

    parser.print_help(sys.stderr)
    return 2


def dispatch_main(
    run_shell=run_desktop,
    print_banner=print,
    launched_as_module: bool | None = None,
    argv: list[str] | None = None,
) -> int:
    if launched_as_module is None:
        launched_as_module = __spec__ is not None

    # Only route to the CLI when caller explicitly passed argv (i.e. main
    # entry). This preserves the old test contract where ``launched_as_module``
    # alone still means "launch desktop shell".
    if argv:
        return run_cli(argv)

    if launched_as_module:
        return run_shell()

    print_banner(build_cli_banner())
    return 0


if __name__ == "__main__":
    raise SystemExit(dispatch_main(argv=sys.argv[1:] or None))
