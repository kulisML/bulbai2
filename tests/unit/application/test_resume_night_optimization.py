"""Tests for the ``resume_night_optimization`` use case.

Spec reference: 2026-04-22-bulbopt-night-optimization-design.md §10.3.

If a night-run process is killed at gen-XX of YY, today there is no way to
resume — the engineer would have to restart from scratch and burn another
8 hours. The ``resume_night_optimization`` use case fixes that:

  1. It opens an existing case via the repository.
  2. Walks ``case_dir/working/night_optimization/generations/`` and finds
     the highest-numbered ``gen-NN`` directory.
  3. Translates ``gen-NN/population.json`` back into KrachtVector
     instances, feeds them to NSGA-II as warm-start, and continues for
     the remaining generations.
  4. Writes new ``gen-XX`` directories that pick up where the killed run
     left off (e.g. ``gen-04`` exists → resume writes ``gen-05``,
     ``gen-06``, ...).
  5. Appends to ``convergence.csv`` instead of overwriting.
  6. Marks the case ``COMPLETED`` (or ``COMPLETED_WITH_WARNINGS``).

Tests use a tiny population / generations and the default surrogate
high-fidelity gate so the whole run finishes in milliseconds.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import trimesh

from bulbopt.application.contracts.models import CreateCaseCommand
from bulbopt.application.use_cases.resume_night_optimization import (
    resume_night_optimization,
)
from bulbopt.application.use_cases.run_night_optimization import (
    NightOptimizationConfig,
    run_night_optimization,
)
from bulbopt.domain.core.models import CaseStatus
from bulbopt.storage.project_repository.filesystem_repository import (
    FilesystemProjectRepository,
)


def _write_watertight_stl(path: Path) -> None:
    mesh = trimesh.creation.box(extents=(4.0, 1.5, 1.0))
    path.write_bytes(trimesh.exchange.stl.export_stl(mesh))


def _seed_completed_night_run(
    *,
    project_root: Path,
    source_path: Path,
    case_name: str = "night-resume",
    generations: int = 2,
    seed: int = 99,
    population: int = 4,
) -> str:
    """Drive a real night-run end-to-end so we have a populated case
    folder with the per-generation snapshots we need for resume tests.
    Returns the resulting ``case_id``."""
    summary = run_night_optimization(
        project_root=project_root,
        command=CreateCaseCommand(
            case_name=case_name,
            source_path=str(source_path),
            vessel_length_m=142.0,
            vessel_beam_m=19.1,
            vessel_draft_m=6.0,
            displacement_t=8420.0,
            speed_knots=[18.0, 20.0],
        ),
        config=NightOptimizationConfig(
            population=population,
            generations=generations,
            high_fidelity_budget=1,
            runtime_budget_hours=1.0,
            seed=seed,
            mid_gate_estimated_seconds_per_eval=0.001,
            high_gate_estimated_seconds_per_eval=0.005,
        ),
    )
    return summary.case_id


def _existing_gen_dirs(case_dir: Path) -> list[str]:
    gens_root = case_dir / "working" / "night_optimization" / "generations"
    if not gens_root.exists():
        return []
    return sorted(p.name for p in gens_root.iterdir() if p.is_dir())


def test_resume_returns_no_op_when_case_already_completed(tmp_path: Path) -> None:
    """If the case has terminated, resume must not redo work and must
    flag the result as a no-op."""
    source_path = tmp_path / "hull.stl"
    _write_watertight_stl(source_path)
    project_root = tmp_path / "projects"

    case_id = _seed_completed_night_run(
        project_root=project_root,
        source_path=source_path,
        generations=2,
    )

    case_dir = project_root / case_id
    gens_before = _existing_gen_dirs(case_dir)

    # Sanity: the seeded run terminated as expected.
    repository = FilesystemProjectRepository(root_dir=project_root)
    case = repository.load_case(case_id)
    assert case.status in {CaseStatus.COMPLETED, CaseStatus.COMPLETED_WITH_WARNINGS}

    summary = resume_night_optimization(
        project_root=project_root,
        case_id=case_id,
    )

    # No new gen-NN directories appeared.
    gens_after = _existing_gen_dirs(case_dir)
    assert gens_after == gens_before

    # Returned summary explicitly carries the no-op flag inside the case
    # status (the public dataclass only has 4 fields, so we expose the
    # outcome through the existing ``status`` and the case payload).
    assert summary.case_id == case_id
    case_after = repository.load_case(case_id)
    night_metrics = case_after.summary_metrics.get("night_optimization", {})
    assert night_metrics.get("resumed") is False


def test_resume_starts_from_highest_existing_generation(tmp_path: Path) -> None:
    """If a case has gen-00 and gen-01 written and ``total_generations``
    is 4, resume must append gen-02 and gen-03 without rewriting the
    earlier two."""
    source_path = tmp_path / "hull.stl"
    _write_watertight_stl(source_path)
    project_root = tmp_path / "projects"

    # Seed a 2-generation case (will produce gen-00, gen-01).
    case_id = _seed_completed_night_run(
        project_root=project_root,
        source_path=source_path,
        generations=2,
    )
    case_dir = project_root / case_id

    # The seeded run terminated at COMPLETED, but for the resume test we
    # want to test the "killed mid-run" path. Re-mark the case so resume
    # treats it as recoverable.
    repository = FilesystemProjectRepository(root_dir=project_root)
    case = repository.load_case(case_id)
    case.status = CaseStatus.RUNNING_NIGHT_OPTIMIZATION
    case.is_recoverable = True
    repository.save_case(case)

    pre_gen_dirs = _existing_gen_dirs(case_dir)
    assert pre_gen_dirs == ["gen-00", "gen-01"]

    # Capture content of the existing snapshots so we can assert they're
    # not overwritten.
    gen0_before = (case_dir / "working" / "night_optimization"
                   / "generations" / "gen-00" / "population.json").read_text(
        encoding="utf-8"
    )
    gen1_before = (case_dir / "working" / "night_optimization"
                   / "generations" / "gen-01" / "population.json").read_text(
        encoding="utf-8"
    )

    # Resume with total_generations=4 → expect gen-02 and gen-03.
    summary = resume_night_optimization(
        project_root=project_root,
        case_id=case_id,
        config=NightOptimizationConfig(
            population=4,
            generations=4,
            high_fidelity_budget=1,
            runtime_budget_hours=1.0,
            seed=99,
            mid_gate_estimated_seconds_per_eval=0.001,
            high_gate_estimated_seconds_per_eval=0.005,
        ),
    )

    post_gen_dirs = _existing_gen_dirs(case_dir)
    # The exact tail set: pre + gen-02, gen-03.
    assert "gen-00" in post_gen_dirs
    assert "gen-01" in post_gen_dirs
    assert "gen-02" in post_gen_dirs
    assert "gen-03" in post_gen_dirs
    # No gen-04 since we asked for 4 generations total (0..3).
    assert "gen-04" not in post_gen_dirs

    # Existing snapshots are untouched.
    gen0_after = (case_dir / "working" / "night_optimization"
                  / "generations" / "gen-00" / "population.json").read_text(
        encoding="utf-8"
    )
    gen1_after = (case_dir / "working" / "night_optimization"
                  / "generations" / "gen-01" / "population.json").read_text(
        encoding="utf-8"
    )
    assert gen0_after == gen0_before
    assert gen1_after == gen1_before

    # The new snapshots have the right generation index inside.
    gen2_payload = json.loads(
        (case_dir / "working" / "night_optimization"
         / "generations" / "gen-02" / "population.json").read_text(encoding="utf-8")
    )
    assert gen2_payload["generation"] == 2

    gen3_payload = json.loads(
        (case_dir / "working" / "night_optimization"
         / "generations" / "gen-03" / "population.json").read_text(encoding="utf-8")
    )
    assert gen3_payload["generation"] == 3

    # Final case status reflects a real completion.
    case_after = repository.load_case(case_id)
    assert case_after.status in {
        CaseStatus.COMPLETED,
        CaseStatus.COMPLETED_WITH_WARNINGS,
    }
    night_metrics = case_after.summary_metrics.get("night_optimization", {})
    assert night_metrics.get("resumed") is True
    assert summary.status in {
        CaseStatus.COMPLETED.value,
        CaseStatus.COMPLETED_WITH_WARNINGS.value,
    }


def test_resume_with_no_snapshots_runs_full_optimization(tmp_path: Path) -> None:
    """When ``generations/`` is empty (or missing), resume should behave
    like a normal night-run starting at gen-00."""
    source_path = tmp_path / "hull.stl"
    _write_watertight_stl(source_path)
    project_root = tmp_path / "projects"

    # Seed a case but blow away the generations dir and reset status.
    case_id = _seed_completed_night_run(
        project_root=project_root,
        source_path=source_path,
        generations=2,
    )
    case_dir = project_root / case_id
    gens_dir = case_dir / "working" / "night_optimization" / "generations"
    # Remove every gen-NN subdirectory so the resume sees an empty tree.
    if gens_dir.exists():
        for child in list(gens_dir.iterdir()):
            if child.is_dir():
                for sub in child.iterdir():
                    sub.unlink()
                child.rmdir()

    # Mark the case recoverable.
    repository = FilesystemProjectRepository(root_dir=project_root)
    case = repository.load_case(case_id)
    case.status = CaseStatus.RUNNING_NIGHT_OPTIMIZATION
    case.is_recoverable = True
    repository.save_case(case)

    # Also wipe convergence.csv to simulate a clean slate.
    convergence_path = case_dir / "working" / "night_optimization" / "convergence.csv"
    if convergence_path.exists():
        convergence_path.unlink()

    summary = resume_night_optimization(
        project_root=project_root,
        case_id=case_id,
        config=NightOptimizationConfig(
            population=4,
            generations=2,
            high_fidelity_budget=1,
            runtime_budget_hours=1.0,
            seed=11,
            mid_gate_estimated_seconds_per_eval=0.001,
            high_gate_estimated_seconds_per_eval=0.005,
        ),
    )

    post_gen_dirs = _existing_gen_dirs(case_dir)
    # Both generations were written (gen-00, gen-01).
    assert "gen-00" in post_gen_dirs
    assert "gen-01" in post_gen_dirs

    case_after = repository.load_case(case_id)
    assert case_after.status in {
        CaseStatus.COMPLETED,
        CaseStatus.COMPLETED_WITH_WARNINGS,
    }
    assert summary.case_id == case_id


def test_resume_appends_to_convergence_csv_not_overwrites(tmp_path: Path) -> None:
    """Pre-populate ``convergence.csv`` with a header + 2 historical
    rows. After resume, the CSV should still carry those rows plus one
    or more new rows."""
    source_path = tmp_path / "hull.stl"
    _write_watertight_stl(source_path)
    project_root = tmp_path / "projects"

    # Seed: 2 generations -> we get gen-00 and gen-01 plus convergence
    # with header + 2 rows.
    case_id = _seed_completed_night_run(
        project_root=project_root,
        source_path=source_path,
        generations=2,
    )
    case_dir = project_root / case_id

    # Mark the case as recoverable so resume actually progresses.
    repository = FilesystemProjectRepository(root_dir=project_root)
    case = repository.load_case(case_id)
    case.status = CaseStatus.RUNNING_NIGHT_OPTIMIZATION
    case.is_recoverable = True
    repository.save_case(case)

    convergence_path = (
        case_dir / "working" / "night_optimization" / "convergence.csv"
    )
    pre_lines = [
        line
        for line in convergence_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(pre_lines) == 3  # 1 header + 2 data rows
    pre_header = pre_lines[0]
    pre_data_rows = pre_lines[1:]

    # Resume out to 4 generations (so 2 new rows get appended).
    resume_night_optimization(
        project_root=project_root,
        case_id=case_id,
        config=NightOptimizationConfig(
            population=4,
            generations=4,
            high_fidelity_budget=1,
            runtime_budget_hours=1.0,
            seed=99,
            mid_gate_estimated_seconds_per_eval=0.001,
            high_gate_estimated_seconds_per_eval=0.005,
        ),
    )

    post_lines = [
        line
        for line in convergence_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Header still present.
    assert post_lines[0] == pre_header
    # Original two rows are still in place at the top.
    assert post_lines[1] == pre_data_rows[0]
    assert post_lines[2] == pre_data_rows[1]
    # And new rows are appended after.
    assert len(post_lines) > len(pre_lines)
