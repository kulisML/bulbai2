"""Tests for the ``run_night_optimization`` use case.

Design reference: 2026-04-22-bulbopt-night-optimization-design.md §11.

The use case orchestrates the night-run flow:

1. Create case (reuse existing ``create_case``) with ``source_path`` and
   metadata.
2. Run ``prepare_geometry`` once via the existing stub adapter (repair,
   detect bulb region, write repaired.stl and analysis).
3. Build a Kracht-parametric cascade strategy.
4. Persist the Pareto front + high-fidelity results + generation log into
   the case working directory.
5. Return a ``CaseSummary`` pointing at the winning candidate.

Tests use a tiny population / generations and pass mock gates so the
whole run finishes in well under a second.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import trimesh

from bulbopt.application.contracts.models import CreateCaseCommand
from bulbopt.application.use_cases import run_night_optimization as run_night_module
from bulbopt.application.use_cases.run_night_optimization import (
    NightOptimizationConfig,
    run_night_optimization,
)
from bulbopt.domain.core.models import CaseStatus
from bulbopt.optimization.learning.cfd_evidence_store import CFDEvidenceStore
from bulbopt.optimization.parametric.kracht_space import KrachtVector
from bulbopt.storage.project_repository.filesystem_repository import (
    FilesystemProjectRepository,
)


def _write_watertight_stl(path: Path) -> None:
    mesh = trimesh.creation.box(extents=(4.0, 1.5, 1.0))
    path.write_bytes(trimesh.exchange.stl.export_stl(mesh))


def _compatibility_fields(source_path: Path, *, backend: str = "surrogate") -> dict:
    return {
        "hull_fingerprint": run_night_module._file_sha256(source_path),
        "settings_hash": run_night_module._solver_settings_hash(backend=backend),
    }


def test_run_night_optimization_creates_case_and_writes_pareto_front(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "hull.stl"
    _write_watertight_stl(source_path)

    summary = run_night_optimization(
        project_root=tmp_path / "projects",
        command=CreateCaseCommand(
            case_name="night-demo",
            source_path=str(source_path),
            vessel_length_m=142.0,
            vessel_beam_m=19.1,
            vessel_draft_m=6.0,
            displacement_t=8420.0,
            speed_knots=[18.0, 20.0],
        ),
        config=NightOptimizationConfig(
            population=10,
            generations=2,
            high_fidelity_budget=2,
            runtime_budget_hours=1.0,
            seed=42,
            mid_gate_estimated_seconds_per_eval=0.01,
            high_gate_estimated_seconds_per_eval=0.05,
        ),
    )

    case_dir = tmp_path / "projects" / summary.case_id
    pareto_path = case_dir / "working" / "night_optimization" / "pareto_front.json"
    hf_path = case_dir / "working" / "night_optimization" / "high_fidelity_results.json"
    budget_path = case_dir / "working" / "night_optimization" / "budget_trace.json"

    assert summary.status in {"completed", "completed_with_warnings"}
    assert pareto_path.exists()
    assert hf_path.exists()
    assert budget_path.exists()

    pareto = json.loads(pareto_path.read_text(encoding="utf-8"))
    assert "candidates" in pareto
    assert len(pareto["candidates"]) >= 1
    for candidate in pareto["candidates"]:
        assert "vector" in candidate
        assert "objectives" in candidate
        # Vector has all 8 Kracht parameters.
        assert len(candidate["vector"]) == 8

    hf = json.loads(hf_path.read_text(encoding="utf-8"))
    assert "results" in hf
    assert len(hf["results"]) <= 2

    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    assert "trace" in budget
    assert "gate_timings" in budget
    assert "mid" in budget["gate_timings"]


def test_run_night_optimization_summary_points_at_winner(tmp_path: Path) -> None:
    source_path = tmp_path / "hull.stl"
    _write_watertight_stl(source_path)

    summary = run_night_optimization(
        project_root=tmp_path / "projects",
        command=CreateCaseCommand(
            case_name="night-winner",
            source_path=str(source_path),
            vessel_length_m=142.0,
            vessel_beam_m=19.1,
            vessel_draft_m=6.0,
            displacement_t=8420.0,
            speed_knots=[18.0, 20.0],
        ),
        config=NightOptimizationConfig(
            population=6,
            generations=2,
            high_fidelity_budget=1,
            runtime_budget_hours=1.0,
            seed=7,
            mid_gate_estimated_seconds_per_eval=0.01,
            high_gate_estimated_seconds_per_eval=0.05,
        ),
    )

    assert summary.best_candidate_id is not None
    case_dir = Path(tmp_path / "projects" / summary.case_id)
    winner_dir = case_dir / "outputs" / "top_candidates" / summary.best_candidate_id
    assert winner_dir.exists()
    assert (winner_dir / "geometry.stl").exists()


def test_run_night_optimization_writes_night_report_html(tmp_path: Path) -> None:
    """Stage 5 renders a Jinja template describing the Pareto front, gate
    timings, and high-fidelity winners."""
    source_path = tmp_path / "hull.stl"
    _write_watertight_stl(source_path)

    summary = run_night_optimization(
        project_root=tmp_path / "projects",
        command=CreateCaseCommand(
            case_name="night-report",
            source_path=str(source_path),
            vessel_length_m=142.0,
            vessel_beam_m=19.1,
            vessel_draft_m=6.0,
            displacement_t=8420.0,
            speed_knots=[18.0, 20.0],
        ),
        config=NightOptimizationConfig(
            population=6,
            generations=2,
            high_fidelity_budget=1,
            runtime_budget_hours=1.0,
            seed=13,
            mid_gate_estimated_seconds_per_eval=0.001,
            high_gate_estimated_seconds_per_eval=0.005,
        ),
    )

    case_dir = tmp_path / "projects" / summary.case_id
    report_path = case_dir / "outputs" / "reports" / "night_report.html"
    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "BulbOpt Night Run" in report_text
    assert "Pareto front" in report_text
    assert "Gate timings" in report_text
    # Report must mention at least one Kracht parameter name.
    assert "length_ratio" in report_text


def test_run_night_optimization_uses_simple_foam_gate_when_openfoam_detected(
    tmp_path: Path, monkeypatch
) -> None:
    """When ``detect_openfoam_available`` returns True, the use case wires
    SimpleFoamHighFidelityGate as the high-fidelity evaluator without
    requiring the caller to pass one. Runs real FFD + case building but
    mocks the OpenFOAM runner so tests stay fast."""
    source_path = tmp_path / "hull.stl"
    _write_watertight_stl(source_path)

    # Force detection True at the import site used by the use case, and
    # mock the runner to return executed_ok.
    from bulbopt.application.use_cases import run_night_optimization as use_case_module
    from bulbopt.infrastructure.adapters import openfoam_runner

    monkeypatch.setattr(
        use_case_module, "detect_openfoam_available", lambda: True
    )

    run_calls: list[dict] = []

    def fake_run(self, case_dir, *, case_manifest=None, execute=False, timeout_seconds=600):
        run_calls.append({"case_dir": case_dir, "execute": execute})
        return {
            "status": "executed_ok",
            "is_recoverable": True,
            "high_fidelity_used": True,
            "executed_steps": [],
        }

    monkeypatch.setattr(
        openfoam_runner.OpenFOAMRunnerAdapter,
        "run_case",
        fake_run,
    )

    summary = run_night_optimization(
        project_root=tmp_path / "projects",
        command=CreateCaseCommand(
            case_name="night-foam",
            source_path=str(source_path),
            vessel_length_m=142.0,
            vessel_beam_m=19.1,
            vessel_draft_m=6.0,
            displacement_t=8420.0,
            speed_knots=[18.0, 20.0],
        ),
        config=NightOptimizationConfig(
            population=5,
            generations=2,
            high_fidelity_budget=2,
            runtime_budget_hours=1.0,
            seed=3,
            mid_gate_estimated_seconds_per_eval=0.001,
            high_gate_estimated_seconds_per_eval=0.005,
        ),
    )

    assert summary.status in {"completed", "completed_with_warnings"}
    # At least one high-fidelity run was invoked through the real gate.
    assert len(run_calls) >= 1
    assert all(call["execute"] for call in run_calls)


def test_run_night_optimization_persists_case_json_with_recoverable_flag(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "hull.stl"
    _write_watertight_stl(source_path)

    summary = run_night_optimization(
        project_root=tmp_path / "projects",
        command=CreateCaseCommand(
            case_name="night-resumable",
            source_path=str(source_path),
            vessel_length_m=142.0,
            vessel_beam_m=19.1,
            vessel_draft_m=6.0,
            displacement_t=8420.0,
            speed_knots=[18.0, 20.0],
        ),
        config=NightOptimizationConfig(
            population=5,
            generations=2,
            high_fidelity_budget=1,
            runtime_budget_hours=1.0,
            seed=1,
            mid_gate_estimated_seconds_per_eval=0.001,
            high_gate_estimated_seconds_per_eval=0.005,
        ),
    )

    case_dir = tmp_path / "projects" / summary.case_id
    payload = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    assert payload["status"] in {"completed", "completed_with_warnings"}
    assert payload["is_recoverable"] is False


def test_run_night_optimization_writes_stl_sanity_json(tmp_path: Path) -> None:
    """L6: every top-candidate STL gets a companion stl_valid.json with
    watertight / winding / volume / counts / checks_passed."""
    source_path = tmp_path / "hull.stl"
    _write_watertight_stl(source_path)

    summary = run_night_optimization(
        project_root=tmp_path / "projects",
        command=CreateCaseCommand(
            case_name="night-sanity",
            source_path=str(source_path),
            vessel_length_m=142.0,
            vessel_beam_m=19.1,
            vessel_draft_m=6.0,
            displacement_t=8420.0,
            speed_knots=[18.0, 20.0],
        ),
        config=NightOptimizationConfig(
            population=5,
            generations=2,
            high_fidelity_budget=2,
            runtime_budget_hours=1.0,
            seed=101,
            mid_gate_estimated_seconds_per_eval=0.001,
            high_gate_estimated_seconds_per_eval=0.005,
        ),
    )

    case_dir = tmp_path / "projects" / summary.case_id
    winner_dir = case_dir / "outputs" / "top_candidates" / summary.best_candidate_id
    sanity_path = winner_dir / "stl_valid.json"
    assert sanity_path.exists()
    report = json.loads(sanity_path.read_text(encoding="utf-8"))
    for key in (
        "watertight",
        "winding_consistent",
        "volume",
        "vertex_count",
        "face_count",
        "checks_passed",
    ):
        assert key in report


def test_run_night_optimization_writes_history_jsonl(tmp_path: Path) -> None:
    """L1: every high-fidelity (vector, cd) pair must be persisted to the
    history JSONL so future runs can warm-start from it."""
    source_path = tmp_path / "hull.stl"
    _write_watertight_stl(source_path)
    history_path = tmp_path / "history" / "history.jsonl"

    run_night_optimization(
        project_root=tmp_path / "projects",
        command=CreateCaseCommand(
            case_name="night-history",
            source_path=str(source_path),
            vessel_length_m=142.0,
            vessel_beam_m=19.1,
            vessel_draft_m=6.0,
            displacement_t=8420.0,
            speed_knots=[18.0, 20.0],
        ),
        config=NightOptimizationConfig(
            population=5,
            generations=2,
            high_fidelity_budget=2,
            runtime_budget_hours=1.0,
            seed=42,
            mid_gate_estimated_seconds_per_eval=0.001,
            high_gate_estimated_seconds_per_eval=0.005,
            history_path=history_path,
        ),
    )

    assert history_path.exists()
    rows = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # At least one high-fidelity row was appended.
    assert len(rows) >= 1
    for row in rows:
        assert "parameters" in row and "cd" in row
        assert "length_ratio" in row["parameters"]


def test_run_night_optimization_writes_engineering_outcome(tmp_path: Path) -> None:
    source_path = tmp_path / "hull.stl"
    _write_watertight_stl(source_path)

    summary = run_night_optimization(
        project_root=tmp_path / "projects",
        command=CreateCaseCommand(
            case_name="night-outcome",
            source_path=str(source_path),
            vessel_length_m=142.0,
            vessel_beam_m=19.1,
            vessel_draft_m=6.0,
            displacement_t=8420.0,
            speed_knots=[18.0, 20.0],
        ),
        config=NightOptimizationConfig(
            population=5,
            generations=2,
            high_fidelity_budget=1,
            runtime_budget_hours=1.0,
            seed=17,
            mid_gate_estimated_seconds_per_eval=0.001,
            high_gate_estimated_seconds_per_eval=0.005,
        ),
    )

    case_dir = tmp_path / "projects" / summary.case_id
    hf = json.loads(
        (
            case_dir
            / "working"
            / "night_optimization"
            / "high_fidelity_results.json"
        ).read_text(encoding="utf-8")
    )

    assert hf["engineering_outcome"]["status"] in {
        "engineering_winner",
        "unverified_winner",
        "no_engineering_winner",
    }
    assert "message" in hf["engineering_outcome"]


def test_run_night_optimization_writes_cfd_evidence_jsonl(tmp_path: Path) -> None:
    """Every high-fidelity evaluation must become durable evidence for
    audit, warm-start filtering, and future surrogate training."""
    source_path = tmp_path / "hull.stl"
    _write_watertight_stl(source_path)
    project_root = tmp_path / "projects"

    def external_high(vectors: list[KrachtVector]) -> list[list[float]]:
        return [[0.321, 0.012] for _ in vectors]

    summary = run_night_optimization(
        project_root=project_root,
        command=CreateCaseCommand(
            case_name="night-evidence",
            source_path=str(source_path),
            vessel_length_m=142.0,
            vessel_beam_m=19.1,
            vessel_draft_m=6.0,
            displacement_t=8420.0,
            speed_knots=[18.0, 20.0],
        ),
        config=NightOptimizationConfig(
            population=5,
            generations=2,
            high_fidelity_budget=1,
            runtime_budget_hours=1.0,
            seed=29,
            mid_gate_estimated_seconds_per_eval=0.001,
            high_gate_estimated_seconds_per_eval=0.005,
        ),
        high_fidelity_evaluator=external_high,
    )

    case_dir = project_root / summary.case_id
    case_evidence_path = (
        case_dir / "working" / "night_optimization" / "cfd_evidence.jsonl"
    )
    project_evidence_path = project_root / ".history" / "cfd_evidence.jsonl"

    assert case_evidence_path.exists()
    assert project_evidence_path.exists()
    case_rows = [
        json.loads(line)
        for line in case_evidence_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    project_rows = [
        json.loads(line)
        for line in project_evidence_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert case_rows == project_rows
    assert len(case_rows) == 1
    row = case_rows[0]
    assert row["schema_version"] == 1
    assert row["record_type"] == "candidate"
    assert row["case_id"] == summary.case_id
    assert row["candidate_id"] == "candidate-001"
    assert row["backend"] == "external"
    assert row["final_cd"] == pytest.approx(0.321)
    assert row["baseline_cd"] is None
    assert row["engineering_valid"] is True
    assert row["engineering_outcome"]["status"] == "unverified_winner"
    assert set(row["parameters"]) >= {"length_ratio", "nose_sharpness"}
    assert row["objectives"] == [0.321, 0.012]
    assert row["geometry"]["stl_report"]["checks_passed"] is True
    assert isinstance(row["geometry"]["parameter_warnings"], list)
    assert row["geometry"]["geometry_risk"] in {"low", "medium", "high"}
    assert row["geometry"]["manufacturability_risk"] in {"clear", "warning"}

    hf_payload = json.loads(
        (
            case_dir
            / "working"
            / "night_optimization"
            / "high_fidelity_results.json"
        ).read_text(encoding="utf-8")
    )
    hf_row = hf_payload["results"][0]
    assert hf_row["geometry"]["stl_report"]["checks_passed"] is True
    assert hf_row["geometry"]["manufacturability_risk"] in {"clear", "warning"}


def test_run_night_optimization_quarantines_rejected_high_fidelity_candidates(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "hull.stl"
    _write_watertight_stl(source_path)

    def penalty_high(vectors: list[KrachtVector]) -> list[list[float]]:
        return [[1e9, 1e9] for _ in vectors]

    summary = run_night_optimization(
        project_root=tmp_path / "projects",
        command=CreateCaseCommand(
            case_name="night-rejected",
            source_path=str(source_path),
            vessel_length_m=142.0,
            vessel_beam_m=19.1,
            vessel_draft_m=6.0,
            displacement_t=8420.0,
            speed_knots=[18.0, 20.0],
        ),
        config=NightOptimizationConfig(
            population=5,
            generations=2,
            high_fidelity_budget=1,
            runtime_budget_hours=1.0,
            seed=23,
            mid_gate_estimated_seconds_per_eval=0.001,
            high_gate_estimated_seconds_per_eval=0.005,
        ),
        high_fidelity_evaluator=penalty_high,
    )

    case_dir = tmp_path / "projects" / summary.case_id
    rejected_dir = case_dir / "outputs" / "rejected_candidates" / "candidate-001"
    rejection_path = rejected_dir / "rejection.json"
    hf = json.loads(
        (
            case_dir
            / "working"
            / "night_optimization"
            / "high_fidelity_results.json"
        ).read_text(encoding="utf-8")
    )

    assert summary.best_candidate_id is None
    assert rejection_path.exists()
    rejection = json.loads(rejection_path.read_text(encoding="utf-8"))
    assert "penalty_objective" in rejection["reasons"]
    assert hf["rejected_candidates"][0]["candidate_id"] == "candidate-001"
    assert "penalty_objective" in hf["rejected_candidates"][0]["reasons"]
    report_text = (
        case_dir / "outputs" / "reports" / "night_report.html"
    ).read_text(encoding="utf-8")
    assert "Rejected high-fidelity candidates" in report_text
    assert "penalty_objective" in report_text


def test_run_night_optimization_warm_starts_from_history(tmp_path: Path) -> None:
    """L1: a second run with a populated history places the top historical
    vectors into the initial NSGA-II population."""
    from bulbopt.application.use_cases import run_night_optimization as use_case_module
    from bulbopt.optimization.strategies import nsga2_strategy as strategy_module

    source_path = tmp_path / "hull.stl"
    _write_watertight_stl(source_path)
    history_path = tmp_path / "history.jsonl"

    # Run 1 — populate the history.
    run_night_optimization(
        project_root=tmp_path / "projects1",
        command=CreateCaseCommand(
            case_name="night-run-one",
            source_path=str(source_path),
            vessel_length_m=142.0,
            vessel_beam_m=19.1,
            vessel_draft_m=6.0,
            displacement_t=8420.0,
            speed_knots=[18.0, 20.0],
        ),
        config=NightOptimizationConfig(
            population=5,
            generations=2,
            high_fidelity_budget=2,
            runtime_budget_hours=1.0,
            seed=1,
            mid_gate_estimated_seconds_per_eval=0.001,
            high_gate_estimated_seconds_per_eval=0.005,
            history_path=history_path,
        ),
    )
    # Sanity: history populated before run 2.
    rows_after_run1 = history_path.read_text(encoding="utf-8").splitlines()
    assert len(rows_after_run1) >= 1

    # Run 2 — capture the initial population by monkeypatching
    # NSGA2Strategy.optimize to record vectors passed by the sampling.
    captured: list[list[float]] = []
    original_build = strategy_module._build_initial_sampling

    def spy_build(**kwargs):
        array = original_build(**kwargs)
        for row in array:
            captured.append([float(x) for x in row])
        return array

    import unittest.mock as _mock

    with _mock.patch.object(
        strategy_module, "_build_initial_sampling", side_effect=spy_build
    ):
        run_night_optimization(
            project_root=tmp_path / "projects2",
            command=CreateCaseCommand(
                case_name="night-run-two",
                source_path=str(source_path),
                vessel_length_m=142.0,
                vessel_beam_m=19.1,
                vessel_draft_m=6.0,
                displacement_t=8420.0,
                speed_knots=[18.0, 20.0],
            ),
            config=NightOptimizationConfig(
                population=5,
                generations=2,
                high_fidelity_budget=2,
                runtime_budget_hours=1.0,
                seed=2,
                mid_gate_estimated_seconds_per_eval=0.001,
                high_gate_estimated_seconds_per_eval=0.005,
                history_path=history_path,
            ),
        )

    # Warm-start path was taken — the spy was called at least once.
    assert len(captured) > 0


def test_run_night_optimization_prefers_engineering_evidence_for_warm_start(
    tmp_path: Path,
) -> None:
    """Safe hot-start uses verified project CFD evidence before legacy history."""
    from bulbopt.optimization.strategies import nsga2_strategy as strategy_module

    source_path = tmp_path / "hull.stl"
    _write_watertight_stl(source_path)
    project_root = tmp_path / "projects"
    evidence_vector = {
        "length_ratio": 0.031,
        "breadth_ratio": 0.085,
        "height_ratio": 0.30,
        "axis_z_ratio": 0.20,
        "longitudinal_pos": 0.60,
        "cross_section_c": 0.74,
        "volume_coef": 0.64,
        "nose_sharpness": 0.50,
    }
    CFDEvidenceStore(project_root / ".history" / "cfd_evidence.jsonl").append_many(
        [
            {
                "schema_version": 1,
                "record_type": "candidate",
                "candidate_id": "candidate-safe",
                "parameters": evidence_vector,
                "final_cd": 0.32,
                "baseline_cd": 0.40,
                "improvement_percent": 20.0,
                "engineering_valid": True,
                **_compatibility_fields(source_path),
            },
            {
                "schema_version": 1,
                "record_type": "candidate",
                "candidate_id": "candidate-worse",
                "parameters": dict(evidence_vector, length_ratio=0.033),
                "final_cd": 0.31,
                "baseline_cd": 0.30,
                "improvement_percent": -3.333,
                "engineering_valid": True,
                **_compatibility_fields(source_path),
            },
            {
                "schema_version": 1,
                "record_type": "candidate",
                "candidate_id": "candidate-bad-geometry",
                "parameters": dict(evidence_vector, length_ratio=0.034),
                "final_cd": 0.29,
                "baseline_cd": 0.40,
                "improvement_percent": 27.5,
                "engineering_valid": True,
                "geometry": {
                    "stl_report": {"checks_passed": False},
                    "parameter_warnings": [],
                    "constraint_violations": [],
                    "geometry_risk": "high",
                    "manufacturability_risk": "clear",
                },
                **_compatibility_fields(source_path),
            },
            {
                "schema_version": 1,
                "record_type": "candidate",
                "candidate_id": "candidate-near-bound",
                "parameters": dict(evidence_vector, length_ratio=0.035),
                "final_cd": 0.28,
                "baseline_cd": 0.40,
                "improvement_percent": 30.0,
                "engineering_valid": True,
                "geometry": {
                    "stl_report": {"checks_passed": True},
                    "parameter_warnings": ["sharp_full_section_near_limit"],
                    "constraint_violations": [],
                    "geometry_risk": "medium",
                    "manufacturability_risk": "warning",
                },
                **_compatibility_fields(source_path),
            },
        ]
    )

    captured: list[list[float]] = []
    original_build = strategy_module._build_initial_sampling

    def spy_build(**kwargs):
        array = original_build(**kwargs)
        for row in array:
            captured.append([float(x) for x in row])
        return array

    import unittest.mock as _mock

    with _mock.patch.object(
        strategy_module, "_build_initial_sampling", side_effect=spy_build
    ):
        summary = run_night_optimization(
            project_root=project_root,
            command=CreateCaseCommand(
                case_name="night-evidence-warm-start",
                source_path=str(source_path),
                vessel_length_m=142.0,
                vessel_beam_m=19.1,
                vessel_draft_m=6.0,
                displacement_t=8420.0,
                speed_knots=[18.0, 20.0],
            ),
            config=NightOptimizationConfig(
                population=5,
                generations=2,
                high_fidelity_budget=1,
                runtime_budget_hours=1.0,
                seed=41,
                mid_gate_estimated_seconds_per_eval=0.001,
                high_gate_estimated_seconds_per_eval=0.005,
            ),
        )

    parameter_order = (
        "length_ratio",
        "breadth_ratio",
        "height_ratio",
        "axis_z_ratio",
        "longitudinal_pos",
        "cross_section_c",
        "volume_coef",
        "nose_sharpness",
    )
    expected = [float(evidence_vector[name]) for name in parameter_order]
    worse = list(expected)
    worse[0] = 0.033
    bad_geometry = list(expected)
    bad_geometry[0] = 0.034
    near_bound = list(expected)
    near_bound[0] = 0.035

    assert any(
        all(abs(a - b) < 1e-9 for a, b in zip(row, expected))
        for row in captured
    )
    assert not any(
        all(abs(a - b) < 1e-9 for a, b in zip(row, worse))
        for row in captured
    )
    assert not any(
        all(abs(a - b) < 1e-9 for a, b in zip(row, bad_geometry))
        for row in captured
    )
    assert not any(
        all(abs(a - b) < 1e-9 for a, b in zip(row, near_bound))
        for row in captured
    )

    case_dir = project_root / summary.case_id
    case_payload = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    eligibility = case_payload["summary_metrics"]["night_optimization"]["warm_start"][
        "evidence_eligibility"
    ]
    assert eligibility["geometry_high_risk"] == 1
    assert eligibility["manufacturability_warning"] == 1
    report_text = (case_dir / "outputs" / "reports" / "night_report.html").read_text(
        encoding="utf-8"
    )
    assert "Geometry high-risk evidence rejected" in report_text
    assert "Manufacturability warning evidence rejected" in report_text


def test_run_night_optimization_limits_and_deduplicates_warm_start(
    tmp_path: Path,
) -> None:
    """Warm-start must preserve exploration by capping seeded rows and
    dropping near-duplicate historical elites."""
    from bulbopt.optimization.strategies import nsga2_strategy as strategy_module

    source_path = tmp_path / "hull.stl"
    _write_watertight_stl(source_path)
    project_root = tmp_path / "projects"
    base_vector = {
        "length_ratio": 0.031,
        "breadth_ratio": 0.085,
        "height_ratio": 0.30,
        "axis_z_ratio": 0.20,
        "longitudinal_pos": 0.60,
        "cross_section_c": 0.74,
        "volume_coef": 0.64,
        "nose_sharpness": 0.50,
    }
    duplicate_vector = dict(base_vector, length_ratio=0.0311)
    diverse_vector = dict(base_vector, length_ratio=0.043, breadth_ratio=0.180)
    extra_vector = dict(base_vector, length_ratio=0.020, longitudinal_pos=0.20)

    CFDEvidenceStore(project_root / ".history" / "cfd_evidence.jsonl").append_many(
        [
            {
                "schema_version": 1,
                "record_type": "candidate",
                "candidate_id": "best",
                "parameters": base_vector,
                "final_cd": 0.30,
                "baseline_cd": 0.40,
                "improvement_percent": 25.0,
                "engineering_valid": True,
                **_compatibility_fields(source_path),
            },
            {
                "schema_version": 1,
                "record_type": "candidate",
                "candidate_id": "near-duplicate",
                "parameters": duplicate_vector,
                "final_cd": 0.31,
                "baseline_cd": 0.40,
                "improvement_percent": 22.5,
                "engineering_valid": True,
                **_compatibility_fields(source_path),
            },
            {
                "schema_version": 1,
                "record_type": "candidate",
                "candidate_id": "diverse",
                "parameters": diverse_vector,
                "final_cd": 0.32,
                "baseline_cd": 0.40,
                "improvement_percent": 20.0,
                "engineering_valid": True,
                **_compatibility_fields(source_path),
            },
            {
                "schema_version": 1,
                "record_type": "candidate",
                "candidate_id": "extra",
                "parameters": extra_vector,
                "final_cd": 0.33,
                "baseline_cd": 0.40,
                "improvement_percent": 17.5,
                "engineering_valid": True,
                **_compatibility_fields(source_path),
            },
        ]
    )

    captured: list[list[float]] = []
    original_build = strategy_module._build_initial_sampling

    def spy_build(**kwargs):
        array = original_build(**kwargs)
        for row in array:
            captured.append([float(x) for x in row])
        return array

    import unittest.mock as _mock

    with _mock.patch.object(
        strategy_module, "_build_initial_sampling", side_effect=spy_build
    ):
        summary = run_night_optimization(
            project_root=project_root,
            command=CreateCaseCommand(
                case_name="night-dedup-warm-start",
                source_path=str(source_path),
                vessel_length_m=142.0,
                vessel_beam_m=19.1,
                vessel_draft_m=6.0,
                displacement_t=8420.0,
                speed_knots=[18.0, 20.0],
            ),
            config=NightOptimizationConfig(
                population=10,
                generations=2,
                high_fidelity_budget=1,
                runtime_budget_hours=1.0,
                seed=44,
                warm_start_ratio=0.2,
                warm_start_dedup_distance=0.02,
                warm_start_mutation_ratio=0.0,
                mid_gate_estimated_seconds_per_eval=0.001,
                high_gate_estimated_seconds_per_eval=0.005,
            ),
        )

    order = (
        "length_ratio",
        "breadth_ratio",
        "height_ratio",
        "axis_z_ratio",
        "longitudinal_pos",
        "cross_section_c",
        "volume_coef",
        "nose_sharpness",
    )
    expected_best = [float(base_vector[name]) for name in order]
    expected_duplicate = [float(duplicate_vector[name]) for name in order]
    expected_diverse = [float(diverse_vector[name]) for name in order]
    expected_extra = [float(extra_vector[name]) for name in order]

    assert captured[:2] == [expected_best, expected_diverse]
    assert expected_duplicate not in captured
    assert expected_extra not in captured

    case_log = (
        project_root / summary.case_id / "logs" / "case.log"
    ).read_text(encoding="utf-8")
    assert '"stage": "night_optimization_warm_start"' in case_log
    assert '"selected": 2' in case_log
    assert '"deduplicated": 1' in case_log
    assert '"random": 8' in case_log


def test_run_night_optimization_mutates_elites_and_reports_warm_start(
    tmp_path: Path,
) -> None:
    """Generation zero should contain elites, bounded mutations, and
    forced random exploration, with the composition visible in the report."""
    from bulbopt.optimization.strategies import nsga2_strategy as strategy_module

    source_path = tmp_path / "hull.stl"
    _write_watertight_stl(source_path)
    project_root = tmp_path / "projects"
    elite = {
        "length_ratio": 0.031,
        "breadth_ratio": 0.085,
        "height_ratio": 0.30,
        "axis_z_ratio": 0.20,
        "longitudinal_pos": 0.60,
        "cross_section_c": 0.74,
        "volume_coef": 0.64,
        "nose_sharpness": 0.50,
    }
    CFDEvidenceStore(project_root / ".history" / "cfd_evidence.jsonl").append_many(
        [
            {
                "schema_version": 1,
                "record_type": "candidate",
                "candidate_id": "elite",
                "parameters": elite,
                "final_cd": 0.30,
                "baseline_cd": 0.40,
                "improvement_percent": 25.0,
                "engineering_valid": True,
                **_compatibility_fields(source_path),
            }
        ]
    )

    captured: list[list[float]] = []
    original_build = strategy_module._build_initial_sampling

    def spy_build(**kwargs):
        array = original_build(**kwargs)
        for row in array:
            captured.append([float(x) for x in row])
        return array

    import unittest.mock as _mock

    with _mock.patch.object(
        strategy_module, "_build_initial_sampling", side_effect=spy_build
    ):
        summary = run_night_optimization(
            project_root=project_root,
            command=CreateCaseCommand(
                case_name="night-mutated-warm-start",
                source_path=str(source_path),
                vessel_length_m=142.0,
                vessel_beam_m=19.1,
                vessel_draft_m=6.0,
                displacement_t=8420.0,
                speed_knots=[18.0, 20.0],
            ),
            config=NightOptimizationConfig(
                population=10,
                generations=2,
                high_fidelity_budget=1,
                runtime_budget_hours=1.0,
                seed=45,
                warm_start_ratio=0.2,
                warm_start_mutation_ratio=0.4,
                random_exploration_ratio=0.6,
                mid_gate_estimated_seconds_per_eval=0.001,
                high_gate_estimated_seconds_per_eval=0.005,
            ),
        )

    order = (
        "length_ratio",
        "breadth_ratio",
        "height_ratio",
        "axis_z_ratio",
        "longitudinal_pos",
        "cross_section_c",
        "volume_coef",
        "nose_sharpness",
    )
    elite_row = [float(elite[name]) for name in order]
    seeded_rows = captured[:4]

    assert seeded_rows[0] == elite_row
    assert len(seeded_rows) == 4
    assert all(row != elite_row for row in seeded_rows[1:])
    assert len({tuple(row) for row in seeded_rows}) == 4

    case_dir = project_root / summary.case_id
    case_payload = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    warm_start = case_payload["summary_metrics"]["night_optimization"]["warm_start"]
    assert warm_start["seeded"] == 1
    assert warm_start["mutated"] == 3
    assert warm_start["random"] == 6
    assert warm_start["random_exploration_ratio"] == 0.6

    report_text = (
        case_dir / "outputs" / "reports" / "night_report.html"
    ).read_text(encoding="utf-8")
    assert "Warm-start composition" in report_text
    assert "Seeded elites" in report_text
    assert "Mutated elites" in report_text
    assert "Random exploration" in report_text


def test_run_night_optimization_trains_surrogate_from_cfd_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ML acceleration should use durable CFD evidence, not only legacy history."""
    from bulbopt.application.use_cases import run_night_optimization as use_case_module

    source_path = tmp_path / "hull.stl"
    _write_watertight_stl(source_path)
    project_root = tmp_path / "projects"
    evidence_path = project_root / ".history" / "cfd_evidence.jsonl"

    rows: list[dict] = []
    for index in range(20):
        rows.append(
            {
                "schema_version": 1,
                "record_type": "candidate",
                "candidate_id": f"candidate-{index:03d}",
                "parameters": {
                    "length_ratio": 0.020 + index * 0.0005,
                    "breadth_ratio": 0.080,
                    "height_ratio": 0.250,
                    "axis_z_ratio": 0.200,
                    "longitudinal_pos": 0.550,
                    "cross_section_c": 0.700,
                    "volume_coef": 0.600,
                    "nose_sharpness": 0.500,
                },
                "final_cd": 0.42 - index * 0.001,
                "baseline_cd": 0.50,
                "improvement_percent": 16.0 + index * 0.1,
                "engineering_valid": True,
                **_compatibility_fields(source_path, backend="external"),
            }
        )
    rows.append(
        {
            "schema_version": 1,
            "record_type": "candidate",
            "candidate_id": "candidate-invalid",
            "parameters": dict(rows[0]["parameters"]),
            "final_cd": 1e9,
            "engineering_valid": False,
            **_compatibility_fields(source_path, backend="external"),
        }
    )
    CFDEvidenceStore(evidence_path).append_many(rows)

    trained: dict[str, int] = {}

    class FakeSurrogate:
        def fit(self, vectors, cds):
            trained["vectors"] = len(vectors)
            trained["cds"] = len(cds)

        def predict(self, vectors):
            vectors = list(vectors)
            return [0.333 for _ in vectors], [0.01 for _ in vectors]

    monkeypatch.setattr(use_case_module, "GPSurrogate", FakeSurrogate)

    def external_high(vectors: list[KrachtVector]) -> list[list[float]]:
        return [[0.40, 0.01] for _ in vectors]

    run_night_optimization(
        project_root=project_root,
        command=CreateCaseCommand(
            case_name="night-evidence-surrogate",
            source_path=str(source_path),
            vessel_length_m=142.0,
            vessel_beam_m=19.1,
            vessel_draft_m=6.0,
            displacement_t=8420.0,
            speed_knots=[18.0, 20.0],
        ),
        config=NightOptimizationConfig(
            population=5,
            generations=2,
            high_fidelity_budget=1,
            runtime_budget_hours=1.0,
            seed=43,
            gp_surrogate_min_history=20,
            mid_gate_estimated_seconds_per_eval=0.001,
            high_gate_estimated_seconds_per_eval=0.005,
        ),
        high_fidelity_evaluator=external_high,
    )

    assert trained == {"vectors": 20, "cds": 20}


def test_run_night_optimization_uses_only_compatible_cfd_evidence(
    tmp_path: Path,
) -> None:
    """Evidence from another hull must be logged and ignored for reuse."""
    from bulbopt.application.use_cases import run_night_optimization as use_case_module
    from bulbopt.optimization.strategies import nsga2_strategy as strategy_module

    source_path = tmp_path / "hull.stl"
    _write_watertight_stl(source_path)
    project_root = tmp_path / "projects"
    hull_fingerprint = use_case_module._file_sha256(source_path)
    settings_hash = use_case_module._solver_settings_hash(backend="external")
    compatible = {
        "length_ratio": 0.031,
        "breadth_ratio": 0.085,
        "height_ratio": 0.30,
        "axis_z_ratio": 0.20,
        "longitudinal_pos": 0.60,
        "cross_section_c": 0.74,
        "volume_coef": 0.64,
        "nose_sharpness": 0.50,
    }
    incompatible = dict(compatible, length_ratio=0.043)
    CFDEvidenceStore(project_root / ".history" / "cfd_evidence.jsonl").append_many(
        [
            {
                "schema_version": 1,
                "record_type": "candidate",
                "candidate_id": "compatible",
                "parameters": compatible,
                "final_cd": 0.32,
                "baseline_cd": 0.40,
                "improvement_percent": 20.0,
                "engineering_valid": True,
                "hull_fingerprint": hull_fingerprint,
                "settings_hash": settings_hash,
            },
            {
                "schema_version": 1,
                "record_type": "candidate",
                "candidate_id": "wrong-hull",
                "parameters": incompatible,
                "final_cd": 0.30,
                "baseline_cd": 0.40,
                "improvement_percent": 25.0,
                "engineering_valid": True,
                "hull_fingerprint": "other-hull",
                "settings_hash": settings_hash,
            },
        ]
    )

    captured: list[list[float]] = []
    original_build = strategy_module._build_initial_sampling

    def spy_build(**kwargs):
        array = original_build(**kwargs)
        for row in array:
            captured.append([float(x) for x in row])
        return array

    def external_high(vectors: list[KrachtVector]) -> list[list[float]]:
        return [[0.33, 0.01] for _ in vectors]

    import unittest.mock as _mock

    with _mock.patch.object(
        strategy_module, "_build_initial_sampling", side_effect=spy_build
    ):
        summary = run_night_optimization(
            project_root=project_root,
            command=CreateCaseCommand(
                case_name="night-compatible-evidence",
                source_path=str(source_path),
                vessel_length_m=142.0,
                vessel_beam_m=19.1,
                vessel_draft_m=6.0,
                displacement_t=8420.0,
                speed_knots=[18.0, 20.0],
            ),
            config=NightOptimizationConfig(
                population=5,
                generations=2,
                high_fidelity_budget=1,
                runtime_budget_hours=1.0,
                seed=46,
                warm_start_mutation_ratio=0.0,
                mid_gate_estimated_seconds_per_eval=0.001,
                high_gate_estimated_seconds_per_eval=0.005,
            ),
            high_fidelity_evaluator=external_high,
        )

    order = (
        "length_ratio",
        "breadth_ratio",
        "height_ratio",
        "axis_z_ratio",
        "longitudinal_pos",
        "cross_section_c",
        "volume_coef",
        "nose_sharpness",
    )
    expected_compatible = [float(compatible[name]) for name in order]
    expected_incompatible = [float(incompatible[name]) for name in order]
    case_dir = project_root / summary.case_id
    evidence_rows = [
        json.loads(line)
        for line in (
            case_dir / "working" / "night_optimization" / "cfd_evidence.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert expected_compatible in captured
    assert expected_incompatible not in captured
    assert evidence_rows[0]["hull_fingerprint"] == hull_fingerprint
    assert evidence_rows[0]["settings_hash"] == settings_hash

    case_log = (case_dir / "logs" / "case.log").read_text(encoding="utf-8")
    assert '"candidate_rows": 2' in case_log
    assert '"eligible": 1' in case_log
    assert '"hull_mismatch": 1' in case_log


def test_run_night_optimization_writes_per_generation_snapshots(
    tmp_path: Path,
) -> None:
    """Spec §10.2 — each generation produces a snapshot directory under
    ``working/night_optimization/generations/gen-NN/`` so an engineer can
    salvage results when the run crashes mid-stream.

    Each ``gen-NN/`` carries:
      * ``population.json`` — vector + objectives per individual
      * ``pareto.json``     — current non-dominated set

    A run-level ``convergence.csv`` records one row per generation with
    header + ``generation, n_individuals, best_obj0, best_obj1, best_obj2,
    mean_obj0`` (3 objectives in this codebase).
    """
    source_path = tmp_path / "hull.stl"
    _write_watertight_stl(source_path)

    summary = run_night_optimization(
        project_root=tmp_path / "projects",
        command=CreateCaseCommand(
            case_name="night-snapshots",
            source_path=str(source_path),
            vessel_length_m=142.0,
            vessel_beam_m=19.1,
            vessel_draft_m=6.0,
            displacement_t=8420.0,
            speed_knots=[18.0, 20.0],
        ),
        config=NightOptimizationConfig(
            population=4,
            generations=2,
            high_fidelity_budget=1,
            runtime_budget_hours=1.0,
            seed=51,
            mid_gate_estimated_seconds_per_eval=0.001,
            high_gate_estimated_seconds_per_eval=0.005,
        ),
    )

    case_dir = tmp_path / "projects" / summary.case_id
    generations_dir = case_dir / "working" / "night_optimization" / "generations"
    assert generations_dir.exists()

    gen0_pop = generations_dir / "gen-00" / "population.json"
    gen1_pop = generations_dir / "gen-01" / "population.json"
    gen0_pareto = generations_dir / "gen-00" / "pareto.json"
    gen1_pareto = generations_dir / "gen-01" / "pareto.json"

    assert gen0_pop.exists()
    assert gen1_pop.exists()
    assert gen0_pareto.exists()
    assert gen1_pareto.exists()

    population = json.loads(gen0_pop.read_text(encoding="utf-8"))
    assert "individuals" in population
    assert len(population["individuals"]) == 4
    for individual in population["individuals"]:
        assert "vector" in individual
        assert "objectives" in individual
        assert len(individual["vector"]) == 8

    pareto = json.loads(gen0_pareto.read_text(encoding="utf-8"))
    assert "candidates" in pareto
    assert len(pareto["candidates"]) >= 1
    for candidate in pareto["candidates"]:
        assert "vector" in candidate
        assert "objectives" in candidate

    convergence_path = (
        case_dir / "working" / "night_optimization" / "convergence.csv"
    )
    assert convergence_path.exists()
    csv_lines = [
        line
        for line in convergence_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(csv_lines) == 3  # 1 header + 2 data rows
    header = csv_lines[0].split(",")
    for column in (
        "generation",
        "n_individuals",
        "best_obj0",
        "best_obj1",
        "best_obj2",
        "mean_obj0",
    ):
        assert column in header
    # First data row's generation index is 0.
    first_row = csv_lines[1].split(",")
    assert first_row[header.index("generation")] == "0"
    assert first_row[header.index("n_individuals")] == "4"


def test_run_night_optimization_transitions_through_new_states(
    tmp_path: Path,
) -> None:
    """The case must enter ``RUNNING_NIGHT_OPTIMIZATION`` during the run
    and finish at ``COMPLETED`` (or ``COMPLETED_WITH_WARNINGS``).

    Hook into the per-generation snapshot callback so we can read the
    persisted case during the run, not only after.
    """
    source_path = tmp_path / "hull.stl"
    _write_watertight_stl(source_path)
    project_root = tmp_path / "projects"
    repository = FilesystemProjectRepository(root_dir=project_root)

    statuses_during_run: list[str] = []
    original_make = run_night_module._make_generation_writer

    def spying_make(*args, **kwargs):  # noqa: D401
        inner = original_make(*args, **kwargs)
        case_for_spy = kwargs.get("case")

        def wrapped(*cb_args, **cb_kwargs):
            inner(*cb_args, **cb_kwargs)
            try:
                reloaded = repository.load_case(case_for_spy.case_id)
                statuses_during_run.append(reloaded.status.value)
            except Exception:
                pass

        return wrapped

    import unittest.mock as _mock

    with _mock.patch.object(
        run_night_module, "_make_generation_writer", side_effect=spying_make
    ):
        summary = run_night_optimization(
            project_root=project_root,
            command=CreateCaseCommand(
                case_name="night-lifecycle",
                source_path=str(source_path),
                vessel_length_m=142.0,
                vessel_beam_m=19.1,
                vessel_draft_m=6.0,
                displacement_t=8420.0,
                speed_knots=[18.0, 20.0],
            ),
            config=NightOptimizationConfig(
                population=4,
                generations=2,
                high_fidelity_budget=1,
                runtime_budget_hours=1.0,
                seed=53,
                mid_gate_estimated_seconds_per_eval=0.001,
                high_gate_estimated_seconds_per_eval=0.005,
            ),
        )

    # During the run (between generations) the case sat in the new state.
    assert CaseStatus.RUNNING_NIGHT_OPTIMIZATION.value in statuses_during_run

    # Final terminal status is one of the two completed flavours.
    assert summary.status in {
        CaseStatus.COMPLETED.value,
        CaseStatus.COMPLETED_WITH_WARNINGS.value,
    }
    final_case = repository.load_case(summary.case_id)
    assert final_case.status in {
        CaseStatus.COMPLETED,
        CaseStatus.COMPLETED_WITH_WARNINGS,
    }


def _kracht_vector_at_offset(index: int) -> KrachtVector:
    """Build a deterministic, in-bounds KrachtVector for synthetic history."""
    return KrachtVector(
        values={
            "length_ratio": 0.020 + index * 0.0005,
            "breadth_ratio": 0.080,
            "height_ratio": 0.250,
            "axis_z_ratio": 0.200,
            "longitudinal_pos": 0.550,
            "cross_section_c": 0.700,
            "volume_coef": 0.600,
            "nose_sharpness": 0.500,
        }
    )


def _read_mid_gate_routing_entries(case_log_path: Path) -> list[dict]:
    """Pull every ``mid_gate_routing`` JSONL row out of ``case.log``."""
    entries: list[dict] = []
    if not case_log_path.exists():
        return entries
    for line in case_log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("stage") == "mid_gate_routing":
            entries.append(payload)
    return entries


def _force_simple_foam_planned_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Force the use case to plan ``simple_foam`` as the high-gate backend
    so history rows tagged ``simple_foam`` are eligible for the GP fit.
    Mocks both the detector and the OpenFOAM runner so the test stays
    fast and self-contained.
    """
    from bulbopt.application.use_cases import run_night_optimization as use_case_module
    from bulbopt.infrastructure.adapters import openfoam_runner

    monkeypatch.setattr(use_case_module, "detect_openfoam_available", lambda: True)

    def fake_run(
        self,
        case_dir,
        *,
        case_manifest=None,
        execute=False,
        timeout_seconds=600,
    ):
        return {
            "status": "executed_ok",
            "is_recoverable": True,
            "high_fidelity_used": True,
            "executed_steps": [],
        }

    monkeypatch.setattr(
        openfoam_runner.OpenFOAMRunnerAdapter,
        "run_case",
        fake_run,
    )


def test_mid_gate_routes_to_gp_when_history_meets_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit 2026-04-26 follow-up: with >= ``gp_surrogate_min_history``
    real-CFD history rows the mid-gate must wire the GP surrogate, and the
    case log must announce that decision so the engineer reading the log
    can tell the GA is **not** running blind on the analytic proxy."""
    from bulbopt.optimization.learning.history_store import HistoryStore

    _force_simple_foam_planned_backend(monkeypatch)

    source_path = tmp_path / "hull.stl"
    _write_watertight_stl(source_path)
    history_path = tmp_path / "history" / "history.jsonl"
    history_store = HistoryStore(path=history_path)
    # Seed exactly 12 real-CFD rows tagged as ``simple_foam`` — matches the
    # threshold so the GP path activates. Cd values vary so the kernel has
    # signal.
    for index in range(12):
        history_store.record(
            _kracht_vector_at_offset(index),
            cd=0.40 - index * 0.01,
            backend="simple_foam",
        )

    summary = run_night_optimization(
        project_root=tmp_path / "projects",
        command=CreateCaseCommand(
            case_name="night-mid-gate-gp",
            source_path=str(source_path),
            vessel_length_m=142.0,
            vessel_beam_m=19.1,
            vessel_draft_m=6.0,
            displacement_t=8420.0,
            speed_knots=[18.0, 20.0],
        ),
        config=NightOptimizationConfig(
            population=4,
            generations=2,
            high_fidelity_budget=1,
            runtime_budget_hours=1.0,
            seed=101,
            mid_gate_estimated_seconds_per_eval=0.001,
            high_gate_estimated_seconds_per_eval=0.005,
            gp_surrogate_min_history=12,
            history_path=history_path,
        ),
    )

    case_log_path = (
        tmp_path / "projects" / summary.case_id / "logs" / "case.log"
    )
    routing_entries = _read_mid_gate_routing_entries(case_log_path)
    assert len(routing_entries) == 1, (
        f"expected exactly one mid_gate_routing entry, got {routing_entries}"
    )
    entry = routing_entries[0]
    assert entry["status"] == "gp"
    assert entry["training_points"] >= 12
    assert entry["threshold"] == 12


def test_mid_gate_routes_to_proxy_fallback_when_history_below_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With fewer than ``gp_surrogate_min_history`` real-CFD rows the
    mid-gate must fall back to the analytic proxy AND the case log must
    say so loudly — including a ``warning`` field — so an engineer
    reading the log knows the GA is operating without a real-Cd
    surrogate."""
    from bulbopt.optimization.learning.history_store import HistoryStore

    _force_simple_foam_planned_backend(monkeypatch)

    source_path = tmp_path / "hull.stl"
    _write_watertight_stl(source_path)
    history_path = tmp_path / "history" / "history.jsonl"
    history_store = HistoryStore(path=history_path)
    # Seed 5 rows — well below the 12-row threshold. Tagged ``simple_foam``
    # so they would have been eligible for the GP fit if there were enough.
    for index in range(5):
        history_store.record(
            _kracht_vector_at_offset(index),
            cd=0.40 - index * 0.01,
            backend="simple_foam",
        )

    summary = run_night_optimization(
        project_root=tmp_path / "projects",
        command=CreateCaseCommand(
            case_name="night-mid-gate-proxy-fallback",
            source_path=str(source_path),
            vessel_length_m=142.0,
            vessel_beam_m=19.1,
            vessel_draft_m=6.0,
            displacement_t=8420.0,
            speed_knots=[18.0, 20.0],
        ),
        config=NightOptimizationConfig(
            population=4,
            generations=2,
            high_fidelity_budget=1,
            runtime_budget_hours=1.0,
            seed=102,
            mid_gate_estimated_seconds_per_eval=0.001,
            high_gate_estimated_seconds_per_eval=0.005,
            gp_surrogate_min_history=12,
            history_path=history_path,
        ),
    )

    case_log_path = (
        tmp_path / "projects" / summary.case_id / "logs" / "case.log"
    )
    routing_entries = _read_mid_gate_routing_entries(case_log_path)
    assert len(routing_entries) == 1, (
        f"expected exactly one mid_gate_routing entry, got {routing_entries}"
    )
    entry = routing_entries[0]
    assert entry["status"] == "proxy_fallback_blind"
    assert entry["threshold"] == 12
    assert entry["history_size"] == 5
    assert "warning" in entry and entry["warning"]

