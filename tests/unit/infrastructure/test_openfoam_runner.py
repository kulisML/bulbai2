from __future__ import annotations

import json
from pathlib import Path

import pytest

from bulbopt.execution.checkpoints.file_checkpoint_store import FileCheckpointStore
from bulbopt.execution.worker.local_worker import LocalWorker
from bulbopt.infrastructure.adapters.openfoam_adapter import OpenFOAMAdapter
from bulbopt.infrastructure.adapters.openfoam_runner import OpenFOAMRunnerAdapter


def test_openfoam_adapter_writes_initial_fields_and_force_coeffs_fo(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Spec §9: the high-fidelity gate needs simpleFoam to actually solve
    the flow, which requires (a) a 0/ time directory with U and p initial
    conditions, (b) transportProperties + turbulenceProperties, and
    (c) a forceCoeffs function object embedded in controlDict so the
    run emits postProcessing/forces/<time>/coefficient.dat."""
    case_dir = tmp_path / "case"
    geometry_path = case_dir / "candidate.stl"
    geometry_path.parent.mkdir(parents=True, exist_ok=True)
    geometry_path.write_text("solid demo\nendsolid demo\n", encoding="utf-8")

    builder = OpenFOAMAdapter()
    monkeypatch.setattr(builder, "is_available", lambda: True)
    builder.build_case(
        case_dir,
        best_candidate_id="candidate-init",
        best_candidate_geometry_path=geometry_path,
    )

    of_case_dir = case_dir / "working" / "openfoam_case"

    # Initial fields for simpleFoam.
    assert (of_case_dir / "0" / "U").exists()
    assert (of_case_dir / "0" / "p").exists()
    assert (of_case_dir / "0" / "k").exists()
    assert (of_case_dir / "0" / "omega").exists()
    assert (of_case_dir / "0" / "nut").exists()

    # Thermo/turbulence constants.
    assert (of_case_dir / "constant" / "transportProperties").exists()
    assert (of_case_dir / "constant" / "turbulenceProperties").exists()

    # controlDict must reference the forceCoeffs function object so the
    # solver writes coefficient.dat without extra setup.
    control_dict = (of_case_dir / "system" / "controlDict").read_text(encoding="utf-8")
    assert "functions" in control_dict
    assert "forceCoeffs" in control_dict
    assert "patches" in control_dict


def test_openfoam_runner_invokes_solver_via_absolute_path_when_bin_dir_configured(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Spec §4.8 real-world path: Windows ``CreateProcessW`` uses the parent
    process's ``%PATH%`` to locate a bare command name regardless of the
    ``env`` argument to ``subprocess.run``. The runner must therefore resolve
    each solver in ``DEFAULT_SOLVER_CHAIN`` to its absolute executable path
    via the configured bin dir, otherwise the subprocess fails with
    ``FileNotFoundError`` even though the binary sits right there.
    """
    import subprocess as subprocess_module

    configured_bin = tmp_path / "fake-foam-bin"
    configured_bin.mkdir()
    # Create fake binaries so the adapter can actually see them on disk.
    for solver in ("blockMesh.exe", "snappyHexMesh.exe"):
        (configured_bin / solver).write_text("fake", encoding="utf-8")
    monkeypatch.setenv("BULBOPT_OPENFOAM_BIN", str(configured_bin))

    case_dir = tmp_path / "case"
    geometry_path = case_dir / "candidate.stl"
    geometry_path.parent.mkdir(parents=True, exist_ok=True)
    geometry_path.write_text("solid demo\nendsolid demo\n", encoding="utf-8")

    builder = OpenFOAMAdapter()
    monkeypatch.setattr(builder, "is_available", lambda: True)
    manifest = builder.build_case(
        case_dir,
        best_candidate_id="candidate-abs",
        best_candidate_geometry_path=geometry_path,
    )

    runner = OpenFOAMRunnerAdapter()
    monkeypatch.setattr(runner, "is_available", lambda: True)

    commands_launched: list[list[str]] = []

    class _FakeCompletedProcess:
        def __init__(self, args, returncode: int, stdout: str = "", stderr: str = "") -> None:
            self.args = args
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(args, **kwargs):
        commands_launched.append(list(args))
        return _FakeCompletedProcess(args, 0)

    monkeypatch.setattr(subprocess_module, "run", fake_run)

    runner.run_case(
        case_dir / "working" / "openfoam_case",
        case_manifest=manifest,
        execute=True,
    )

    assert commands_launched, "Expected subprocess.run to be called"
    first_cmd = commands_launched[0][0]
    # Must be an absolute path ending with .exe, not a bare name.
    assert first_cmd.lower().endswith("blockmesh.exe"), (
        f"Expected absolute blockMesh.exe path, got: {first_cmd}"
    )
    from pathlib import Path as _P

    assert _P(first_cmd).is_absolute(), (
        f"Expected absolute path for solver, got relative: {first_cmd}"
    )
    # Second command is snappyHexMesh; solver flags come after the exe.
    second_cmd = commands_launched[1][0]
    assert second_cmd.lower().endswith("snappyhexmesh.exe"), (
        f"Expected absolute snappyHexMesh.exe path, got: {second_cmd}"
    )
    # Original flags preserved.
    assert "-overwrite" in commands_launched[1], (
        f"Expected -overwrite flag preserved; got: {commands_launched[1]}"
    )


def test_openfoam_runner_prepends_configured_bin_dir_to_subprocess_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Spec §4.8 real-world path: the solver chain must reach blockMesh.exe
    when the engineer sets ``BULBOPT_OPENFOAM_BIN`` (the portable way to
    drive a MinGW Windows build). The runner must prepend that directory
    to the subprocess PATH even when it is not on the parent shell's PATH.
    """
    import subprocess as subprocess_module

    configured_bin = tmp_path / "fake-foam-bin"
    configured_bin.mkdir()
    # Ensure BULBOPT_OPENFOAM_BIN points to the fake dir.
    monkeypatch.setenv("BULBOPT_OPENFOAM_BIN", str(configured_bin))

    case_dir = tmp_path / "case"
    geometry_path = case_dir / "candidate.stl"
    geometry_path.parent.mkdir(parents=True, exist_ok=True)
    geometry_path.write_text("solid demo\nendsolid demo\n", encoding="utf-8")

    builder = OpenFOAMAdapter()
    monkeypatch.setattr(builder, "is_available", lambda: True)
    manifest = builder.build_case(
        case_dir,
        best_candidate_id="candidate-path",
        best_candidate_geometry_path=geometry_path,
    )

    runner = OpenFOAMRunnerAdapter()
    monkeypatch.setattr(runner, "is_available", lambda: True)

    observed: list[str] = []

    class _FakeCompletedProcess:
        def __init__(self, args, returncode: int, stdout: str = "", stderr: str = "") -> None:
            self.args = args
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(args, **kwargs):
        observed.append(kwargs.get("env", {}).get("PATH", ""))
        return _FakeCompletedProcess(args, 0)

    monkeypatch.setattr(subprocess_module, "run", fake_run)

    runner.run_case(
        case_dir / "working" / "openfoam_case",
        case_manifest=manifest,
        execute=True,
    )

    assert observed, "Expected subprocess.run to be called"
    first_path = observed[0]
    # On Windows the adapter shortens paths, so the long form, the short form
    # (8.3 name) and at minimum the leaf name must appear in the prepended PATH.
    long_form = str(configured_bin)
    leaf = configured_bin.name
    assert (
        long_form in first_path
        or leaf in first_path
        or leaf[:6].upper() in first_path.upper()
    ), f"Expected configured bin dir in subprocess PATH; got: {first_path[:300]}"
    # Must be first entry so MinGW loader resolves before system dirs.
    assert first_path.split(";")[0].lower().split("\\")[-1].startswith(leaf[:6].lower()) or \
           first_path.split(";")[0].upper().split("\\")[-1].startswith(leaf[:6].upper()), \
        f"Expected configured bin dir to be the first PATH entry; got: {first_path[:200]}"


def test_openfoam_runner_writes_full_solver_logs_to_case_logs_dir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Spec §9 logs/ directory: the full stdout/stderr of each solver step
    must be persisted to ``case_dir/logs/openfoam/<solver>.log`` (and
    ``.err.log``) so an engineer can tail or grep them later without
    rerunning. The manifest keeps the truncated tails for the quick-look
    UI but the files hold the complete output.
    """
    import subprocess as subprocess_module

    case_dir = tmp_path / "case-log"
    geometry_path = case_dir / "candidate.stl"
    geometry_path.parent.mkdir(parents=True, exist_ok=True)
    geometry_path.write_text("solid demo\nendsolid demo\n", encoding="utf-8")

    builder = OpenFOAMAdapter()
    monkeypatch.setattr(builder, "is_available", lambda: True)
    manifest = builder.build_case(
        case_dir,
        best_candidate_id="candidate-log",
        best_candidate_geometry_path=geometry_path,
    )

    runner = OpenFOAMRunnerAdapter()
    monkeypatch.setattr(runner, "is_available", lambda: True)

    long_stdout = "HEADER\n" + ("x" * 5000) + "\nFOOTER"
    long_stderr = "WARN\n" + ("y" * 3500)

    class _FakeCompletedProcess:
        def __init__(self, args, returncode, stdout, stderr):
            self.args = args
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(args, **kwargs):
        if args[0].lower().endswith("blockmesh") or args[0].lower().endswith("blockmesh.exe"):
            return _FakeCompletedProcess(args, 0, long_stdout, long_stderr)
        return _FakeCompletedProcess(args, 0, "snap OK", "")

    monkeypatch.setattr(subprocess_module, "run", fake_run)

    result = runner.run_case(
        case_dir / "working" / "openfoam_case",
        case_manifest=manifest,
        execute=True,
    )

    logs_dir = case_dir / "logs" / "openfoam"
    assert logs_dir.is_dir(), "Expected logs/openfoam/ to be created"
    block_log = logs_dir / "blockMesh.log"
    block_err = logs_dir / "blockMesh.err.log"
    snap_log = logs_dir / "snappyHexMesh.log"
    assert block_log.exists()
    assert block_err.exists()
    assert snap_log.exists()

    # Full content preserved (not truncated to 2000 chars like the tail).
    assert "HEADER" in block_log.read_text(encoding="utf-8")
    assert "FOOTER" in block_log.read_text(encoding="utf-8")
    assert len(block_log.read_text(encoding="utf-8")) > 2000
    assert len(block_err.read_text(encoding="utf-8")) > 2000

    # Manifest references the log files for each step.
    executed = result.get("executed_steps", [])
    assert executed and "stdout_log" in executed[0]
    assert executed[0]["stdout_log"].endswith("blockMesh.log")
    assert executed[0]["stderr_log"].endswith("blockMesh.err.log")


def test_openfoam_runner_includes_simple_foam_in_default_solver_chain(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Real drag numbers require simpleFoam to follow mesh generation."""
    import subprocess as subprocess_module

    case_dir = tmp_path / "case"
    geometry_path = case_dir / "candidate.stl"
    geometry_path.parent.mkdir(parents=True, exist_ok=True)
    geometry_path.write_text("solid demo\nendsolid demo\n", encoding="utf-8")

    builder = OpenFOAMAdapter()
    monkeypatch.setattr(builder, "is_available", lambda: True)
    manifest = builder.build_case(
        case_dir,
        best_candidate_id="candidate-sf",
        best_candidate_geometry_path=geometry_path,
    )

    runner = OpenFOAMRunnerAdapter()
    monkeypatch.setattr(runner, "is_available", lambda: True)

    invocations: list[list[str]] = []

    class _FakeProc:
        def __init__(self, args, rc=0, out="", err=""):
            self.args, self.returncode, self.stdout, self.stderr = args, rc, out, err

    def fake_run(args, **kwargs):
        invocations.append(list(args))
        return _FakeProc(args, 0)

    monkeypatch.setattr(subprocess_module, "run", fake_run)

    runner.run_case(
        case_dir / "working" / "openfoam_case",
        case_manifest=manifest,
        execute=True,
    )

    commands = [cmd[0] for cmd in invocations]
    # simpleFoam must appear after blockMesh + snappyHexMesh.
    commands_lower = [c.lower() for c in commands]
    assert any("blockmesh" in c for c in commands_lower)
    assert any("snappyhexmesh" in c for c in commands_lower)
    assert any("simplefoam" in c for c in commands_lower)


def test_openfoam_runner_records_simple_foam_convergence_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The run manifest should expose whether simpleFoam produced residuals
    and reached ``End`` so CFD evidence can distinguish a completed solve from
    a merely successful subprocess."""
    import subprocess as subprocess_module

    case_dir = tmp_path / "case-convergence"
    geometry_path = case_dir / "candidate.stl"
    geometry_path.parent.mkdir(parents=True, exist_ok=True)
    geometry_path.write_text("solid demo\nendsolid demo\n", encoding="utf-8")

    builder = OpenFOAMAdapter()
    monkeypatch.setattr(builder, "is_available", lambda: True)
    manifest = builder.build_case(
        case_dir,
        best_candidate_id="candidate-conv",
        best_candidate_geometry_path=geometry_path,
    )

    runner = OpenFOAMRunnerAdapter()
    monkeypatch.setattr(runner, "is_available", lambda: True)

    class _FakeProc:
        def __init__(self, args, rc=0, stdout="", stderr=""):
            self.args, self.returncode, self.stdout, self.stderr = args, rc, stdout, stderr

    simple_log = (
        "Time = 200\n"
        "smoothSolver:  Solving for Ux, Initial residual = 1e-04, Final residual = 1e-06, No Iterations 2\n"
        "GAMG:  Solving for p, Initial residual = 2e-04, Final residual = 2e-06, No Iterations 3\n"
        "End\n"
    )

    def fake_run(args, **kwargs):
        solver = args[0].lower()
        if "checkmesh" in solver:
            return _FakeProc(args, 0, stdout="Mesh OK.\n")
        if "simplefoam" in solver:
            return _FakeProc(args, 0, stdout=simple_log)
        return _FakeProc(args, 0, stdout="ok\n")

    monkeypatch.setattr(subprocess_module, "run", fake_run)

    result = runner.run_case(
        case_dir / "working" / "openfoam_case",
        case_manifest=manifest,
        execute=True,
    )

    simple_step = next(
        step
        for step in result["executed_steps"]
        if step["command"][0] == "simpleFoam"
    )
    report = simple_step["solver_report"]
    assert report["residuals_available"] is True
    assert report["completed"] is True
    assert report["last_time"] == pytest.approx(200.0)
    assert report["residual_equations"] == ["Ux", "p"]


def test_openfoam_runner_executes_solver_chain_when_available(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Spec §4.8: when OpenFOAM is available and execution is requested, the
    runner must actually invoke the solver chain (blockMesh + snappyHexMesh)
    and record each step's exit code in the run manifest so the report can
    flip ``high_fidelity_used`` to True.
    """
    import subprocess as subprocess_module

    case_dir = tmp_path / "case"
    geometry_path = case_dir / "candidate.stl"
    geometry_path.parent.mkdir(parents=True, exist_ok=True)
    geometry_path.write_text("solid demo\nendsolid demo\n", encoding="utf-8")

    builder = OpenFOAMAdapter()
    monkeypatch.setattr(builder, "is_available", lambda: True)
    manifest = builder.build_case(
        case_dir,
        best_candidate_id="candidate-42",
        best_candidate_geometry_path=geometry_path,
    )

    runner = OpenFOAMRunnerAdapter()
    monkeypatch.setattr(runner, "is_available", lambda: True)

    invocations: list[dict] = []

    class _FakeCompletedProcess:
        def __init__(self, args, returncode: int, stdout: str = "", stderr: str = "") -> None:
            self.args = args
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(args, **kwargs):
        invocations.append({"args": list(args), "cwd": str(kwargs.get("cwd"))})
        return _FakeCompletedProcess(args, 0, stdout="ok\n")

    monkeypatch.setattr(subprocess_module, "run", fake_run)

    result = runner.run_case(
        case_dir / "working" / "openfoam_case",
        case_manifest=manifest,
        execute=True,
    )

    assert result["status"] == "executed_ok"
    assert result["is_recoverable"] is True
    assert result["best_candidate_id"] == "candidate-42"
    assert len(invocations) >= 2, "Expected at least blockMesh + snappyHexMesh"
    assert invocations[0]["args"][0] == "blockMesh"
    assert invocations[1]["args"][0] == "snappyHexMesh"
    executed_steps = result.get("executed_steps", [])
    assert executed_steps, "manifest must record each executed step"
    assert all(step["returncode"] == 0 for step in executed_steps)


def test_openfoam_runner_marks_failed_execution_when_solver_exits_nonzero(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """When any step in the solver chain returns a non-zero exit code, the
    runner must stop, mark the manifest ``executed_failed`` with a reason
    pointing at the failing step, and keep the case recoverable so the user
    can re-run via Resume after fixing the solver environment.
    """
    import subprocess as subprocess_module

    case_dir = tmp_path / "case"
    geometry_path = case_dir / "candidate.stl"
    geometry_path.parent.mkdir(parents=True, exist_ok=True)
    geometry_path.write_text("solid demo\nendsolid demo\n", encoding="utf-8")

    builder = OpenFOAMAdapter()
    monkeypatch.setattr(builder, "is_available", lambda: True)
    manifest = builder.build_case(
        case_dir,
        best_candidate_id="candidate-99",
        best_candidate_geometry_path=geometry_path,
    )

    runner = OpenFOAMRunnerAdapter()
    monkeypatch.setattr(runner, "is_available", lambda: True)

    class _FakeCompletedProcess:
        def __init__(self, args, returncode: int, stdout: str = "", stderr: str = "") -> None:
            self.args = args
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(args, **kwargs):
        if args[0] == "blockMesh":
            return _FakeCompletedProcess(args, 0, stdout="ok\n")
        # snappyHexMesh fails
        return _FakeCompletedProcess(args, 1, stdout="", stderr="boom\n")

    monkeypatch.setattr(subprocess_module, "run", fake_run)

    result = runner.run_case(
        case_dir / "working" / "openfoam_case",
        case_manifest=manifest,
        execute=True,
    )

    assert result["status"] == "executed_failed"
    assert "snappyHexMesh" in result["reason"]
    assert result["is_recoverable"] is True
    assert result["high_fidelity_used"] is False
    assert len(result["executed_steps"]) == 2
    assert result["executed_steps"][0]["returncode"] == 0
    assert result["executed_steps"][1]["returncode"] == 1


def test_openfoam_runner_returns_recoverable_skip_when_solver_is_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    case_dir = tmp_path / "case"
    geometry_path = case_dir / "candidate.stl"
    geometry_path.parent.mkdir(parents=True, exist_ok=True)
    geometry_path.write_text("solid demo\nendsolid demo\n", encoding="utf-8")

    builder = OpenFOAMAdapter()
    monkeypatch.setattr(builder, "is_available", lambda: False)
    manifest = builder.build_case(
        case_dir,
        best_candidate_id="candidate-7",
        best_candidate_geometry_path=geometry_path,
    )

    runner = OpenFOAMRunnerAdapter()
    monkeypatch.setattr(runner, "is_available", lambda: False)
    result = runner.run_case(case_dir / "working" / "openfoam_case", case_manifest=manifest)

    assert result["status"] == "skipped"
    assert result["reason"] == "openfoam_unavailable"
    assert result["is_recoverable"] is True
    assert (case_dir / "working" / "openfoam_case" / "openfoam_run_manifest.json").exists()


def test_openfoam_runner_fails_early_when_checkmesh_reports_illegal_cells(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """mesh-quality design §4 Part B: checkMesh sits between
    snappyHexMesh and simpleFoam; a positive illegal-cell count must
    stop the chain with ``executed_failed`` (reason pointing at checkMesh)
    before the expensive simpleFoam solve burns any CPU."""
    import subprocess as subprocess_module

    case_dir = tmp_path / "case-check"
    geometry_path = case_dir / "candidate.stl"
    geometry_path.parent.mkdir(parents=True, exist_ok=True)
    geometry_path.write_text("solid demo\nendsolid demo\n", encoding="utf-8")

    builder = OpenFOAMAdapter()
    monkeypatch.setattr(builder, "is_available", lambda: True)
    manifest = builder.build_case(
        case_dir,
        best_candidate_id="candidate-bad-mesh",
        best_candidate_geometry_path=geometry_path,
    )

    runner = OpenFOAMRunnerAdapter()
    monkeypatch.setattr(runner, "is_available", lambda: True)

    bad_check_log = (
        "Max non-orthogonality = 88.4\n"
        "Max skewness = 9.1\n"
        "Max aspect ratio = 240.1\n"
        "***Number of cells with incorrect orientation: 4\n"
        "cells with zero or negative volume: 1\n"
        "Failed 2 mesh checks.\n"
    )

    class _FakeProc:
        def __init__(self, args, rc, stdout="", stderr=""):
            self.args, self.returncode, self.stdout, self.stderr = args, rc, stdout, stderr

    invocations: list[str] = []

    def fake_run(args, **kwargs):
        solver = args[0].lower()
        invocations.append(solver)
        if "checkmesh" in solver:
            # checkMesh typically returns 0 even when illegal cells exist;
            # the parser-based gate is what stops the chain.
            return _FakeProc(args, 0, stdout=bad_check_log)
        return _FakeProc(args, 0, stdout="ok\n")

    monkeypatch.setattr(subprocess_module, "run", fake_run)

    result = runner.run_case(
        case_dir / "working" / "openfoam_case",
        case_manifest=manifest,
        execute=True,
    )

    assert result["status"] == "executed_failed"
    assert "checkMesh" in result["reason"]
    # simpleFoam must NOT have been invoked: we bail after checkMesh.
    assert not any("simplefoam" in cmd for cmd in invocations)
    # The check_mesh_report rides in the executed_steps entry for the
    # checkMesh step so downstream reporting can surface the metrics.
    check_step = next(
        (step for step in result["executed_steps"]
         if step["command"][0] == "checkMesh"),
        None,
    )
    assert check_step is not None
    assert check_step["check_mesh_report"]["n_illegal_cells"] == 5
    assert check_step["check_mesh_report"]["max_skewness"] == pytest.approx(9.1)


def test_openfoam_runner_continues_when_checkmesh_reports_clean(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A clean checkMesh (no illegal cells) must let simpleFoam run."""
    import subprocess as subprocess_module

    case_dir = tmp_path / "case-clean"
    geometry_path = case_dir / "candidate.stl"
    geometry_path.parent.mkdir(parents=True, exist_ok=True)
    geometry_path.write_text("solid demo\nendsolid demo\n", encoding="utf-8")

    builder = OpenFOAMAdapter()
    monkeypatch.setattr(builder, "is_available", lambda: True)
    manifest = builder.build_case(
        case_dir,
        best_candidate_id="candidate-clean-mesh",
        best_candidate_geometry_path=geometry_path,
    )

    runner = OpenFOAMRunnerAdapter()
    monkeypatch.setattr(runner, "is_available", lambda: True)

    clean_check_log = (
        "Max non-orthogonality = 42.0\n"
        "Max skewness = 1.7\n"
        "Max aspect ratio = 4.8\n"
        "Mesh OK.\n"
    )

    class _FakeProc:
        def __init__(self, args, rc, stdout="", stderr=""):
            self.args, self.returncode, self.stdout, self.stderr = args, rc, stdout, stderr

    invocations: list[str] = []

    def fake_run(args, **kwargs):
        solver = args[0].lower()
        invocations.append(solver)
        if "checkmesh" in solver:
            return _FakeProc(args, 0, stdout=clean_check_log)
        return _FakeProc(args, 0, stdout="ok\n")

    monkeypatch.setattr(subprocess_module, "run", fake_run)

    result = runner.run_case(
        case_dir / "working" / "openfoam_case",
        case_manifest=manifest,
        execute=True,
    )

    assert result["status"] == "executed_ok"
    assert any("simplefoam" in cmd for cmd in invocations)


def test_openfoam_runner_can_be_executed_through_local_worker(tmp_path: Path, monkeypatch) -> None:
    case_dir = tmp_path / "case"
    geometry_path = case_dir / "candidate.stl"
    geometry_path.parent.mkdir(parents=True, exist_ok=True)
    geometry_path.write_text("solid demo\nendsolid demo\n", encoding="utf-8")

    builder = OpenFOAMAdapter()
    monkeypatch.setattr(builder, "is_available", lambda: False)
    manifest = builder.build_case(
        case_dir,
        best_candidate_id="candidate-9",
        best_candidate_geometry_path=geometry_path,
    )

    runner = OpenFOAMRunnerAdapter()
    monkeypatch.setattr(runner, "is_available", lambda: False)
    worker = LocalWorker(checkpoint_store=FileCheckpointStore(root_dir=tmp_path / "checkpoints"))

    result = worker.run(
        "case-foam",
        "openfoam-runner",
        lambda: runner.run_case(case_dir / "working" / "openfoam_case", case_manifest=manifest),
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "openfoam_unavailable"
    checkpoint_payload = json.loads(
        (tmp_path / "checkpoints" / "case-foam-openfoam-runner.json").read_text(encoding="utf-8")
    )
    assert checkpoint_payload["status"] == "completed"
    assert checkpoint_payload["result"]["status"] == "skipped"
