from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from pathlib import Path

from bulbopt.infrastructure.adapters.openfoam_adapter import (
    OpenFOAMAdapter,
    _configured_bin_dir,
    _detect_wsl_openfoam,
    _shorten_path,
)


class OpenFOAMRunnerAdapter:
    """Optional runner boundary for worker-driven CFD execution (spec §4.8)."""

    DEFAULT_SOLVER_CHAIN: tuple[tuple[str, ...], ...] = (
        ("blockMesh",),
        ("snappyHexMesh", "-overwrite"),
        # checkMesh sits between snappyHexMesh and simpleFoam so we can
        # fail the chain early when the generated mesh has illegal cells
        # — simpleFoam on a broken mesh wastes minutes of CPU before
        # blowing up with a cryptic error (mesh-quality design §4 Part B).
        ("checkMesh",),
        ("simpleFoam",),
    )

    def is_available(self) -> bool:
        return OpenFOAMAdapter().is_available()

    def run_case(
        self,
        openfoam_case_dir: Path,
        *,
        case_manifest: dict | None = None,
        execute: bool = False,
        timeout_seconds: int = 600,
    ) -> dict[str, bool | str]:
        manifest = case_manifest or self._read_case_manifest(openfoam_case_dir)
        run_manifest_path = openfoam_case_dir / "openfoam_run_manifest.json"

        if not self.is_available():
            result = {
                "runner_status": "skipped",
                "runner_reason": "openfoam_unavailable",
                "runner_recoverable": True,
                "runner_case_directory": str(openfoam_case_dir),
                "status": "skipped",
                "reason": "openfoam_unavailable",
                "is_recoverable": True,
            }
            run_manifest_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            return result

        if execute:
            wsl_info = (
                _detect_wsl_openfoam()
                if manifest.get("runtime_backend") == "wsl"
                else None
            )
            if wsl_info and not _configured_bin_dir():
                return self._execute_solver_chain_wsl(
                    openfoam_case_dir=openfoam_case_dir,
                    manifest=manifest,
                    run_manifest_path=run_manifest_path,
                    timeout_seconds=timeout_seconds,
                    wsl_info=wsl_info,
                )
            return self._execute_solver_chain(
                openfoam_case_dir=openfoam_case_dir,
                manifest=manifest,
                run_manifest_path=run_manifest_path,
                timeout_seconds=timeout_seconds,
            )

        result = {
            "runner_status": "ready",
            "runner_reason": "available_for_execution",
            "runner_recoverable": True,
            "runner_case_directory": str(openfoam_case_dir),
            "recommended_commands": [
                "blockMesh",
                "snappyHexMesh -overwrite",
                "interFoam",
            ],
            "status": "ready",
            "reason": "available_for_execution",
            "is_recoverable": True,
            "best_candidate_id": manifest.get("best_candidate_id"),
        }
        run_manifest_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def _execute_solver_chain_wsl(
        self,
        *,
        openfoam_case_dir: Path,
        manifest: dict,
        run_manifest_path: Path,
        timeout_seconds: int,
        wsl_info: dict[str, str],
    ) -> dict[str, bool | str]:
        executed_steps: list[dict] = []
        overall_returncode = 0
        failing_step: str | None = None
        case_dir = openfoam_case_dir.parents[1]
        solver_logs_dir = case_dir / "logs" / "openfoam"
        solver_logs_dir.mkdir(parents=True, exist_ok=True)
        wsl_case_dir = self._to_wsl_path(openfoam_case_dir)
        if not wsl_case_dir:
            result = {
                "runner_status": "executed_failed",
                "runner_reason": "solver_chain_failed_at_path_conversion",
                "runner_recoverable": True,
                "runner_case_directory": str(openfoam_case_dir),
                "runtime_backend": "wsl",
                "wsl_distro": wsl_info.get("distro", "unknown"),
                "wsl_bashrc": wsl_info.get("bashrc", ""),
                "status": "executed_failed",
                "reason": "solver_chain_failed_at_path_conversion",
                "is_recoverable": True,
                "best_candidate_id": manifest.get("best_candidate_id"),
                "executed_steps": [],
                "high_fidelity_used": False,
            }
            run_manifest_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            return result

        from bulbopt.infrastructure.adapters.mesh_check_parser import (
            parse_check_mesh_output,
        )

        for command in self.DEFAULT_SOLVER_CHAIN:
            solver_name = command[0]
            stdout_log_path = solver_logs_dir / f"{solver_name}.log"
            stderr_log_path = solver_logs_dir / f"{solver_name}.err.log"
            shell_command = (
                f"source {shlex.quote(wsl_info['bashrc'])} >/dev/null 2>&1 && "
                f"cd {shlex.quote(wsl_case_dir)} && "
                + " ".join(shlex.quote(part) for part in command)
            )
            try:
                completed = subprocess.run(
                    ["wsl", "-d", wsl_info.get("distro", "Ubuntu"), "bash", "-lc", shell_command],
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
                stdout_text = completed.stdout or ""
                stderr_text = completed.stderr or ""
                stdout_log_path.write_text(stdout_text, encoding="utf-8")
                stderr_log_path.write_text(stderr_text, encoding="utf-8")
                step = {
                    "command": list(command),
                    "returncode": int(completed.returncode),
                    "stdout_tail": stdout_text[-2000:],
                    "stderr_tail": stderr_text[-2000:],
                    "stdout_log": str(stdout_log_path),
                    "stderr_log": str(stderr_log_path),
                }
                executed_steps.append(step)
                if completed.returncode != 0:
                    overall_returncode = completed.returncode
                    failing_step = command[0]
                    break
                if solver_name == "checkMesh":
                    report = parse_check_mesh_output(stdout_text)
                    step["check_mesh_report"] = {
                        "max_non_orthogonality": report["max_non_orthogonality"],
                        "max_skewness": report["max_skewness"],
                        "max_aspect_ratio": report["max_aspect_ratio"],
                        "n_illegal_cells": report["n_illegal_cells"],
                        "failed_checks": report["failed_checks"],
                        "mesh_ok": report["mesh_ok"],
                    }
                    if int(report["n_illegal_cells"]) > 0:
                        overall_returncode = 1
                        failing_step = "checkMesh"
                        break
                if solver_name == "simpleFoam":
                    step["solver_report"] = _parse_simple_foam_output(stdout_text)
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
                error_text = str(exc)
                stderr_log_path.write_text(error_text, encoding="utf-8")
                executed_steps.append(
                    {
                        "command": list(command),
                        "returncode": -1,
                        "error": error_text,
                        "stdout_log": str(stdout_log_path),
                        "stderr_log": str(stderr_log_path),
                    }
                )
                overall_returncode = -1
                failing_step = command[0]
                break

        status = "executed_ok" if overall_returncode == 0 and failing_step is None else "executed_failed"
        reason = (
            "solver_chain_completed"
            if status == "executed_ok"
            else f"solver_chain_failed_at_{failing_step or 'unknown'}"
        )
        result = {
            "runner_status": status,
            "runner_reason": reason,
            "runner_recoverable": True,
            "runner_case_directory": str(openfoam_case_dir),
            "runtime_backend": "wsl",
            "wsl_distro": wsl_info.get("distro", "unknown"),
            "wsl_bashrc": wsl_info.get("bashrc", ""),
            "status": status,
            "reason": reason,
            "is_recoverable": True,
            "best_candidate_id": manifest.get("best_candidate_id"),
            "executed_steps": executed_steps,
            "high_fidelity_used": status == "executed_ok",
        }
        run_manifest_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def _execute_solver_chain(
        self,
        *,
        openfoam_case_dir: Path,
        manifest: dict,
        run_manifest_path: Path,
        timeout_seconds: int,
    ) -> dict[str, bool | str]:
        executed_steps: list[dict] = []
        overall_returncode = 0
        failing_step: str | None = None

        # Per-solver log directory lives next to the case log (spec §9):
        #   <case_dir>/logs/openfoam/<solver>.log + .err.log
        # openfoam_case_dir is <case_dir>/working/openfoam_case so
        # parents[1] == <case_dir>.
        case_dir = openfoam_case_dir.parents[1]
        solver_logs_dir = case_dir / "logs" / "openfoam"
        solver_logs_dir.mkdir(parents=True, exist_ok=True)

        # Build subprocess env with the configured OpenFOAM bin directory
        # prepended to PATH. On Windows with non-ASCII install paths, switch
        # both cwd and bin_dir to their 8.3 short form so the MinGW dynamic
        # linker resolves the DLL dependencies.
        subprocess_env = os.environ.copy()
        bin_dir = _configured_bin_dir()
        short_bin: str | None = None
        if bin_dir:
            short_bin = _shorten_path(bin_dir)
            subprocess_env["PATH"] = short_bin + os.pathsep + subprocess_env.get("PATH", "")
            # Derive WM_PROJECT_DIR / HOME from bin_dir so the solver can locate
            # the global etc/controlDict. Bin dir shape:
            #   <HOME>/<WM_PROJECT_DIR_NAME>/platforms/<TYPE>/bin
            # so WM_PROJECT_DIR is bin_dir.parents[2] and HOME is parents[3].
            from pathlib import Path as _Path

            bin_path = _Path(bin_dir)
            if len(bin_path.parents) >= 4:
                wm_project_dir = bin_path.parents[2]
                home_dir = bin_path.parents[3]
                subprocess_env.setdefault("WM_PROJECT", "OpenFOAM")
                subprocess_env.setdefault("WM_PROJECT_DIR", _shorten_path(str(wm_project_dir)))
                subprocess_env.setdefault("HOME", _shorten_path(str(home_dir)))
                subprocess_env.setdefault("FOAM_ETC", _shorten_path(str(wm_project_dir / "etc")))
        subprocess_cwd = _shorten_path(str(openfoam_case_dir))

        from bulbopt.infrastructure.adapters.mesh_check_parser import (
            parse_check_mesh_output,
        )

        for command in self.DEFAULT_SOLVER_CHAIN:
            resolved_command = self._resolve_command(command, short_bin)
            solver_name = command[0]
            stdout_log_path = solver_logs_dir / f"{solver_name}.log"
            stderr_log_path = solver_logs_dir / f"{solver_name}.err.log"
            try:
                completed = subprocess.run(
                    resolved_command,
                    cwd=subprocess_cwd,
                    env=subprocess_env,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
                stdout_text = completed.stdout or ""
                stderr_text = completed.stderr or ""
                stdout_log_path.write_text(stdout_text, encoding="utf-8")
                stderr_log_path.write_text(stderr_text, encoding="utf-8")
                step = {
                    "command": list(command),
                    "returncode": int(completed.returncode),
                    "stdout_tail": stdout_text[-2000:],
                    "stderr_tail": stderr_text[-2000:],
                    "stdout_log": str(stdout_log_path),
                    "stderr_log": str(stderr_log_path),
                }
                executed_steps.append(step)
                if completed.returncode != 0:
                    overall_returncode = completed.returncode
                    failing_step = command[0]
                    break
                # Fail the chain after checkMesh when illegal cells are
                # reported, even if the checkMesh process itself returned
                # zero — OpenFOAM tolerates illegal cells by convention
                # but simpleFoam cannot solve on them.
                if solver_name == "checkMesh":
                    report = parse_check_mesh_output(stdout_text)
                    step["check_mesh_report"] = {
                        "max_non_orthogonality": report["max_non_orthogonality"],
                        "max_skewness": report["max_skewness"],
                        "max_aspect_ratio": report["max_aspect_ratio"],
                        "n_illegal_cells": report["n_illegal_cells"],
                        "failed_checks": report["failed_checks"],
                        "mesh_ok": report["mesh_ok"],
                    }
                    if int(report["n_illegal_cells"]) > 0:
                        overall_returncode = 1
                        failing_step = "checkMesh"
                        break
                if solver_name == "simpleFoam":
                    step["solver_report"] = _parse_simple_foam_output(stdout_text)
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
                error_text = str(exc)
                stderr_log_path.write_text(error_text, encoding="utf-8")
                executed_steps.append(
                    {
                        "command": list(command),
                        "returncode": -1,
                        "error": error_text,
                        "stdout_log": str(stdout_log_path),
                        "stderr_log": str(stderr_log_path),
                    }
                )
                overall_returncode = -1
                failing_step = command[0]
                break

        status = "executed_ok" if overall_returncode == 0 and failing_step is None else "executed_failed"
        reason = (
            "solver_chain_completed"
            if status == "executed_ok"
            else f"solver_chain_failed_at_{failing_step or 'unknown'}"
        )
        result = {
            "runner_status": status,
            "runner_reason": reason,
            "runner_recoverable": True,
            "runner_case_directory": str(openfoam_case_dir),
            "status": status,
            "reason": reason,
            "is_recoverable": True,
            "best_candidate_id": manifest.get("best_candidate_id"),
            "executed_steps": executed_steps,
            "high_fidelity_used": status == "executed_ok",
        }
        run_manifest_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def _resolve_command(
        self,
        command: tuple[str, ...],
        bin_dir: str | None,
    ) -> list[str]:
        """Resolve the solver name to its absolute .exe path when a bin
        directory is configured.

        Windows ``CreateProcessW`` uses the parent process's ``%PATH%`` to
        look up bare command names, not the ``env`` passed to
        ``subprocess.run``. Passing the absolute path side-steps the quirk
        and lets the solver launch from any parent shell.
        """
        if not command:
            return list(command)
        head, *tail = command
        if bin_dir and not os.path.isabs(head):
            exe_name = head if head.lower().endswith(".exe") else f"{head}.exe"
            candidate = os.path.join(bin_dir, exe_name)
            if os.path.isfile(candidate):
                return [candidate, *tail]
        return list(command)

    def _read_case_manifest(self, openfoam_case_dir: Path) -> dict:
        manifest_path = openfoam_case_dir / "openfoam_case_manifest.json"
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def _to_wsl_path(self, path: Path) -> str | None:
        resolved = Path(path).resolve()
        drive = resolved.drive.rstrip(":").lower()
        if drive:
            parts = [part for part in resolved.parts[1:] if part not in {"\\", "/"}]
            return "/mnt/" + drive + "/" + "/".join(parts).replace("\\", "/")
        return str(resolved).replace("\\", "/")


def _parse_simple_foam_output(stdout_text: str) -> dict:
    residual_rows: list[dict] = []
    equations: list[str] = []
    for match in re.finditer(
        r"Solving for\s+([^,]+),\s+Initial residual\s+=\s+([0-9.eE+-]+),"
        r"\s+Final residual\s+=\s+([0-9.eE+-]+)",
        stdout_text,
    ):
        equation = match.group(1).strip()
        if equation not in equations:
            equations.append(equation)
        residual_rows.append(
            {
                "equation": equation,
                "initial": float(match.group(2)),
                "final": float(match.group(3)),
            }
        )

    time_values = [
        float(match.group(1))
        for match in re.finditer(r"^\s*Time\s+=\s+([0-9.eE+-]+)", stdout_text, re.M)
    ]
    completed = bool(re.search(r"^\s*End\s*$", stdout_text, re.M))
    return {
        "residuals_available": bool(residual_rows),
        "completed": completed,
        "last_time": time_values[-1] if time_values else None,
        "residual_equations": equations,
        "last_residuals": residual_rows[-len(equations):] if equations else [],
    }
