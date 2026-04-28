"""Multi-Froude support tests.

Spec: a candidate optimal at one Froude number may be poor at others.
``OpenFOAMAdapter.build_case`` must therefore accept an optional
``froude_number`` kwarg that scales the inlet velocity via
``Uref = froude_number * sqrt(g * lRef)``. With default args the case
must remain bit-for-bit identical (same ``0/U`` velocity 5.0) so the 310
existing tests keep passing.

``SimpleFoamHighFidelityGate.evaluate`` must accept a list of Froude
numbers (and matching weights) and aggregate the per-Fr Cd as a
weighted mean.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest
import trimesh

from bulbopt.infrastructure.adapters.openfoam_adapter import OpenFOAMAdapter
from bulbopt.infrastructure.adapters.simple_foam_gate import (
    SimpleFoamHighFidelityGate,
)
from bulbopt.optimization.parametric.ffd_deformer import BulbFFDDeformer
from bulbopt.optimization.parametric.kracht_space import KrachtDesignSpace


_VELOCITY_TUPLE_PATTERN = re.compile(
    r"internalField\s+uniform\s+\(([-+0-9eE.\s]+)\)"
)


def _parse_internal_velocity(u_text: str) -> tuple[float, float, float]:
    """Extract the (Ux, Uy, Uz) triple from a 0/U field text."""
    match = _VELOCITY_TUPLE_PATTERN.search(u_text)
    assert match is not None, f"could not find internalField in 0/U: {u_text!r}"
    parts = match.group(1).split()
    assert len(parts) == 3
    return float(parts[0]), float(parts[1]), float(parts[2])


def test_build_case_writes_velocity_for_given_froude(tmp_path: Path) -> None:
    """build_case(froude_number=0.30) must scale the 0/U inlet velocity to
    Uref = 0.30 * sqrt(g * lRef). The case template uses lRef = 10.0 m and
    g = 9.81 m/s^2, so the expected Ux is 0.30 * sqrt(9.81 * 10) ~= 2.971..."""
    case_dir = tmp_path / "case"
    geometry_path = case_dir / "candidate.stl"
    geometry_path.parent.mkdir(parents=True, exist_ok=True)
    geometry_path.write_text("solid demo\nendsolid demo\n", encoding="utf-8")

    adapter = OpenFOAMAdapter()
    adapter.build_case(
        case_dir,
        best_candidate_id="candidate-fr",
        best_candidate_geometry_path=geometry_path,
        froude_number=0.30,
    )

    u_path = case_dir / "working" / "openfoam_case" / "0" / "U"
    assert u_path.exists()
    u_text = u_path.read_text(encoding="utf-8")

    ux, uy, uz = _parse_internal_velocity(u_text)
    expected = 0.30 * math.sqrt(9.81 * 10.0)
    assert ux == pytest.approx(expected, rel=1e-9)
    assert uy == 0.0
    assert uz == 0.0


def test_build_case_default_velocity_unchanged(tmp_path: Path) -> None:
    """With no froude_number kwarg the adapter must produce the legacy
    inlet velocity (5.0 m/s along x) so existing 310 tests keep passing."""
    case_dir = tmp_path / "case"
    geometry_path = case_dir / "candidate.stl"
    geometry_path.parent.mkdir(parents=True, exist_ok=True)
    geometry_path.write_text("solid demo\nendsolid demo\n", encoding="utf-8")

    adapter = OpenFOAMAdapter()
    adapter.build_case(
        case_dir,
        best_candidate_id="candidate-default",
        best_candidate_geometry_path=geometry_path,
    )

    u_text = (case_dir / "working" / "openfoam_case" / "0" / "U").read_text(
        encoding="utf-8"
    )
    ux, uy, uz = _parse_internal_velocity(u_text)
    assert ux == pytest.approx(5.0, rel=1e-9)
    assert uy == 0.0
    assert uz == 0.0


def test_evaluate_aggregates_cd_across_froude_numbers(tmp_path: Path) -> None:
    """The gate must run simpleFoam once per Fr in ``froude_numbers`` and
    aggregate Cd as a weighted mean. With Cd=0.30 at Fr=0.20 and Cd=0.40
    at Fr=0.30 plus weights [0.4, 0.6], the aggregated Cd is 0.36."""

    # Each call to fake_run gets a different "Fr" via the build_case kwarg
    # the gate is expected to forward.
    fr_per_run: list[float | None] = []
    cd_by_fr = {0.20: 0.30, 0.30: 0.40}

    def fake_build(
        case_dir,
        *,
        best_candidate_id,
        best_candidate_geometry_path,
        froude_number=None,
    ):
        fr_per_run.append(froude_number)
        # Persist the Fr in the manifest so fake_run can find it back.
        return {
            "adapter": "openfoam",
            "case_directory": str(case_dir),
            "froude_number": froude_number,
        }

    def fake_run(case_dir, *, case_manifest, execute):
        # Materialise a forceCoeffs.dat under postProcessing so the gate's
        # parser produces the Cd we want for this Fr.
        fr = case_manifest.get("froude_number")
        cd = cd_by_fr[float(fr)]
        post_dir = Path(case_dir) / "postProcessing" / "forceCoeffs" / "0"
        post_dir.mkdir(parents=True, exist_ok=True)
        (post_dir / "forceCoeffs.dat").write_text(
            "# Time Cm Cd Cl Cl(f) Cl(r)\n"
            "0 0.0 0.0 0 0 0\n"
            f"200 0.0 {cd:.6f} 0 0 0\n",
            encoding="utf-8",
        )
        return {
            "status": "executed_ok",
            "is_recoverable": True,
            "high_fidelity_used": True,
        }

    baseline = trimesh.creation.box(extents=(4.0, 1.5, 1.0))
    extents = baseline.extents.astype(float)
    primary = int(np.argmax(extents))
    axis_values = baseline.vertices[:, primary]
    region = {
        "axis_index": primary,
        "axis_min": float(axis_values.min()) + float(extents[primary]) * 0.75,
        "axis_max": float(axis_values.max()),
    }

    gate = SimpleFoamHighFidelityGate(
        work_root=tmp_path,
        baseline_mesh=baseline,
        region=region,
        deformer=BulbFFDDeformer(),
        build_case=fake_build,
        run_case=fake_run,
    )

    vectors = KrachtDesignSpace().sample(n=1, seed=7)
    objectives = gate.evaluate(
        vectors,
        froude_numbers=[0.20, 0.30],
        froude_weights=[0.4, 0.6],
    )

    assert len(objectives) == 1
    aggregated_cd, _ = objectives[0]
    expected = 0.4 * 0.30 + 0.6 * 0.40  # 0.36
    assert aggregated_cd == pytest.approx(expected, rel=1e-9)
    # Both Froude numbers must have been forwarded to build_case.
    assert sorted(float(x) for x in fr_per_run) == [0.20, 0.30]
