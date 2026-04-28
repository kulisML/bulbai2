from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bulbopt.infrastructure.adapters.simple_foam_gate import _read_force_coeffs
from bulbopt.infrastructure.adapters.simple_foam_gate import SimpleFoamHighFidelityGate
from bulbopt.optimization.parametric.ffd_deformer import BulbFFDDeformer
from bulbopt.optimization.parametric.kracht_space import KrachtVector

import trimesh


def test_read_force_coeffs_finds_openfoam13_force_coeffs_dat(tmp_path: Path) -> None:
    post_dir = tmp_path / "postProcessing" / "forceCoeffs" / "0"
    post_dir.mkdir(parents=True)
    (post_dir / "forceCoeffs.dat").write_text(
        "# Time Cm Cd Cl Cl(f) Cl(r)\n"
        "0 0.0 0.5 0 0 0\n"
        "200 0.0 0.324057128630 0 0 0\n",
        encoding="utf-8",
    )

    result = _read_force_coeffs(
        tmp_path,
        reference_velocity=None,
        reference_area=None,
        fluid_density=None,
    )

    assert result is not None
    assert result["final_cd"] == pytest.approx(0.324057128630)


def test_simple_foam_gate_penalizes_constraint_violation_before_building_case(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def build_case(*args, **kwargs):
        calls.append("build")
        return {}

    def run_case(*args, **kwargs):
        calls.append("run")
        return {"status": "executed_ok"}

    gate = SimpleFoamHighFidelityGate(
        work_root=tmp_path,
        baseline_mesh=trimesh.creation.box(extents=(4.0, 1.5, 1.0)),
        region={"axis_index": 0, "axis_min": 0.0, "axis_max": 2.0},
        deformer=BulbFFDDeformer(),
        build_case=build_case,
        run_case=run_case,
    )
    risky_vector = KrachtVector(
        values={
            "length_ratio": 0.044,
            "breadth_ratio": 0.016,
            "height_ratio": 0.54,
            "axis_z_ratio": 0.25,
            "longitudinal_pos": 0.75,
            "cross_section_c": 0.999,
            "volume_coef": 0.89,
            "nose_sharpness": 0.055,
        }
    )

    objectives = gate.evaluate([risky_vector])

    assert calls == []
    assert objectives == [[1e9, 1e9]]


def test_simple_foam_gate_skips_cfd_when_deformed_stl_is_invalid(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    class BrokenDeformer:
        def deform(self, mesh, region, vector):
            vertices = np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            )
            faces = np.array([[0, 1, 2]])
            return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    def build_case(*args, **kwargs):
        calls.append("build")
        return {}

    def run_case(*args, **kwargs):
        calls.append("run")
        return {"status": "executed_ok"}

    gate = SimpleFoamHighFidelityGate(
        work_root=tmp_path,
        baseline_mesh=trimesh.creation.box(extents=(4.0, 1.5, 1.0)),
        region={"axis_index": 0, "axis_min": 0.0, "axis_max": 2.0},
        deformer=BrokenDeformer(),
        build_case=build_case,
        run_case=run_case,
    )
    vector = KrachtVector(
        values={
            "length_ratio": 0.020,
            "breadth_ratio": 0.080,
            "height_ratio": 0.30,
            "axis_z_ratio": 0.20,
            "longitudinal_pos": 0.60,
            "cross_section_c": 0.75,
            "volume_coef": 0.65,
            "nose_sharpness": 0.50,
        }
    )

    objectives = gate.evaluate([vector])

    assert calls == []
    assert objectives == [[1e9, 1e9]]
    assert gate.evaluation_records[0]["solver_status"] == "skipped"
    assert gate.evaluation_records[0]["solver_reason"] == "stl_invalid"
    assert "non_positive_volume" in gate.evaluation_records[0]["stl_report"]["failure_reasons"]
