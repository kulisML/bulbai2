"""Unit tests for :mod:`mesh_check_parser`."""
from __future__ import annotations

from pathlib import Path

import pytest

from bulbopt.infrastructure.adapters.mesh_check_parser import (
    parse_check_mesh_log,
    parse_check_mesh_output,
)


# A canonical checkMesh log excerpt — representative of OpenFOAM v9+ /
# ESI releases. Kept deliberately short so the regex tolerance is tested
# against real wording rather than our own paraphrase.
HEALTHY_LOG = """\
/*---------------------------------------------------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  9                                     |
|   \\\\  /    A nd           | Website:  https://openfoam.org                  |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
Create polyMesh for time = 0

Time = 0

Mesh stats
    points:           123456
    faces:            345678
    internal faces:   340000

Checking geometry...
    Overall domain bounding box (-5 -2 -1.2) (20 2 1.2)
    Mesh non-orthogonality Max: 43.7 average: 12.3
    Mesh has non-orthogonality Max non-orthogonality = 43.7
    Max skewness = 1.82 OK.
    Max aspect ratio = 5.4 OK.
    Minimum face area = 1e-05. Maximum face area = 0.02.  Face area magnitudes OK.
    All angles in faces are within 89.9 degrees
    Mesh OK.

End
"""

BAD_LOG = """\
Mesh stats
    points:           45
    faces:            80
    internal faces:   60

Checking geometry...
    Mesh non-orthogonality Max non-orthogonality = 88.4 average: 31.0
    Max skewness = 7.9 (threshold 4)
    Max aspect ratio = 123.4 FAILED.
    ***Number of cells with incorrect orientation: 3
    cells with zero or negative volume: 2
    cells with concave faces: 1

    Failed 3 mesh checks.

End
"""


def test_parse_healthy_log_extracts_headline_metrics():
    result = parse_check_mesh_output(HEALTHY_LOG)
    assert result["max_non_orthogonality"] == pytest.approx(43.7)
    assert result["max_skewness"] == pytest.approx(1.82)
    assert result["max_aspect_ratio"] == pytest.approx(5.4)
    assert result["n_illegal_cells"] == 0
    assert result["mesh_ok"] is True


def test_parse_bad_log_sums_illegal_cells():
    result = parse_check_mesh_output(BAD_LOG)
    # 3 (incorrect orientation) + 2 (zero volume) + 1 (concave) = 6 illegal cells.
    assert result["n_illegal_cells"] == 6
    assert result["max_non_orthogonality"] == pytest.approx(88.4)
    assert result["max_skewness"] == pytest.approx(7.9)
    assert result["failed_checks"] == 3
    assert result["mesh_ok"] is False


def test_parse_empty_log_is_all_none_or_zero():
    result = parse_check_mesh_output("")
    assert result["max_non_orthogonality"] is None
    assert result["max_skewness"] is None
    assert result["max_aspect_ratio"] is None
    assert result["n_illegal_cells"] == 0
    assert result["mesh_ok"] is False
    assert result["failed_checks"] is None


def test_parse_check_mesh_log_from_path(tmp_path: Path):
    log_path = tmp_path / "log.checkMesh"
    log_path.write_text(HEALTHY_LOG, encoding="utf-8")
    result = parse_check_mesh_log(log_path)
    assert result["max_non_orthogonality"] == pytest.approx(43.7)
    assert result["mesh_ok"] is True


def test_parse_missing_log_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        parse_check_mesh_log(tmp_path / "does_not_exist.log")


def test_parse_scientific_notation_values():
    log = """\
Max non-orthogonality = 4.37e+01 average: 1.2e+01
Max skewness = 1.82E+00 OK.
Max aspect ratio = 5.4e0 OK.
"""
    result = parse_check_mesh_output(log)
    assert result["max_non_orthogonality"] == pytest.approx(43.7)
    assert result["max_skewness"] == pytest.approx(1.82)
    assert result["max_aspect_ratio"] == pytest.approx(5.4)


def test_parse_esi_illegal_cells_convention():
    log = """\
*** 5 illegal cells found.
"""
    result = parse_check_mesh_output(log)
    assert result["n_illegal_cells"] == 5
