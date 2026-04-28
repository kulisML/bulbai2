"""Tests for forceCoeffs coefficient.dat parser.

Design reference: 2026-04-22-bulbopt-night-optimization-design.md §9.

OpenFOAM's forces function object writes per-iteration drag / lift /
moment coefficients to
``postProcessing/forces/<time>/coefficient.dat``. The file starts with a
few '#' header lines naming the columns, followed by whitespace-separated
numeric rows, one per reported timestep.

The parser returns the last converged drag coefficient — that is what
the high-fidelity cascade gate feeds back into NSGA-II as the objective.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from bulbopt.infrastructure.adapters.force_coeffs_parser import (
    ForceCoeffsNotFoundError,
    parse_drag_coefficient_dat,
)


_SAMPLE_DAT = """\
# Forces coefficient output
# Column 1: Time
# Column 2: Cd
# Column 3: Cs
# Column 4: Cl
    0.0   1.2345   0.0   0.0
    1.0   0.9100   0.0   0.0
    5.0   0.8200   0.0   0.0
   10.0   0.7500   0.0   0.0
"""


def test_parser_reads_last_drag_coefficient(tmp_path: Path) -> None:
    dat = tmp_path / "coefficient.dat"
    dat.write_text(_SAMPLE_DAT, encoding="utf-8")

    result = parse_drag_coefficient_dat(dat)

    assert result["final_cd"] == pytest.approx(0.75)
    assert result["iterations"] == 4
    assert result["final_time"] == pytest.approx(10.0)


def test_parser_reads_openfoam13_force_coeffs_dat_cd_column(tmp_path: Path) -> None:
    dat = tmp_path / "forceCoeffs.dat"
    dat.write_text(
        "# Force coefficients\n"
        "# Time Cm Cd Cl Cl(f) Cl(r)\n"
        "0 0.001 0.410000 0.0 0.0 0.0\n"
        "200 0.002 0.324057128630 0.0 0.0 0.0\n",
        encoding="utf-8",
    )

    result = parse_drag_coefficient_dat(dat)

    assert result["final_cd"] == pytest.approx(0.324057128630)
    assert result["source_format"] == "openfoam13_forceCoeffs"


def test_parser_reports_stable_final_cd_window(tmp_path: Path) -> None:
    dat = tmp_path / "forceCoeffs.dat"
    dat.write_text(
        "# Time Cm Cd Cl Cl(f) Cl(r)\n"
        "0 0.0 0.5000 0 0 0\n"
        "100 0.0 0.3260 0 0 0\n"
        "200 0.0 0.3250 0 0 0\n"
        "300 0.0 0.3245 0 0 0\n"
        "400 0.0 0.3244 0 0 0\n",
        encoding="utf-8",
    )

    result = parse_drag_coefficient_dat(dat)

    assert result["cd_stability"]["stable"] is True
    assert result["cd_stability"]["window_size"] == 4
    assert result["cd_stability"]["max_delta"] == pytest.approx(0.0016)
    assert result["solver_convergence"]["status"] == "cd_stable"


def test_parser_reports_unstable_final_cd_window(tmp_path: Path) -> None:
    dat = tmp_path / "forceCoeffs.dat"
    dat.write_text(
        "# Time Cm Cd Cl Cl(f) Cl(r)\n"
        "0 0.0 0.5000 0 0 0\n"
        "100 0.0 0.3300 0 0 0\n"
        "200 0.0 0.3000 0 0 0\n"
        "300 0.0 0.3600 0 0 0\n"
        "400 0.0 0.3240 0 0 0\n",
        encoding="utf-8",
    )

    result = parse_drag_coefficient_dat(dat)

    assert result["cd_stability"]["stable"] is False
    assert result["cd_stability"]["max_delta"] == pytest.approx(0.06)
    assert result["solver_convergence"]["status"] == "cd_unstable"


def test_parser_ignores_comments_and_blank_lines(tmp_path: Path) -> None:
    dat = tmp_path / "coefficient.dat"
    dat.write_text(
        "# header\n"
        "\n"
        "   0 0.5 0 0\n"
        "\n"
        "# mid comment\n"
        "  10 0.3 0 0\n",
        encoding="utf-8",
    )

    result = parse_drag_coefficient_dat(dat)
    assert result["final_cd"] == pytest.approx(0.3)
    assert result["iterations"] == 2


def test_parser_raises_when_file_missing(tmp_path: Path) -> None:
    with pytest.raises(ForceCoeffsNotFoundError):
        parse_drag_coefficient_dat(tmp_path / "nope.dat")


def test_parser_raises_when_no_data_rows(tmp_path: Path) -> None:
    dat = tmp_path / "empty.dat"
    dat.write_text("# only header\n# nothing else\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no data rows"):
        parse_drag_coefficient_dat(dat)


def test_parser_converts_cd_to_drag_newtons_when_reference_supplied(
    tmp_path: Path,
) -> None:
    dat = tmp_path / "coefficient.dat"
    dat.write_text("  5.0  0.25  0  0\n", encoding="utf-8")

    result = parse_drag_coefficient_dat(
        dat,
        reference_velocity_m_s=10.0,
        reference_area_m2=2.0,
        fluid_density_kg_m3=1000.0,
    )

    # F = Cd * 0.5 * rho * V^2 * A = 0.25 * 0.5 * 1000 * 100 * 2 = 25000 N
    assert result["drag_newtons"] == pytest.approx(25000.0)


def test_parser_returns_none_drag_newtons_without_reference_params(
    tmp_path: Path,
) -> None:
    dat = tmp_path / "coefficient.dat"
    dat.write_text("  5.0  0.25  0  0\n", encoding="utf-8")

    result = parse_drag_coefficient_dat(dat)

    assert result.get("drag_newtons") is None
