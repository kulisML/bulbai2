"""Parser for OpenFOAM ``forces`` / ``forceCoeffs`` coefficient.dat files.

Design reference: 2026-04-22-bulbopt-night-optimization-design.md §9.

The forceCoeffs function object writes one row per reported timestep:

    # Forces coefficient output
    # Column 1: Time
    # Column 2: Cd
    # Column 3: Cs
    # Column 4: Cl
    0.0  1.2345  0.0  0.0
    1.0  0.9100  0.0  0.0
    ...

The parser:

* skips comment (``#``) and blank lines,
* extracts the last numeric row's Cd (column 2) as the converged drag
  coefficient (assumes the solver wrote the final converged state),
* optionally converts Cd to a drag force in Newtons when the caller
  supplies a reference velocity, area, and fluid density:
      F_drag = Cd * 0.5 * rho * V^2 * A
  so the cascade can feed Newtons straight into the GA objective.

The parser returns a plain dict so the result serialises to JSON without
transformation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


class ForceCoeffsNotFoundError(FileNotFoundError):
    """Raised when the expected coefficient.dat does not exist."""


def parse_drag_coefficient_dat(
    path: Path,
    *,
    reference_velocity_m_s: Optional[float] = None,
    reference_area_m2: Optional[float] = None,
    fluid_density_kg_m3: Optional[float] = None,
) -> dict:
    """Return final drag info from an OpenFOAM forceCoeffs output file."""
    path = Path(path)
    if not path.exists():
        raise ForceCoeffsNotFoundError(str(path))

    source_format = "legacy_coefficient"
    cd_index = 1
    rows: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                header = stripped.lower()
                if "time" in header and "cm" in header and "cd" in header:
                    source_format = "openfoam13_forceCoeffs"
                    cd_index = 2
                continue
            parts = stripped.split()
            if len(parts) <= cd_index:
                continue
            try:
                time_s = float(parts[0])
                cd = float(parts[cd_index])
            except ValueError:
                continue
            rows.append((time_s, cd))

    if not rows:
        raise ValueError(
            f"forceCoeffs file has no data rows: {path}"
        )

    final_time, final_cd = rows[-1]
    cd_stability = _cd_stability(rows)
    result: dict = {
        "final_cd": final_cd,
        "iterations": len(rows),
        "final_time": final_time,
        "source_format": source_format,
        "cd_stability": cd_stability,
        "solver_convergence": {
            "status": (
                "cd_stable"
                if cd_stability["stable"]
                else (
                    "cd_unstable"
                    if cd_stability["window_size"] >= 3
                    else "cd_insufficient_samples"
                )
            ),
            "cd_stable": cd_stability["stable"],
            "residuals_available": None,
        },
    }

    if (
        reference_velocity_m_s is not None
        and reference_area_m2 is not None
        and fluid_density_kg_m3 is not None
    ):
        drag_newtons = (
            final_cd
            * 0.5
            * float(fluid_density_kg_m3)
            * float(reference_velocity_m_s) ** 2
            * float(reference_area_m2)
        )
        result["drag_newtons"] = drag_newtons
    else:
        result["drag_newtons"] = None

    return result


def _cd_stability(rows: list[tuple[float, float]]) -> dict:
    window = rows[-4:]
    values = [float(cd) for _time, cd in window]
    if not values:
        return {
            "stable": False,
            "window_size": 0,
            "max_delta": None,
            "relative_delta": None,
        }

    max_delta = max(values) - min(values)
    final_abs = abs(values[-1])
    relative_delta = max_delta / max(final_abs, 1e-12)
    return {
        "stable": len(values) >= 3 and relative_delta <= 0.01,
        "window_size": len(values),
        "max_delta": float(max_delta),
        "relative_delta": float(relative_delta),
    }
