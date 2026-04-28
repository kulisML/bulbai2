"""Parser for OpenFOAM ``checkMesh`` log output.

Used by the night-run high-fidelity pipeline to catch bad snappyHexMesh
output BEFORE the (expensive) simpleFoam solve burns hours on a mesh
that would never converge. The parser extracts four headline metrics
from the textual log — everything else ``checkMesh`` reports is less
actionable for the optimisation loop:

* ``max_non_orthogonality`` — corner-angle deviation (deg). Robust
  simpleFoam cases keep this below ~65.
* ``max_skewness`` — largest cell skewness; > 4 is usually solver death.
* ``max_aspect_ratio`` — longest / shortest cell edge ratio.
* ``n_illegal_cells`` — sum of "cells with incorrect orientation",
  "zero or negative cells", and "cells with concave faces" lines. A
  strictly positive number means the mesh is unusable.

The parser is format-tolerant to handle OpenFOAM v9+, ESI, and
Foundation releases. Missing fields resolve to ``None`` so callers can
distinguish "field not reported" from "field reported as zero".
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


# Match floats possibly in scientific notation.
_NUM_RE = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

_MAX_NON_ORTHO_PATTERN = re.compile(
    r"Max non-orthogonality\s*=\s*(" + _NUM_RE + r")",
    re.IGNORECASE,
)
_MAX_SKEW_PATTERN = re.compile(
    r"Max skewness\s*=\s*(" + _NUM_RE + r")",
    re.IGNORECASE,
)
_MAX_ASPECT_PATTERN = re.compile(
    r"Max aspect ratio\s*=\s*(" + _NUM_RE + r")",
    re.IGNORECASE,
)
# ``*** Illegal cells found: N`` is the ESI convention; "faces in error"
# and "cells with incorrect orientation: N" are the Foundation wording.
_ILLEGAL_PATTERNS = (
    re.compile(r"\*+\s*([0-9]+)\s+illegal cells found", re.IGNORECASE),
    re.compile(r"Illegal cells found:\s*(\d+)", re.IGNORECASE),
    re.compile(r"Number of cells with incorrect orientation\s*:\s*(\d+)", re.IGNORECASE),
    re.compile(r"cells with zero or negative volume:\s*(\d+)", re.IGNORECASE),
    re.compile(r"cells with concave faces:\s*(\d+)", re.IGNORECASE),
)

_FAILED_CHECKS_PATTERN = re.compile(
    r"Failed\s+(\d+)\s+mesh checks", re.IGNORECASE,
)
_MESH_OK_PATTERN = re.compile(r"Mesh OK\b", re.IGNORECASE)


def parse_check_mesh_output(log_text: str) -> dict:
    """Extract the headline metrics from a checkMesh log blob.

    Returns a dict with keys:
    * ``max_non_orthogonality`` — float | None
    * ``max_skewness`` — float | None
    * ``max_aspect_ratio`` — float | None
    * ``n_illegal_cells`` — int (0 when none reported)
    * ``failed_checks`` — int | None (from the "Failed N mesh checks" line)
    * ``mesh_ok`` — bool (True iff the log contains a "Mesh OK" line)
    """
    text = log_text or ""
    return {
        "max_non_orthogonality": _first_float(text, _MAX_NON_ORTHO_PATTERN),
        "max_skewness": _first_float(text, _MAX_SKEW_PATTERN),
        "max_aspect_ratio": _first_float(text, _MAX_ASPECT_PATTERN),
        "n_illegal_cells": _sum_int_matches(text, _ILLEGAL_PATTERNS),
        "failed_checks": _first_int(text, _FAILED_CHECKS_PATTERN),
        "mesh_ok": bool(_MESH_OK_PATTERN.search(text)),
    }


def parse_check_mesh_log(log_path: Path) -> dict:
    """Read ``log_path`` and pass the contents through :func:`parse_check_mesh_output`.

    Missing files raise :class:`FileNotFoundError` — callers decide
    whether to treat that as an error or simply a skip signal.
    """
    path = Path(log_path)
    if not path.exists():
        raise FileNotFoundError(f"checkMesh log not found: {path}")
    return parse_check_mesh_output(path.read_text(encoding="utf-8", errors="replace"))


# ---- helpers --------------------------------------------------------------


def _first_float(text: str, pattern: re.Pattern[str]) -> Optional[float]:
    match = pattern.search(text)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def _first_int(text: str, pattern: re.Pattern[str]) -> Optional[int]:
    match = pattern.search(text)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def _sum_int_matches(text: str, patterns: tuple[re.Pattern[str], ...]) -> int:
    """Sum positive integer captures across every pattern that matches.

    checkMesh reports illegal cells under multiple headings; the
    interesting invariant is "> 0 ⇒ mesh unusable", so we just add them.
    """
    total = 0
    for pattern in patterns:
        for match in pattern.finditer(text):
            try:
                total += max(int(match.group(1)), 0)
            except (TypeError, ValueError):
                continue
    return int(total)


__all__ = [
    "parse_check_mesh_output",
    "parse_check_mesh_log",
]
