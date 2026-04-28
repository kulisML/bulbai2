from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_real_wsl_openfoam_for_infrastructure_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Infrastructure unit tests fake solver subprocesses; do not probe WSL."""
    monkeypatch.setenv("BULBOPT_DISABLE_WSL_OPENFOAM", "1")
