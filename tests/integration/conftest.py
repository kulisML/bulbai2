from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_real_wsl_openfoam_for_integration_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep integration tests deterministic on machines with WSL OpenFOAM."""
    monkeypatch.setenv("BULBOPT_DISABLE_WSL_OPENFOAM", "1")
