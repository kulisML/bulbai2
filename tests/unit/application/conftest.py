"""Shared fixtures for application-layer tests.

The night-run use case persists HF history to a JSONL store. When tests
don't explicitly override ``history_path``, that default points at the
user's real home directory — we'd leak test data into the developer's
machine. This autouse fixture reroutes the default to a per-test
temporary file so tests stay hermetic.
"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_bulbopt_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``Path.home()`` -> ``tmp_path/home`` for the duration of
    each test so anything the use case writes under ``~/.bulbopt/`` lands
    inside the test's sandbox instead of polluting the dev's real home.
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir(parents=True, exist_ok=True)

    # Patch at Path level — works regardless of how the code obtains the
    # home path as long as it uses Path.home().
    original_home = Path.home
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
    try:
        yield fake_home
    finally:
        # monkeypatch undoes, but preserve type.
        _ = original_home


@pytest.fixture(autouse=True)
def _disable_real_openfoam_for_application_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Application unit tests must not launch WSL/OpenFOAM accidentally.

    Tests that need to assert OpenFOAM wiring can still monkeypatch this
    import-site function to ``True`` inside the test body.
    """
    monkeypatch.setenv("BULBOPT_DISABLE_WSL_OPENFOAM", "1")
    from bulbopt.application.use_cases import run_night_optimization

    monkeypatch.setattr(
        run_night_optimization,
        "detect_openfoam_available",
        lambda: False,
    )
