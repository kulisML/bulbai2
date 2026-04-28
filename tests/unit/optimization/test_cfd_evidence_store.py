from __future__ import annotations

from pathlib import Path

import pytest

from bulbopt.optimization.learning.cfd_evidence_store import CFDEvidenceStore


def _row(**overrides) -> dict:
    payload = {
        "schema_version": 1,
        "record_type": "candidate",
        "candidate_id": "candidate-001",
        "parameters": {
            "length_ratio": 0.031,
            "breadth_ratio": 0.085,
            "height_ratio": 0.30,
            "axis_z_ratio": 0.20,
            "longitudinal_pos": 0.60,
            "cross_section_c": 0.74,
            "volume_coef": 0.64,
            "nose_sharpness": 0.50,
        },
        "final_cd": 0.32,
        "baseline_cd": 0.40,
        "improvement_percent": 20.0,
        "engineering_valid": True,
        "hull_fingerprint": "hull-a",
        "settings_hash": "settings-a",
        "geometry": {
            "stl_report": {"checks_passed": True},
            "parameter_warnings": [],
            "constraint_violations": [],
            "geometry_risk": "low",
            "manufacturability_risk": "clear",
        },
    }
    payload.update(overrides)
    return payload


def test_cfd_evidence_store_filters_warm_start_by_compatibility(
    tmp_path: Path,
) -> None:
    store = CFDEvidenceStore(tmp_path / "cfd_evidence.jsonl")
    store.append_many(
        [
            _row(candidate_id="compatible", final_cd=0.34),
            _row(candidate_id="wrong-hull", hull_fingerprint="hull-b", final_cd=0.30),
            _row(candidate_id="wrong-settings", settings_hash="settings-b", final_cd=0.31),
        ]
    )

    vectors = store.top_k_safe_warm_start(
        5,
        hull_fingerprint="hull-a",
        settings_hash="settings-a",
    )
    summary = store.warm_start_eligibility_summary(
        hull_fingerprint="hull-a",
        settings_hash="settings-a",
    )

    assert len(vectors) == 1
    assert vectors[0].values["length_ratio"] == pytest.approx(0.031)
    assert summary == {
        "candidate_rows": 3,
        "eligible": 1,
        "hull_mismatch": 1,
        "settings_mismatch": 1,
        "not_engineering_valid": 0,
        "geometry_high_risk": 0,
        "manufacturability_warning": 0,
        "not_improving": 0,
        "missing_final_cd": 0,
        "missing_parameters": 0,
    }


def test_cfd_evidence_store_filters_surrogate_training_by_compatibility(
    tmp_path: Path,
) -> None:
    store = CFDEvidenceStore(tmp_path / "cfd_evidence.jsonl")
    store.append_many(
        [
            _row(candidate_id="compatible", final_cd=0.34),
            _row(candidate_id="wrong-hull", hull_fingerprint="hull-b", final_cd=0.30),
            _row(candidate_id="penalty", final_cd=1e9),
            _row(candidate_id="invalid", engineering_valid=False, final_cd=0.25),
        ]
    )

    pairs = store.surrogate_training_pairs(
        hull_fingerprint="hull-a",
        settings_hash="settings-a",
    )

    assert len(pairs) == 1
    vector, cd = pairs[0]
    assert vector.values["length_ratio"] == pytest.approx(0.031)
    assert cd == pytest.approx(0.34)


def test_cfd_evidence_store_lists_best_compatible_rows_by_cd(
    tmp_path: Path,
) -> None:
    store = CFDEvidenceStore(tmp_path / "cfd_evidence.jsonl")
    store.append_many(
        [
            _row(candidate_id="slower", final_cd=0.36),
            _row(candidate_id="best", final_cd=0.31),
            _row(candidate_id="wrong-hull", hull_fingerprint="hull-b", final_cd=0.20),
        ]
    )

    rows = store.best_candidate_rows(
        1,
        hull_fingerprint="hull-a",
        settings_hash="settings-a",
    )

    assert [row["candidate_id"] for row in rows] == ["best"]


def test_cfd_evidence_store_excludes_geometry_risk_from_warm_start(
    tmp_path: Path,
) -> None:
    store = CFDEvidenceStore(tmp_path / "cfd_evidence.jsonl")
    store.append_many(
        [
            _row(candidate_id="safe", final_cd=0.34),
            _row(
                candidate_id="bad-stl",
                final_cd=0.30,
                geometry={
                    "stl_report": {"checks_passed": False},
                    "parameter_warnings": [],
                    "constraint_violations": [],
                    "geometry_risk": "high",
                    "manufacturability_risk": "clear",
                },
            ),
            _row(
                candidate_id="near-bound",
                final_cd=0.31,
                geometry={
                    "stl_report": {"checks_passed": True},
                    "parameter_warnings": ["sharp_full_section_near_limit"],
                    "constraint_violations": [],
                    "geometry_risk": "medium",
                    "manufacturability_risk": "warning",
                },
            ),
        ]
    )

    rows = store.best_candidate_rows(
        5,
        hull_fingerprint="hull-a",
        settings_hash="settings-a",
    )
    summary = store.warm_start_eligibility_summary(
        hull_fingerprint="hull-a",
        settings_hash="settings-a",
    )

    assert [row["candidate_id"] for row in rows] == ["safe"]
    assert summary["candidate_rows"] == 3
    assert summary["eligible"] == 1
    assert summary["geometry_high_risk"] == 1
    assert summary["manufacturability_warning"] == 1
