"""Froude sweep tests for :class:`StubEvaluationAdapter`.

The evaluator aggregates resistance across 5 realistic displacement-hull
Froude numbers (0.15 → 0.35) so ship scoring reflects the whole
operational envelope rather than a single design speed.
"""
from __future__ import annotations

import pytest

from bulbopt.infrastructure.adapters.stub_evaluation import (
    DEFAULT_FROUDE_SAMPLES,
    StubEvaluationAdapter,
)


def test_froude_sweep_has_five_samples_by_default():
    adapter = StubEvaluationAdapter()
    result = adapter._resistance_across_froude_sweep(
        resistance_proxy=1.0,
        geometry_metrics={"slenderness_ratio": 4.0},
        operational_profile_weights=None,
    )
    assert len(result["samples"]) == 5
    assert [s["froude"] for s in result["samples"]] == list(DEFAULT_FROUDE_SAMPLES)


def test_froude_sweep_produces_distinct_resistance_values():
    adapter = StubEvaluationAdapter()
    result = adapter._resistance_across_froude_sweep(
        resistance_proxy=0.8,
        geometry_metrics={"slenderness_ratio": 4.0},
        operational_profile_weights=None,
    )
    resistances = [s["resistance"] for s in result["samples"]]
    # 5 distinct samples → 5 distinct resistance values (non-monotonic is
    # fine; the wave-resistance hump peaks around Fr ≈ 0.3).
    assert len(set(resistances)) == 5


def test_froude_sweep_equal_weights_normalised_to_unit_sum():
    adapter = StubEvaluationAdapter()
    result = adapter._resistance_across_froude_sweep(
        resistance_proxy=1.0,
        geometry_metrics={"slenderness_ratio": 4.0},
        operational_profile_weights=None,
    )
    assert result["profile_source"] == "equal"
    assert sum(result["weights"]) == pytest.approx(1.0)
    assert all(w == pytest.approx(1.0 / 5) for w in result["weights"])


def test_froude_sweep_respects_user_profile_weights():
    adapter = StubEvaluationAdapter()
    result = adapter._resistance_across_froude_sweep(
        resistance_proxy=1.0,
        geometry_metrics={"slenderness_ratio": 4.0},
        operational_profile_weights=[0.0, 0.0, 1.0, 0.0, 0.0],
    )
    assert result["profile_source"] == "user_defined"
    # The aggregate resistance should equal exactly the middle-sample's
    # resistance when the weight vector picks only that sample.
    middle = result["samples"][2]
    assert result["aggregate_resistance"] == pytest.approx(middle["resistance"])


def test_froude_sweep_slenderness_reduces_peak_amplification():
    """A more slender hull should see a smaller resistance peak at Fr≈0.3."""
    adapter = StubEvaluationAdapter()
    stout = adapter._resistance_across_froude_sweep(
        resistance_proxy=1.0,
        geometry_metrics={"slenderness_ratio": 2.0},
        operational_profile_weights=None,
    )
    slender = adapter._resistance_across_froude_sweep(
        resistance_proxy=1.0,
        geometry_metrics={"slenderness_ratio": 8.0},
        operational_profile_weights=None,
    )
    peak_stout = max(s["resistance"] for s in stout["samples"])
    peak_slender = max(s["resistance"] for s in slender["samples"])
    assert peak_slender < peak_stout


def test_froude_sweep_custom_samples_respected():
    adapter = StubEvaluationAdapter()
    result = adapter._resistance_across_froude_sweep(
        resistance_proxy=1.0,
        geometry_metrics={"slenderness_ratio": 4.0},
        operational_profile_weights=None,
        froude_samples=(0.1, 0.2, 0.4),
    )
    assert len(result["samples"]) == 3
    assert [s["froude"] for s in result["samples"]] == [0.1, 0.2, 0.4]


def test_evaluate_candidates_embeds_froude_sweep_in_score_components(tmp_path):
    """Full end-to-end through evaluate_candidates: each candidate gets a
    ``froude_sweep_samples`` list in its score_components payload."""
    import trimesh

    adapter = StubEvaluationAdapter()
    # Fake candidate directory layout expected by the adapter.
    case_dir = tmp_path / "projects" / "case-foo"
    candidate_dir = case_dir / "working" / "candidates"
    repaired_dir = case_dir / "working" / "repaired"
    candidate_dir.mkdir(parents=True)
    repaired_dir.mkdir(parents=True)
    mesh = trimesh.creation.box(extents=(4.0, 1.5, 1.0))
    repaired_path = repaired_dir / "repaired.stl"
    repaired_path.write_bytes(trimesh.exchange.stl.export_stl(mesh))
    cand_path = candidate_dir / "candidate-1.stl"
    cand_path.write_bytes(trimesh.exchange.stl.export_stl(mesh))

    candidates = [
        {
            "candidate_id": "candidate-1",
            "geometry_path": str(cand_path),
            "bulb_region": {"axis_index": 0, "axis_min": 1.5, "axis_max": 2.0},
        }
    ]
    evaluated = adapter.evaluate_candidates(candidates=candidates, speed_knots=[18.0])
    assert len(evaluated) == 1
    score = evaluated[0]["score_components"]
    assert "froude_sweep_samples" in score
    assert len(score["froude_sweep_samples"]) == 5
    assert score["froude_sweep_aggregate_resistance"] > 0.0
