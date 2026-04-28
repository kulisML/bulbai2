"""Tests for Kracht-style bulb parametric space.

Design reference: 2026-04-22-bulbopt-night-optimization-design.md §4.
The space has 8 bounded continuous variables; the GA samples them, and
downstream FFD maps each sample into a concrete mesh deformation.
"""
from __future__ import annotations

import pytest

from bulbopt.optimization.parametric.kracht_space import (
    KRACHT_PARAMETER_NAMES,
    TINY_SHARP_KRACHT_VECTOR,
    KrachtDesignSpace,
    KrachtVector,
)


def test_kracht_design_space_exposes_eight_parameters() -> None:
    """Spec §4.1 defines exactly 8 parameters; any drift is a bug."""
    space = KrachtDesignSpace()
    assert len(space.bounds) == 8
    assert list(space.bounds.keys()) == list(KRACHT_PARAMETER_NAMES)


def test_kracht_design_space_bounds_match_spec_ranges() -> None:
    """Exact bounds from design spec §4.1 table.

    Audit 2026-04-26 (Add #1): ``volume_coef`` is widened to allow
    negative values so the optimizer can subtract bulb volume when the
    baseline already has too much. ``breadth_ratio`` and
    ``height_ratio`` remain non-negative because they are *fractions of
    box size*, not signed deltas — flipping their sign is a behavioural
    change for the FFD that the audit explicitly reserves for a future
    iteration.
    """
    space = KrachtDesignSpace()
    expected = {
        "length_ratio":     (0.010, 0.045),
        "breadth_ratio":    (0.015, 0.200),
        "height_ratio":     (0.100, 0.650),
        "axis_z_ratio":     (0.050, 0.500),
        "longitudinal_pos": (0.000, 1.000),
        "cross_section_c":  (0.250, 1.000),
        "volume_coef":      (-0.400, 0.900),
        "nose_sharpness":   (0.050, 1.000),
    }
    for name, (lo, hi) in expected.items():
        lo_got, hi_got = space.bounds[name]
        assert lo_got == pytest.approx(lo), f"{name} lower"
        assert hi_got == pytest.approx(hi), f"{name} upper"


def test_kracht_design_space_sample_is_deterministic_with_seed() -> None:
    """Reproducibility: same seed → same batch."""
    space = KrachtDesignSpace()
    batch_a = space.sample(n=5, seed=42)
    batch_b = space.sample(n=5, seed=42)
    assert len(batch_a) == 5
    assert batch_a == batch_b


def test_kracht_design_space_sample_respects_bounds() -> None:
    """Every sampled value must lie within its declared range."""
    space = KrachtDesignSpace()
    batch = space.sample(n=50, seed=1)
    for vector in batch:
        for name, value in vector.values.items():
            lo, hi = space.bounds[name]
            assert lo <= value <= hi, f"{name}={value} out of [{lo}, {hi}]"


def test_kracht_design_space_different_seeds_give_different_samples() -> None:
    space = KrachtDesignSpace()
    batch_a = space.sample(n=5, seed=1)
    batch_b = space.sample(n=5, seed=2)
    assert batch_a != batch_b


def test_kracht_design_space_validate_accepts_valid_vector() -> None:
    space = KrachtDesignSpace()
    vector = KrachtVector(
        values={
            "length_ratio":     0.02,
            "breadth_ratio":    0.08,
            "height_ratio":     0.30,
            "axis_z_ratio":     0.20,
            "longitudinal_pos": 0.60,
            "cross_section_c":  0.75,
            "volume_coef":      0.65,
            "nose_sharpness":   0.50,
        }
    )
    assert space.validate(vector) is True


def test_kracht_design_space_validate_rejects_out_of_bounds() -> None:
    space = KrachtDesignSpace()
    # length_ratio above its upper bound (0.045) → invalid
    vector = KrachtVector(
        values={
            "length_ratio":     0.100,  # TOO HIGH
            "breadth_ratio":    0.08,
            "height_ratio":     0.30,
            "axis_z_ratio":     0.20,
            "longitudinal_pos": 0.60,
            "cross_section_c":  0.75,
            "volume_coef":      0.65,
            "nose_sharpness":   0.50,
        }
    )
    assert space.validate(vector) is False


def test_kracht_design_space_rejects_near_zero_nose_sharpness() -> None:
    space = KrachtDesignSpace()
    vector = KrachtVector(
        values={
            "length_ratio":     0.02,
            "breadth_ratio":    0.08,
            "height_ratio":     0.30,
            "axis_z_ratio":     0.20,
            "longitudinal_pos": 0.60,
            "cross_section_c":  0.75,
            "volume_coef":      0.65,
            "nose_sharpness":   0.0063,
        }
    )

    assert space.validate(vector) is False
    assert "nose_sharpness_below_min" in space.constraint_violations(vector)


def test_kracht_design_space_rejects_full_section_with_too_sharp_nose() -> None:
    space = KrachtDesignSpace()
    vector = KrachtVector(
        values={
            "length_ratio":     0.044,
            "breadth_ratio":    0.016,
            "height_ratio":     0.54,
            "axis_z_ratio":     0.25,
            "longitudinal_pos": 0.75,
            "cross_section_c":  0.999,
            "volume_coef":      0.89,
            "nose_sharpness":   0.055,
        }
    )

    assert space.validate(vector) is False
    assert "full_section_with_sharp_nose" in space.constraint_violations(vector)


def test_kracht_design_space_warns_on_near_bound_manufacturability_risk() -> None:
    space = KrachtDesignSpace()
    vector = KrachtVector(
        values={
            "length_ratio":     0.044,
            "breadth_ratio":    0.08,
            "height_ratio":     0.54,
            "axis_z_ratio":     0.49,
            "longitudinal_pos": 0.84,
            "cross_section_c":  0.96,
            "volume_coef":      0.87,
            "nose_sharpness":   0.09,
        }
    )

    assert space.validate(vector) is True
    assert space.constraint_violations(vector) == []
    warnings = space.manufacturability_warnings(vector)
    assert "nose_sharpness_near_lower_bound" in warnings
    assert "cross_section_c_near_upper_bound" in warnings
    assert "sharp_full_section_near_limit" in warnings
    assert "aft_high_bulb_near_limit" in warnings


def test_kracht_design_space_validate_rejects_missing_parameter() -> None:
    space = KrachtDesignSpace()
    vector = KrachtVector(values={"length_ratio": 0.02})  # missing the other 7
    assert space.validate(vector) is False


def test_kracht_vector_to_array_and_back_roundtrip() -> None:
    """NSGA-II operates on numpy arrays; the space must convert both ways."""
    space = KrachtDesignSpace()
    vector = space.sample(n=1, seed=7)[0]
    array = space.to_array(vector)
    assert len(array) == 8
    restored = space.from_array(list(array))
    assert restored.values == vector.values


def test_kracht_design_space_to_array_preserves_declared_order() -> None:
    """GA array index must map to parameter name deterministically."""
    space = KrachtDesignSpace()
    vector = KrachtVector(
        values={
            "length_ratio":     0.015,
            "breadth_ratio":    0.020,
            "height_ratio":     0.150,
            "axis_z_ratio":     0.100,
            "longitudinal_pos": 0.500,
            "cross_section_c":  0.500,
            "volume_coef":      0.500,
            "nose_sharpness":   0.500,
        }
    )
    array = space.to_array(vector)
    assert list(array) == [0.015, 0.020, 0.150, 0.100, 0.500, 0.500, 0.500, 0.500]


# Audit 2026-04-26 — Add #1: negative volume_coef enables the optimizer to
# subtract bulb volume from a baseline that already has too much. The new
# bounds must still be exercised by ``sample`` and the new "deflate the
# bulb into a sharp nose" pathology must be rejected by
# ``constraint_violations``.


def test_volume_coef_can_be_negative() -> None:
    """Sample 100 vectors and verify the widened ``volume_coef`` bounds.

    With the new lower bound ``-0.40`` at least one of 100 LHS-style
    uniform samples should land on the negative side; all samples must
    stay inside ``[-0.40, 0.90]``.
    """
    space = KrachtDesignSpace()
    batch = space.sample(n=100, seed=12345)
    coefs = [vector.values["volume_coef"] for vector in batch]
    assert any(c < 0.0 for c in coefs), (
        "expected at least one negative volume_coef sample with n=100"
    )
    assert min(coefs) >= -0.40 - 1e-9
    assert max(coefs) <= 0.90 + 1e-9


def test_negative_volume_coef_with_sharp_nose_is_rejected_by_constraint() -> None:
    """Add #1 coupled constraint: ``volume_coef < -0.30`` AND
    ``nose_sharpness < 0.10`` is unphysical (the FFD would invert the
    nose tip into the body). The constraint must reject that pair, and
    must accept the same negative ``volume_coef`` paired with a healthy
    ``nose_sharpness``.
    """
    space = KrachtDesignSpace()
    bad = KrachtVector(
        values={
            "length_ratio":     0.02,
            "breadth_ratio":    0.08,
            "height_ratio":     0.30,
            "axis_z_ratio":     0.20,
            "longitudinal_pos": 0.60,
            "cross_section_c":  0.75,
            "volume_coef":      -0.35,
            "nose_sharpness":   0.05,
        }
    )
    violations = space.constraint_violations(bad)
    assert violations, (
        "expected coupled constraint to reject volume_coef=-0.35 with "
        f"nose_sharpness=0.05; got {violations!r}"
    )
    assert "deflate_with_sharp_nose" in violations or any(
        "deflate" in v for v in violations
    )

    good = KrachtVector(
        values={
            "length_ratio":     0.02,
            "breadth_ratio":    0.08,
            "height_ratio":     0.30,
            "axis_z_ratio":     0.20,
            "longitudinal_pos": 0.60,
            "cross_section_c":  0.75,
            "volume_coef":      -0.35,
            "nose_sharpness":   0.5,
        }
    )
    assert space.validate(good) is True
    assert space.constraint_violations(good) == []


# Audit 2026-04-26 — Module C: tightened bounds biased toward the
# empirically-validated tiny/sharp Kracht vector that drove a -2.48% Cd
# improvement on docs/base_hull.stl. The default bounds remain unchanged;
# ``KrachtDesignSpace.tightened()`` is a purely additive factory that
# returns a new space with bounds focussed on the known-good region.


def test_tightened_bounds_drops_known_bad_regions() -> None:
    """``tightened()`` removes the wide-bulb / round-dome / heavy-inflate
    regions that empirically produce candidates 2x WORSE than baseline.
    """
    space = KrachtDesignSpace.tightened()
    assert space.bounds["length_ratio"][1] <= 0.030
    assert space.bounds["nose_sharpness"][0] >= 0.5
    assert space.bounds["volume_coef"][1] <= 0.40

    # Sampling must respect the tightened bounds — never produce
    # length_ratio > 0.030.
    batch = space.sample(n=200, seed=2026)
    for vector in batch:
        assert vector.values["length_ratio"] <= 0.030 + 1e-9


def test_tiny_sharp_constant_is_within_tightened_bounds() -> None:
    """The empirically-validated -2.48% Cd vector must lie inside the
    tightened bounds (otherwise the tightened factory would exclude it).
    """
    space = KrachtDesignSpace.tightened()
    for name in KRACHT_PARAMETER_NAMES:
        lo, hi = space.bounds[name]
        value = TINY_SHARP_KRACHT_VECTOR.values[name]
        assert lo <= value <= hi, (
            f"{name}={value} not within tightened bounds [{lo}, {hi}]"
        )
