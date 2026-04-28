"""Kracht-style bulb parametric space.

Design reference: docs/superpowers/specs/2026-04-22-bulbopt-night-optimization-design.md §4.1.

Eight continuous design variables with physical meaning, bounded by values
adapted from Kracht (1978) and Hoekstra/Raven (2011). The space provides
three services for the optimization layer:

1. ``sample(n, seed)`` — Latin-hypercube-style uniform sampling inside the
   bounds. Deterministic under a fixed seed so NSGA-II runs are reproducible.
2. ``validate(vector)`` — binary check that every parameter is present and
   inside its declared range.
3. ``to_array`` / ``from_array`` — canonical conversion between the dict
   form (human-readable, JSON-friendly) and the numpy-friendly float vector
   used by pymoo operators.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Tuple


# Declared in the order the GA array uses. Do not reorder without updating
# existing case payloads — this order is part of the public schema.
KRACHT_PARAMETER_NAMES: Tuple[str, ...] = (
    "length_ratio",
    "breadth_ratio",
    "height_ratio",
    "axis_z_ratio",
    "longitudinal_pos",
    "cross_section_c",
    "volume_coef",
    "nose_sharpness",
)


_DEFAULT_BOUNDS: Dict[str, Tuple[float, float]] = {
    "length_ratio":     (0.010, 0.045),
    "breadth_ratio":    (0.015, 0.200),
    "height_ratio":     (0.100, 0.650),
    "axis_z_ratio":     (0.050, 0.500),
    "longitudinal_pos": (0.000, 1.000),
    "cross_section_c":  (0.250, 1.000),
    # Audit 2026-04-26 (Add #1): widened lower bound from +0.40 to -0.40
    # so the optimizer can subtract bulb volume when the baseline hull
    # already has too much. The FFD's offset multiplier is
    # ``0.5 + volume_coef``, so values in ``[-0.40, 0.0)`` produce a
    # smaller forward push than the previous minimum (multiplier 0.9).
    # ``breadth_ratio`` and ``height_ratio`` stay non-negative because
    # they're scaling fractions (not signed deltas) — see
    # ``_kracht_to_lattice_offsets`` in ffd_deformer.py: a sign flip
    # there is a behavioural change reserved for a future iteration.
    "volume_coef":      (-0.400, 0.900),
    "nose_sharpness":   (0.050, 1.000),
}


# Audit 2026-04-26 — Module C: bounds biased toward the empirically-
# validated tiny/sharp Kracht region. A real CFD run on
# ``docs/base_hull.stl`` showed the ``TINY_SHARP_KRACHT_VECTOR`` below
# reduces Cd by 2.48 %, while the default Kracht bounds also include
# large bulb regions that produce candidates 2x WORSE than baseline.
# Tightening the search space toward the known-good region dramatically
# improves sample efficiency for the night-optimization run.
#
# Only four dimensions are overridden; the rest inherit
# ``_DEFAULT_BOUNDS`` so any future change to those defaults flows
# through automatically.
_TIGHTENED_BOUND_OVERRIDES: Dict[str, Tuple[float, float]] = {
    "length_ratio":   (0.005, 0.030),  # was (0.010, 0.045) — drop wide-bulb upper end
    "volume_coef":    (-0.400, 0.400),  # was (-0.400, 0.900) — drop heavy-inflate end
    "nose_sharpness": (0.500, 1.000),   # was (0.050, 1.000) — drop round-dome end
    # ``breadth_ratio``, ``height_ratio``, ``axis_z_ratio``,
    # ``longitudinal_pos``, ``cross_section_c`` inherit defaults.
}


@dataclass(slots=True, frozen=True)
class KrachtVector:
    """One concrete 8-parameter sample.

    ``values`` must contain every KRACHT_PARAMETER_NAMES key. The dataclass
    is frozen so a vector can safely be cached / checkpointed.
    """

    values: Mapping[str, float]


@dataclass(slots=True)
class KrachtDesignSpace:
    """Bounded 8-D design space.

    Bounds default to the values from the design spec §4.1; callers may
    override them (e.g. tighter ranges for a local_optimize refinement).
    """

    bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: dict(_DEFAULT_BOUNDS))

    @classmethod
    def tightened(cls) -> "KrachtDesignSpace":
        """Return a space with bounds biased toward the known-good region.

        The tightened bounds drop the wide-bulb upper end of
        ``length_ratio``, the heavy-inflate end of ``volume_coef``, and
        the round-dome end of ``nose_sharpness``. The remaining
        dimensions inherit defaults — see ``_TIGHTENED_BOUND_OVERRIDES``
        for the audit rationale.
        """
        merged = dict(_DEFAULT_BOUNDS)
        merged.update(_TIGHTENED_BOUND_OVERRIDES)
        return cls(bounds=merged)

    def sample(self, n: int, seed: int | None = None) -> List[KrachtVector]:
        """Uniform random sample of ``n`` vectors.

        We use :mod:`random` rather than :mod:`numpy.random` so the outputs
        serialise directly to plain floats without numpy wrapping the values
        (important for JSON checkpointing).
        """
        rng = random.Random(seed)
        samples: List[KrachtVector] = []
        for _ in range(int(n)):
            values = {
                name: rng.uniform(*self.bounds[name])
                for name in KRACHT_PARAMETER_NAMES
            }
            samples.append(KrachtVector(values=values))
        return samples

    def validate(self, vector: KrachtVector) -> bool:
        """Return True iff ``vector`` has every declared parameter in range."""
        return not self.constraint_violations(vector)

    def constraint_violations(self, vector: KrachtVector) -> List[str]:
        """Return machine-readable engineering constraint violations.

        Bounds are the first line of defence; coupled checks catch
        parameter combinations that are individually legal but prone to
        folded or unmanufacturable bulb geometry.
        """
        values = vector.values
        violations: List[str] = []
        for name in KRACHT_PARAMETER_NAMES:
            if name not in values:
                violations.append(f"{name}_missing")
                continue
            lo, hi = self.bounds[name]
            value = float(values[name])
            if value < lo or value > hi:
                if name == "nose_sharpness" and value < lo:
                    violations.append("nose_sharpness_below_min")
                else:
                    violations.append(f"{name}_out_of_bounds")

        if violations:
            return violations

        nose = float(values["nose_sharpness"])
        cross_section = float(values["cross_section_c"])
        longitudinal_pos = float(values["longitudinal_pos"])
        height_ratio = float(values["height_ratio"])
        volume_coef = float(values["volume_coef"])

        if cross_section > 0.98 and nose < 0.08:
            violations.append("full_section_with_sharp_nose")
        if longitudinal_pos > 0.85 and height_ratio > 0.55:
            violations.append("aft_high_bulb_geometry_risk")
        if volume_coef > 0.88 and nose < 0.07:
            violations.append("high_volume_with_sharp_nose")
        # Audit 2026-04-26 (Add #1): when the optimizer asks the FFD to
        # *deflate* the bulb (volume_coef well below zero) AND simulta-
        # neously demand a sharp nose tip, the resulting deformation
        # inverts the forward-most lattice slab into the body of the
        # bulb. Reject the pair as unmanufacturable.
        if volume_coef < -0.30 and nose < 0.10:
            violations.append("deflate_with_sharp_nose")

        return violations

    def manufacturability_warnings(self, vector: KrachtVector) -> List[str]:
        """Return soft risk labels for valid but boundary-seeking geometry.

        These warnings are intentionally not hard rejections: they mark
        candidates that are still legal but close to regions where the FFD
        parameterisation can create sharp noses, over-full sections, or aft
        high bulbs that deserve manual review before manufacturing.
        """
        if self.constraint_violations(vector):
            return []

        values = vector.values
        warnings: List[str] = []

        def add(label: str) -> None:
            if label not in warnings:
                warnings.append(label)

        length_ratio = float(values["length_ratio"])
        height_ratio = float(values["height_ratio"])
        axis_z_ratio = float(values["axis_z_ratio"])
        longitudinal_pos = float(values["longitudinal_pos"])
        cross_section = float(values["cross_section_c"])
        volume_coef = float(values["volume_coef"])
        nose = float(values["nose_sharpness"])

        if length_ratio >= 0.043:
            add("length_ratio_near_upper_bound")
        if height_ratio >= 0.52:
            add("height_ratio_near_upper_bound")
        if axis_z_ratio >= 0.48:
            add("axis_z_ratio_near_upper_bound")
        if longitudinal_pos >= 0.80:
            add("longitudinal_pos_near_aft_limit")
        if cross_section >= 0.95:
            add("cross_section_c_near_upper_bound")
        if volume_coef >= 0.86:
            add("volume_coef_near_upper_bound")
        if nose <= 0.10:
            add("nose_sharpness_near_lower_bound")

        if cross_section >= 0.95 and nose <= 0.10:
            add("sharp_full_section_near_limit")
        if volume_coef >= 0.86 and nose <= 0.10:
            add("high_volume_sharp_nose_near_limit")
        if longitudinal_pos >= 0.80 and height_ratio >= 0.52:
            add("aft_high_bulb_near_limit")
        if axis_z_ratio >= 0.48 and height_ratio >= 0.52:
            add("high_axis_high_bulb_near_limit")

        return warnings

    def to_array(self, vector: KrachtVector) -> List[float]:
        """Convert a vector to a list of floats in declared order."""
        return [float(vector.values[name]) for name in KRACHT_PARAMETER_NAMES]

    def from_array(self, array: Iterable[float]) -> KrachtVector:
        """Invert :meth:`to_array` — build a KrachtVector from a float list."""
        values = {
            name: float(value)
            for name, value in zip(KRACHT_PARAMETER_NAMES, array, strict=True)
        }
        return KrachtVector(values=values)


# Audit 2026-04-26 — Module C: empirically-validated -2.48 % Cd Kracht
# vector from a real CFD run on ``docs/base_hull.stl``. Used as the
# warm-start seed for the night-optimization GA and as the centroid of
# the ``KrachtDesignSpace.tightened()`` search region.
TINY_SHARP_KRACHT_VECTOR: KrachtVector = KrachtVector(
    values={
        "length_ratio":     0.005,
        "breadth_ratio":    0.05,
        "height_ratio":     0.10,
        "axis_z_ratio":     0.25,
        "longitudinal_pos": 0.5,
        "cross_section_c":  0.8,
        "volume_coef":      -0.20,
        "nose_sharpness":   0.95,
    }
)
