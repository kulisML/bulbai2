from __future__ import annotations

from pathlib import Path
from typing import Sequence

import trimesh


# Realistic displacement-hull Froude range (Fr = V / sqrt(g * L)). Values
# below 0.2 are harbour-speed; above 0.35 hulls start to plane — outside
# the ships we target. Uniformly-spaced 5-sample sweep is the default.
DEFAULT_FROUDE_SAMPLES: tuple[float, ...] = (0.15, 0.20, 0.25, 0.30, 0.35)


class StubEvaluationAdapter:
    def evaluate_candidates(
        self,
        candidates: list[dict],
        objective_weights: dict[str, float] | None = None,
        speed_knots: list[float] | None = None,
        operational_profile_weights: list[float] | None = None,
        wave_height_m: float = 0.0,
        wave_period_s: float = 0.0,
        wave_scenario_heights_m: list[float] | None = None,
        wave_scenario_periods_s: list[float] | None = None,
        wave_scenario_weights: list[float] | None = None,
        acceptability_thresholds: dict[str, float] | None = None,
    ) -> list[dict]:
        weights = objective_weights or self._default_objective_weights()
        speeds = speed_knots or [18.0]
        thresholds = acceptability_thresholds or self._default_acceptability_thresholds()
        evaluated: list[dict] = []
        for candidate in candidates:
            candidate_path = Path(candidate["geometry_path"])
            repaired_path = candidate_path.parents[1] / "repaired" / "repaired.stl"
            candidate_mesh = self._load_mesh(candidate_path)
            repaired_mesh = self._load_mesh(repaired_path)
            geometry_metrics = self._build_geometry_metrics(candidate, candidate_mesh, repaired_mesh)
            hydrostatics_metrics = self._build_hydrostatics_metrics(
                geometry_metrics,
                candidate_mesh,
                repaired_mesh,
                thresholds=thresholds,
            )
            score_components = self._build_score_components(geometry_metrics, hydrostatics_metrics)
            reference_resistance_proxy = self._build_reference_resistance_proxy(
                candidate,
                repaired_mesh,
            )
            calm_water_metrics = self._build_calm_water_metrics(
                score_components,
                speeds,
                geometry_metrics=geometry_metrics,
                reference_resistance_proxy=reference_resistance_proxy,
                operational_profile_weights=operational_profile_weights,
            )
            score_components["calm_water_penalty"] = calm_water_metrics["calm_water_penalty"]
            wave_response_metrics = self._build_wave_response_metrics(
                geometry_metrics,
                calm_water_metrics,
                wave_height_m=wave_height_m,
                wave_period_s=wave_period_s,
                wave_scenario_heights_m=wave_scenario_heights_m,
                wave_scenario_periods_s=wave_scenario_periods_s,
                wave_scenario_weights=wave_scenario_weights,
            )
            score_components["wave_penalty"] = wave_response_metrics["wave_penalty"]
            multi_condition_objective = self._build_multi_condition_objective(
                calm_water_metrics,
                wave_response_metrics,
                condition_weights=weights,
            )
            score_components["multi_condition_penalty"] = multi_condition_objective["combined_penalty"]
            selection_priority = self._build_selection_priority(
                calm_water_metrics,
                hydrostatics_metrics,
                wave_response_metrics,
                multi_condition_objective,
            )
            score_components["selection_priority_score"] = selection_priority["selection_priority_score"]
            acceptability = self._build_acceptability_summary(
                hydrostatics_metrics,
                calm_water_metrics,
                wave_response_metrics,
                thresholds=thresholds,
            )
            resistance_proxy = max(score_components["resistance_proxy"], 1e-6)
            fast_score = round(1.0 / resistance_proxy, 6)

            # Froude sweep: aggregate the resistance contribution across 5
            # realistic displacement-hull Fr values so the scoring reflects
            # performance across the whole operational envelope, not a
            # single design speed.
            froude_sweep = self._resistance_across_froude_sweep(
                resistance_proxy=resistance_proxy,
                geometry_metrics=geometry_metrics,
                operational_profile_weights=operational_profile_weights,
            )
            score_components["froude_sweep_samples"] = froude_sweep["samples"]
            score_components["froude_sweep_aggregate_resistance"] = (
                froude_sweep["aggregate_resistance"]
            )

            mid_score = round(
                (weights["resistance_weight"] * resistance_proxy)
                - (weights["axial_gain_weight"] * geometry_metrics["axial_gain_m"])
                - (weights["draft_reduction_weight"] * geometry_metrics["draft_reduction_m"])
                + (weights["beam_growth_weight"] * geometry_metrics["beam_growth_m"]),
                6,
            )
            mid_score = round(
                mid_score
                + (0.05 * score_components["hydrostatic_penalty"])
                + (0.012 * score_components["multi_condition_penalty"])
                + (0.02 * froude_sweep["aggregate_resistance"]),
                6,
            )
            evaluated.append(
                {
                    **candidate,
                    "status": "mid_score_ready",
                    "fast_score": fast_score,
                    "mid_score": mid_score,
                    "objective_weights": weights,
                    "geometry_metrics": geometry_metrics,
                    "hydrostatics_metrics": hydrostatics_metrics,
                    "calm_water_metrics": calm_water_metrics,
                    "wave_response_metrics": wave_response_metrics,
                    "multi_condition_objective": multi_condition_objective,
                    "selection_priority": selection_priority,
                    "acceptability": acceptability,
                    "score_components": score_components,
                }
            )
        return evaluated

    def _default_objective_weights(self) -> dict[str, float]:
        return {
            "resistance_weight": 1.0,
            "axial_gain_weight": 0.8,
            "draft_reduction_weight": 0.1,
            "beam_growth_weight": 0.05,
            "calm_water_condition_weight": 0.7,
            "wave_condition_weight": 0.3,
        }

    def _default_acceptability_thresholds(self) -> dict[str, float]:
        return {
            "max_volume_delta_pct": 4.5,
            "max_draft_delta_m": 0.05,
            "max_speed_balance_ratio": 4.0,
            "max_wave_penalty": 1.5,
            "reject_volume_delta_pct": 9.0,
            "reject_draft_delta_m": 0.1,
            "reject_speed_balance_ratio": 8.0,
            "reject_wave_penalty": 3.0,
        }

    def _load_mesh(self, source_path: Path) -> trimesh.Trimesh:
        mesh = trimesh.load(source_path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
            raise ValueError(f"Unable to load mesh for evaluation: {source_path}")
        mesh = mesh.copy()
        mesh.remove_unreferenced_vertices()
        return mesh

    def _build_geometry_metrics(
        self,
        candidate: dict,
        candidate_mesh: trimesh.Trimesh,
        repaired_mesh: trimesh.Trimesh,
    ) -> dict[str, float]:
        bulb_region = candidate.get("bulb_region", {})
        primary_axis = int(bulb_region.get("axis_index", int(candidate_mesh.extents.argmax())))
        beam_axis, draft_axis = [axis for axis in range(3) if axis != primary_axis]

        candidate_extents = candidate_mesh.extents.astype(float)
        repaired_extents = repaired_mesh.extents.astype(float)

        vertices = candidate_mesh.vertices
        nose_threshold = float(bulb_region.get("axis_min", float(vertices[:, primary_axis].min())))
        nose_mask = vertices[:, primary_axis] >= nose_threshold
        if nose_mask.any():
            nose_vertices = vertices[nose_mask]
            nose_area_proxy = float(
                (nose_vertices[:, beam_axis].max() - nose_vertices[:, beam_axis].min())
                * (nose_vertices[:, draft_axis].max() - nose_vertices[:, draft_axis].min())
            )
        else:
            nose_area_proxy = float(candidate_extents[beam_axis] * candidate_extents[draft_axis])

        beam_extent = float(candidate_extents[beam_axis])
        draft_extent = float(candidate_extents[draft_axis])
        axial_extent = float(candidate_extents[primary_axis])
        repaired_beam = float(repaired_extents[beam_axis])
        repaired_draft = float(repaired_extents[draft_axis])
        repaired_axial = float(repaired_extents[primary_axis])

        return {
            "axial_extent_m": axial_extent,
            "beam_extent_m": beam_extent,
            "draft_extent_m": draft_extent,
            "surface_area_m2": float(candidate_mesh.area),
            "slenderness_ratio": axial_extent / max(beam_extent, draft_extent, 1e-6),
            "nose_area_proxy_m2": nose_area_proxy,
            "axial_gain_m": max(axial_extent - repaired_axial, 0.0),
            "beam_growth_m": max(beam_extent - repaired_beam, 0.0),
            "draft_reduction_m": max(repaired_draft - draft_extent, 0.0),
        }

    def _build_hydrostatics_metrics(
        self,
        geometry_metrics: dict[str, float],
        candidate_mesh: trimesh.Trimesh,
        repaired_mesh: trimesh.Trimesh,
        thresholds: dict[str, float],
    ) -> dict[str, float]:
        max_volume_delta_pct = float(thresholds["max_volume_delta_pct"])
        max_draft_delta_m = float(thresholds["max_draft_delta_m"])
        reject_volume_delta_pct = float(thresholds["reject_volume_delta_pct"])
        reject_draft_delta_m = float(thresholds["reject_draft_delta_m"])
        candidate_volume = self._mesh_volume_proxy(candidate_mesh)
        repaired_volume = self._mesh_volume_proxy(repaired_mesh)
        volume_delta_pct = abs(candidate_volume - repaired_volume) / max(repaired_volume, 1e-6) * 100.0
        draft_delta_m = geometry_metrics["draft_extent_m"] - float(repaired_mesh.extents.astype(float)[2])
        hydrostatic_penalty = (volume_delta_pct / 10.0) + abs(draft_delta_m)
        warnings: list[str] = []
        volume_level = "ok"
        draft_level = "ok"
        if volume_delta_pct > reject_volume_delta_pct:
            volume_level = "reject"
        elif volume_delta_pct > max_volume_delta_pct:
            volume_level = "warn"
        if abs(draft_delta_m) > reject_draft_delta_m:
            draft_level = "reject"
        elif abs(draft_delta_m) > max_draft_delta_m:
            draft_level = "warn"
        if volume_level != "ok":
            warnings.append("volume_delta_exceeds_limit")
        if draft_level != "ok":
            warnings.append("draft_delta_exceeds_limit")
        severity_order = {"ok": 0, "warn": 1, "reject": 2}
        constraint_status = max((volume_level, draft_level), key=severity_order.get)
        return {
            "volume_proxy_m3": round(candidate_volume, 6),
            "reference_volume_proxy_m3": round(repaired_volume, 6),
            "volume_delta_pct": round(volume_delta_pct, 6),
            "draft_delta_m": round(draft_delta_m, 6),
            "hydrostatic_penalty": round(hydrostatic_penalty, 6),
            "constraint_status": constraint_status,
            "warnings": warnings,
        }

    def _build_score_components(
        self,
        geometry_metrics: dict[str, float],
        hydrostatics_metrics: dict[str, float],
    ) -> dict[str, float]:
        frontal_area_proxy = geometry_metrics["beam_extent_m"] * geometry_metrics["draft_extent_m"]
        resistance_proxy = frontal_area_proxy / max(geometry_metrics["axial_extent_m"], 1e-6)
        return {
            "frontal_area_proxy_m2": round(frontal_area_proxy, 6),
            "resistance_proxy": max(round(resistance_proxy, 6), 1e-6),
            "hydrostatic_penalty": hydrostatics_metrics["hydrostatic_penalty"],
            "calm_water_penalty": 0.0,
            "wave_penalty": 0.0,
            "multi_condition_penalty": 0.0,
            "effective_power_penalty": 0.0,
        }

    def _build_wave_response_metrics(
        self,
        geometry_metrics: dict[str, float],
        calm_water_metrics: dict[str, float | list[dict[str, float]]],
        wave_height_m: float,
        wave_period_s: float,
        wave_scenario_heights_m: list[float] | None = None,
        wave_scenario_periods_s: list[float] | None = None,
        wave_scenario_weights: list[float] | None = None,
    ) -> dict[str, float | str]:
        scenarios, scenario_source = self._resolve_wave_scenarios(
            wave_height_m=wave_height_m,
            wave_period_s=wave_period_s,
            wave_scenario_heights_m=wave_scenario_heights_m,
            wave_scenario_periods_s=wave_scenario_periods_s,
            wave_scenario_weights=wave_scenario_weights,
        )
        if not scenarios:
            return {
                "wave_height_m": max(float(wave_height_m), 0.0),
                "wave_period_s": max(float(wave_period_s), 0.0),
                "scenario_source": "disabled",
                "scenario_count": 0,
                "scenarios": [],
                "scenario_rows": [],
                "dominant_scenario_label": "n/a",
                "added_resistance_proxy": 0.0,
                "added_power_proxy_kw": 0.0,
                "wave_penalty": 0.0,
                "condition_status": "disabled",
            }

        base_resistance = float(calm_water_metrics.get("aggregate_resistance_proxy", 0.0))
        base_power = float(calm_water_metrics.get("aggregate_power_proxy_kw", 0.0))
        slenderness = max(float(geometry_metrics.get("slenderness_ratio", 1.0)), 1.0)
        scenarios = [
            self._score_wave_scenario(
                scenario=item,
                geometry_metrics=geometry_metrics,
                base_resistance=base_resistance,
                base_power=base_power,
                slenderness=slenderness,
            )
            for item in scenarios
        ]
        dominant_scenario = max(scenarios, key=lambda item: item["scenario_weight"])
        scenario_rows = [
            (
                f"{item['label']} weight={item['scenario_weight']} "
                f"added_resistance={item['added_resistance_proxy']} "
                f"added_power={item['added_power_proxy_kw']} "
                f"penalty={item['condition_penalty']}"
            )
            for item in scenarios
        ]
        added_resistance_proxy = round(
            sum(item["added_resistance_proxy"] * item["scenario_weight"] for item in scenarios),
            6,
        )
        added_power_proxy_kw = round(
            sum(item["added_power_proxy_kw"] * item["scenario_weight"] for item in scenarios),
            6,
        )
        wave_penalty = round(
            sum(item["condition_penalty"] * item["scenario_weight"] for item in scenarios),
            6,
        )
        return {
            "wave_height_m": dominant_scenario["wave_height_m"],
            "wave_period_s": dominant_scenario["wave_period_s"],
            "scenario_source": scenario_source,
            "scenario_count": len(scenarios),
            "scenarios": scenarios,
            "scenario_rows": scenario_rows,
            "dominant_scenario_label": dominant_scenario["label"],
            "added_resistance_proxy": added_resistance_proxy,
            "added_power_proxy_kw": added_power_proxy_kw,
            "wave_penalty": wave_penalty,
            "condition_status": "active",
        }

    def _score_wave_scenario(
        self,
        *,
        scenario: dict[str, float | str],
        geometry_metrics: dict[str, float],
        base_resistance: float,
        base_power: float,
        slenderness: float,
    ) -> dict[str, float | str]:
        normalized_height = float(scenario["wave_height_m"])
        normalized_period = float(scenario["wave_period_s"])
        wave_height_factor = normalized_height / max(float(geometry_metrics.get("draft_extent_m", 1.0)), 1e-6)
        period_factor = normalized_period / max(float(geometry_metrics.get("axial_extent_m", 1.0)) / 3.5, 1e-6)
        bow_response_factor = (wave_height_factor * 0.65) + (period_factor * 0.35)
        slenderness_relief = 1.0 / min(slenderness, 8.0)
        added_resistance_proxy = max(
            round(base_resistance * bow_response_factor * (0.55 + slenderness_relief), 6),
            0.0,
        )
        added_power_proxy_kw = max(
            round(base_power * bow_response_factor * (0.2 + (0.6 * slenderness_relief)), 6),
            0.0,
        )
        return {
            **scenario,
            "added_resistance_proxy": added_resistance_proxy,
            "added_power_proxy_kw": added_power_proxy_kw,
            "condition_penalty": round(added_resistance_proxy + (added_power_proxy_kw / 150.0), 6),
        }

    def _resolve_wave_scenarios(
        self,
        wave_height_m: float,
        wave_period_s: float,
        wave_scenario_heights_m: list[float] | None,
        wave_scenario_periods_s: list[float] | None,
        wave_scenario_weights: list[float] | None,
    ) -> tuple[list[dict[str, float | str]], str]:
        if (
            wave_scenario_heights_m
            and wave_scenario_periods_s
            and len(wave_scenario_heights_m) == len(wave_scenario_periods_s)
        ):
            weights, source = self._resolve_wave_profile_weights(wave_scenario_heights_m, wave_scenario_weights)
            scenarios = [
                self._build_wave_scenario(
                    height_m=float(height),
                    period_s=float(period),
                    scenario_weight=weight,
                )
                for height, period, weight in zip(wave_scenario_heights_m, wave_scenario_periods_s, weights)
                if float(height) > 0.0 and float(period) > 0.0
            ]
            if scenarios:
                return scenarios, source

        if max(float(wave_height_m), 0.0) > 0.0 and max(float(wave_period_s), 0.0) > 0.0:
            return (
                [
                    self._build_wave_scenario(
                        height_m=float(wave_height_m),
                        period_s=float(wave_period_s),
                        scenario_weight=1.0,
                    )
                ],
                "single_condition",
            )

        return [], "disabled"

    def _resolve_wave_profile_weights(
        self,
        wave_scenario_heights_m: list[float],
        wave_scenario_weights: list[float] | None,
    ) -> tuple[list[float], str]:
        if wave_scenario_weights and len(wave_scenario_weights) == len(wave_scenario_heights_m):
            normalized = [max(float(weight), 0.0) for weight in wave_scenario_weights]
            total = sum(normalized)
            if total > 0.0:
                return ([weight / total for weight in normalized], "user_defined")

        count = max(len(wave_scenario_heights_m), 1)
        return ([1.0 / count for _ in wave_scenario_heights_m], "uniform")

    def _build_wave_scenario(
        self,
        *,
        height_m: float,
        period_s: float,
        scenario_weight: float,
    ) -> dict[str, float | str]:
        normalized_height = max(float(height_m), 0.0)
        normalized_period = max(float(period_s), 0.0)
        return {
            "wave_height_m": normalized_height,
            "wave_period_s": normalized_period,
            "scenario_weight": round(max(float(scenario_weight), 0.0), 6),
            "label": f"{normalized_height:.1f}m@{normalized_period:.1f}s",
        }

    def _build_calm_water_metrics(
        self,
        score_components: dict[str, float],
        speed_knots: list[float],
        geometry_metrics: dict[str, float],
        reference_resistance_proxy: float,
        operational_profile_weights: list[float] | None = None,
    ) -> dict[str, float | list[dict[str, float]]]:
        resistance_proxy = float(score_components["resistance_proxy"])
        normalized_speeds = [max(float(speed), 0.0) for speed in speed_knots]
        speed_weights, profile_source = self._resolve_operational_profile_weights(
            normalized_speeds,
            operational_profile_weights,
        )
        speed_points: list[dict[str, float]] = []
        for speed, speed_weight in zip(normalized_speeds, speed_weights):
            speed_factor = speed / 10.0
            froude_number = self._build_froude_number(
                speed,
                reference_length_m=max(float(geometry_metrics.get("axial_extent_m", 1.0)), 1.0),
            )
            wetted_surface_factor = 1.0 + (0.035 * min(speed_factor, 3.0))
            blockiness_factor = 1.0 + (0.08 * min(resistance_proxy, 1.5))
            bow_area_factor = 1.0 + (
                0.04 * min(float(geometry_metrics.get("nose_area_proxy_m2", 0.0)), 4.0)
            )
            effective_resistance_factor = (
                wetted_surface_factor
                * blockiness_factor
                * bow_area_factor
                * (1.0 + (0.2 * froude_number))
            )
            resistance_at_speed = round(resistance_proxy * (speed_factor**2), 6)
            effective_resistance_proxy = round(resistance_at_speed * effective_resistance_factor, 6)
            power_proxy_kw = round(resistance_at_speed * speed * 12.5, 6)
            effective_power_proxy_kw = round(effective_resistance_proxy * speed * 12.5, 6)
            reference_resistance_at_speed = round(reference_resistance_proxy * (speed_factor**2), 6)
            reference_power_proxy_kw = round(reference_resistance_at_speed * speed * 12.5, 6)
            reference_effective_power_proxy_kw = round(
                reference_power_proxy_kw * (1.0 + (0.16 * froude_number)),
                6,
            )
            fuel_proxy_kgph = self._build_fuel_proxy_kgph(
                effective_power_proxy_kw,
                reference_effective_power_proxy_kw,
            )
            reference_fuel_proxy_kgph = self._build_fuel_proxy_kgph(
                reference_effective_power_proxy_kw,
                reference_effective_power_proxy_kw,
            )
            condition_penalty = round(effective_resistance_proxy + (effective_power_proxy_kw / 1000.0), 6)
            speed_points.append(
                {
                    "speed_knots": speed,
                    "speed_weight": round(speed_weight, 6),
                    "froude_number": round(froude_number, 6),
                    "resistance_proxy": resistance_at_speed,
                    "effective_resistance_proxy": effective_resistance_proxy,
                    "power_proxy_kw": power_proxy_kw,
                    "effective_power_proxy_kw": effective_power_proxy_kw,
                    "fuel_proxy_kgph": fuel_proxy_kgph,
                    "reference_resistance_proxy": reference_resistance_at_speed,
                    "reference_power_proxy_kw": reference_power_proxy_kw,
                    "reference_effective_power_proxy_kw": reference_effective_power_proxy_kw,
                    "reference_fuel_proxy_kgph": reference_fuel_proxy_kgph,
                    "resistance_improvement_pct": round(
                        self._relative_improvement(reference_resistance_at_speed, resistance_at_speed), 6
                    ),
                    "power_improvement_pct": round(
                        self._relative_improvement(reference_power_proxy_kw, power_proxy_kw), 6
                    ),
                    "effective_power_improvement_pct": round(
                        self._relative_improvement(reference_effective_power_proxy_kw, effective_power_proxy_kw),
                        6,
                    ),
                    "fuel_improvement_pct": round(
                        self._relative_improvement(reference_fuel_proxy_kgph, fuel_proxy_kgph), 6
                    ),
                    "surrogate_components": {
                        "wetted_surface_factor": round(wetted_surface_factor, 6),
                        "blockiness_factor": round(blockiness_factor, 6),
                        "bow_area_factor": round(bow_area_factor, 6),
                        "effective_resistance_factor": round(effective_resistance_factor, 6),
                    },
                    "condition_penalty": condition_penalty,
                }
            )

        mean_resistance = sum(item["resistance_proxy"] for item in speed_points) / max(len(speed_points), 1)
        mean_power = sum(item["power_proxy_kw"] for item in speed_points) / max(len(speed_points), 1)
        mean_effective_power = sum(item["effective_power_proxy_kw"] for item in speed_points) / max(len(speed_points), 1)
        mean_fuel = sum(item["fuel_proxy_kgph"] for item in speed_points) / max(len(speed_points), 1)
        mean_froude = sum(item["froude_number"] for item in speed_points) / max(len(speed_points), 1)
        mean_reference_power = sum(item["reference_power_proxy_kw"] for item in speed_points) / max(len(speed_points), 1)
        mean_reference_effective_power = sum(
            item["reference_effective_power_proxy_kw"] for item in speed_points
        ) / max(len(speed_points), 1)
        mean_reference_fuel = sum(item["reference_fuel_proxy_kgph"] for item in speed_points) / max(len(speed_points), 1)
        aggregate_resistance = sum(item["resistance_proxy"] * item["speed_weight"] for item in speed_points)
        aggregate_power = sum(item["power_proxy_kw"] * item["speed_weight"] for item in speed_points)
        aggregate_effective_power = sum(item["effective_power_proxy_kw"] * item["speed_weight"] for item in speed_points)
        aggregate_fuel = sum(item["fuel_proxy_kgph"] * item["speed_weight"] for item in speed_points)
        reference_aggregate_resistance = sum(
            item["reference_resistance_proxy"] * item["speed_weight"] for item in speed_points
        )
        reference_aggregate_power = sum(item["reference_power_proxy_kw"] * item["speed_weight"] for item in speed_points)
        reference_aggregate_effective_power = sum(
            item["reference_effective_power_proxy_kw"] * item["speed_weight"] for item in speed_points
        )
        reference_aggregate_fuel = sum(item["reference_fuel_proxy_kgph"] * item["speed_weight"] for item in speed_points)
        calm_water_penalty = round(sum(item["condition_penalty"] * item["speed_weight"] for item in speed_points), 6)
        dominant_speed = max(speed_points, key=lambda item: item["speed_weight"])["speed_knots"] if speed_points else 0.0
        max_penalty = max((item["condition_penalty"] for item in speed_points), default=0.0)
        min_penalty = min((item["condition_penalty"] for item in speed_points), default=1e-6)
        speed_balance_ratio = round(max_penalty / max(min_penalty, 1e-6), 6)
        score_components["effective_power_penalty"] = round(aggregate_effective_power / 1000.0, 6)
        return {
            "surrogate_model": "enhanced_geometry_v1",
            "speed_points": speed_points,
            "speed_count": len(speed_points),
            "speed_knots": normalized_speeds,
            "operational_profile_weights": [round(weight, 6) for weight in speed_weights],
            "profile_source": profile_source,
            "speed_balance_ratio": speed_balance_ratio,
            "mean_froude_number": round(mean_froude, 6),
            "mean_resistance_proxy": round(mean_resistance, 6),
            "mean_power_proxy_kw": round(mean_power, 6),
            "mean_effective_power_proxy_kw": round(mean_effective_power, 6),
            "mean_fuel_proxy_kgph": round(mean_fuel, 6),
            "reference_mean_power_proxy_kw": round(mean_reference_power, 6),
            "reference_mean_effective_power_proxy_kw": round(mean_reference_effective_power, 6),
            "reference_mean_fuel_proxy_kgph": round(mean_reference_fuel, 6),
            "aggregate_resistance_proxy": round(aggregate_resistance, 6),
            "aggregate_power_proxy_kw": round(aggregate_power, 6),
            "aggregate_effective_power_proxy_kw": round(aggregate_effective_power, 6),
            "aggregate_fuel_proxy_kgph": round(aggregate_fuel, 6),
            "reference_aggregate_resistance_proxy": round(reference_aggregate_resistance, 6),
            "reference_aggregate_power_proxy_kw": round(reference_aggregate_power, 6),
            "reference_aggregate_effective_power_proxy_kw": round(reference_aggregate_effective_power, 6),
            "reference_aggregate_fuel_proxy_kgph": round(reference_aggregate_fuel, 6),
            "resistance_improvement_pct": round(
                self._relative_improvement(reference_aggregate_resistance, aggregate_resistance), 6
            ),
            "power_improvement_pct": round(
                self._relative_improvement(reference_aggregate_power, aggregate_power), 6
            ),
            "effective_power_improvement_pct": round(
                self._relative_improvement(reference_aggregate_effective_power, aggregate_effective_power),
                6,
            ),
            "fuel_improvement_pct": round(
                self._relative_improvement(reference_aggregate_fuel, aggregate_fuel), 6
            ),
            "dominant_speed_knots": dominant_speed,
            "calm_water_penalty": calm_water_penalty,
        }

    def _resolve_operational_profile_weights(
        self,
        speed_knots: list[float],
        operational_profile_weights: list[float] | None,
    ) -> tuple[list[float], str]:
        if operational_profile_weights and len(operational_profile_weights) == len(speed_knots):
            normalized = [max(float(weight), 0.0) for weight in operational_profile_weights]
            total = sum(normalized)
            if total > 0.0:
                return ([weight / total for weight in normalized], "user_defined")

        total_speed = sum(speed_knots) or float(len(speed_knots)) or 1.0
        return ([speed / total_speed for speed in speed_knots], "derived_from_speed_knots")

    def _build_acceptability_summary(
        self,
        hydrostatics_metrics: dict[str, float | list[str] | str],
        calm_water_metrics: dict[str, float | list[dict[str, float]]],
        wave_response_metrics: dict[str, float | str],
        thresholds: dict[str, float],
    ) -> dict[str, bool | str | list[str]]:
        reasons: list[str] = []
        hydro_status = str(hydrostatics_metrics.get("constraint_status", "ok"))
        operational_status = "ok"
        speed_balance_ratio = float(calm_water_metrics.get("speed_balance_ratio", 0.0))
        speed_balance_threshold = float(thresholds["max_speed_balance_ratio"])
        reject_speed_balance_ratio = float(thresholds["reject_speed_balance_ratio"])
        if speed_balance_ratio > reject_speed_balance_ratio:
            operational_status = "reject"
        elif speed_balance_ratio > speed_balance_threshold:
            operational_status = "warn"
        if operational_status != "ok":
            reasons.append("operational_profile_imbalance")
            reasons.append("operational_profile_warn")
        wave_status = "ok"
        wave_penalty = float(wave_response_metrics.get("wave_penalty", 0.0))
        reject_wave_penalty = float(thresholds["reject_wave_penalty"])
        max_wave_penalty = float(thresholds["max_wave_penalty"])
        if wave_penalty > reject_wave_penalty:
            wave_status = "reject"
        elif wave_penalty > max_wave_penalty:
            wave_status = "warn"
        if wave_status == "warn":
            reasons.append("wave_response_warn")
        elif wave_status == "reject":
            reasons.append("wave_response_reject")
        if hydro_status != "ok":
            reasons.append("hydrostatics_warn")
        level = self._resolve_acceptability_level(hydro_status, operational_status, wave_status)
        return {
            "is_acceptable": level != "reject",
            "level": level,
            "hydrostatics_status": hydro_status,
            "operational_profile_status": operational_status,
            "wave_response_status": wave_status,
            "reasons": list(dict.fromkeys(reasons)),
        }

    def _resolve_acceptability_level(self, hydro_status: str, operational_status: str, wave_status: str) -> str:
        severity_order = {"ok": 0, "warn": 1, "reject": 2}
        return max((hydro_status, operational_status, wave_status), key=severity_order.get)

    def _build_multi_condition_objective(
        self,
        calm_water_metrics: dict[str, float | list[dict[str, float]]],
        wave_response_metrics: dict[str, float | str],
        condition_weights: dict[str, float],
    ) -> dict[str, float | str]:
        calm_weight = max(float(condition_weights.get("calm_water_condition_weight", 0.7)), 0.0)
        wave_weight = max(float(condition_weights.get("wave_condition_weight", 0.3)), 0.0)
        total_weight = calm_weight + wave_weight
        if total_weight <= 0.0:
            calm_weight = 0.7
            wave_weight = 0.3
            total_weight = 1.0
        normalized_calm = calm_weight / total_weight
        normalized_wave = wave_weight / total_weight
        calm_component = float(calm_water_metrics.get("calm_water_penalty", 0.0)) * normalized_calm
        wave_component = float(wave_response_metrics.get("wave_penalty", 0.0)) * normalized_wave
        combined_penalty = round(calm_component + wave_component, 6)
        combined_objective_score = round(
            (float(calm_water_metrics.get("aggregate_resistance_proxy", 0.0)) * normalized_calm)
            + (
                (
                    float(calm_water_metrics.get("aggregate_resistance_proxy", 0.0))
                    + float(wave_response_metrics.get("added_resistance_proxy", 0.0))
                )
                * normalized_wave
            ),
            6,
        )
        dominant_condition = "wave_response" if wave_component > calm_component else "calm_water"
        return {
            "combined_penalty": combined_penalty,
            "combined_objective_score": combined_objective_score,
            "dominant_condition": dominant_condition,
            "calm_water_weight": round(normalized_calm, 6),
            "wave_response_weight": round(normalized_wave, 6),
        }

    def _build_selection_priority(
        self,
        calm_water_metrics: dict[str, float | list[dict[str, float]]],
        hydrostatics_metrics: dict[str, float | list[str] | str],
        wave_response_metrics: dict[str, float | str],
        multi_condition_objective: dict[str, float | str],
    ) -> dict[str, float | str]:
        aggregate_effective_power = float(calm_water_metrics.get("aggregate_effective_power_proxy_kw", 0.0))
        hydro_penalty = float(hydrostatics_metrics.get("hydrostatic_penalty", 0.0))
        wave_penalty = float(wave_response_metrics.get("wave_penalty", 0.0))
        combined_penalty = float(multi_condition_objective.get("combined_penalty", 0.0))
        score = round(
            (aggregate_effective_power / 1000.0)
            + (0.35 * hydro_penalty)
            + (0.18 * wave_penalty)
            + (0.02 * combined_penalty),
            6,
        )
        if score <= 6.0:
            focus_band = "promote"
        elif score <= 12.0:
            focus_band = "review"
        else:
            focus_band = "screen"
        return {
            "calibration_model": "multi_condition_v1",
            "selection_priority_score": score,
            "cfd_focus_band": focus_band,
            "aggregate_effective_power_proxy_kw": round(aggregate_effective_power, 6),
            "hydrostatic_penalty": round(hydro_penalty, 6),
            "wave_penalty": round(wave_penalty, 6),
            "combined_penalty": round(combined_penalty, 6),
        }

    def _mesh_volume_proxy(self, mesh: trimesh.Trimesh) -> float:
        if mesh.is_volume:
            return float(abs(mesh.volume))
        try:
            return float(abs(mesh.convex_hull.volume))
        except Exception:
            extents = mesh.extents.astype(float)
            return float(extents[0] * extents[1] * extents[2])

    def _build_reference_resistance_proxy(self, candidate: dict, repaired_mesh: trimesh.Trimesh) -> float:
        bulb_region = candidate.get("bulb_region", {})
        primary_axis = int(bulb_region.get("axis_index", int(repaired_mesh.extents.argmax())))
        beam_axis, draft_axis = [axis for axis in range(3) if axis != primary_axis]
        repaired_extents = repaired_mesh.extents.astype(float)
        frontal_area_proxy = float(repaired_extents[beam_axis] * repaired_extents[draft_axis])
        return max(round(frontal_area_proxy / max(float(repaired_extents[primary_axis]), 1e-6), 6), 1e-6)

    def _relative_improvement(self, reference_value: float, candidate_value: float) -> float:
        return ((reference_value - candidate_value) / max(reference_value, 1e-6)) * 100.0

    def _build_fuel_proxy_kgph(self, power_proxy_kw: float, reference_power_proxy_kw: float) -> float:
        load_ratio = power_proxy_kw / max(reference_power_proxy_kw, 1e-6)
        specific_fuel_consumption = self._specific_fuel_consumption_kg_per_kwh(load_ratio)
        return round(power_proxy_kw * specific_fuel_consumption, 6)

    def _build_froude_number(self, speed_knots: float, reference_length_m: float) -> float:
        speed_ms = float(speed_knots) * 0.514444
        return speed_ms / max((9.81 * max(reference_length_m, 1e-6)) ** 0.5, 1e-6)

    def _resistance_across_froude_sweep(
        self,
        *,
        resistance_proxy: float,
        geometry_metrics: dict[str, float],
        operational_profile_weights: list[float] | None,
        froude_samples: Sequence[float] = DEFAULT_FROUDE_SAMPLES,
    ) -> dict[str, object]:
        """Compute a per-Froude resistance contribution.

        The resistance proxy is the base geometric "frontal area / axial
        extent" scalar. At each sampled Froude number we multiply it by a
        wave-resistance amplification that rises with Fr (the hull-wave
        interaction grows faster-than-linearly near the hump) and reduces
        back at low Fr where viscous drag dominates. The model is the
        same wave resistance curve the single-speed path used, just
        evaluated on a grid.

        The aggregate is a weighted mean — either the caller's
        operational profile weights (truncated / padded to match the
        sample count) or equal weights per sample when unspecified.

        Returns
        -------
        dict with:
            ``samples`` — list of ``{froude, resistance}`` pairs for every
            sampled Fr.
            ``aggregate_resistance`` — weighted mean across samples.
            ``weights`` — normalised weight vector actually used.
            ``profile_source`` — ``"user_defined"`` or ``"equal"``.
        """
        froudes = [max(float(fr), 0.0) for fr in froude_samples]
        if not froudes:
            return {
                "samples": [],
                "aggregate_resistance": 0.0,
                "weights": [],
                "profile_source": "empty",
            }

        # Wave-resistance amplification. Cw(Fr) peaks around Fr ≈ 0.3 (the
        # classical prismatic-coefficient wave hump) then drops again.
        # Reproduce that shape with a simple quadratic around 0.3.
        slenderness = max(float(geometry_metrics.get("slenderness_ratio", 4.0)), 1.0)
        slenderness_relief = 1.0 / min(slenderness, 8.0)
        samples: list[dict[str, float]] = []
        for fr in froudes:
            # Amplification factor: 1 at Fr=0, peak ~1.6 at Fr=0.3, then
            # decays. This is not a full Michell/Savitsky model — it is a
            # proxy that captures the qualitative shape so different Fr
            # samples actually produce different resistance values.
            amplification = 1.0 + (4.5 * fr * fr) - (6.0 * (fr - 0.3) * (fr - 0.3))
            amplification = max(amplification, 0.0)
            # Slender hulls see less added wave resistance at peak.
            amplification *= 0.85 + (0.3 * slenderness_relief)
            resistance = round(resistance_proxy * (0.25 + amplification), 6)
            samples.append({"froude": round(fr, 6), "resistance": resistance})

        weights, source = self._resolve_froude_sweep_weights(
            froudes, operational_profile_weights
        )
        aggregate = sum(
            sample["resistance"] * weight for sample, weight in zip(samples, weights)
        )
        return {
            "samples": samples,
            "aggregate_resistance": round(aggregate, 6),
            "weights": [round(w, 6) for w in weights],
            "profile_source": source,
        }

    def _resolve_froude_sweep_weights(
        self,
        froudes: list[float],
        operational_profile_weights: list[float] | None,
    ) -> tuple[list[float], str]:
        """Pick a normalised weight vector matching the Froude samples.

        Caller-supplied weights are trimmed or padded with zeros to match
        the sample count, then normalised. Falling back to equal weights
        when the caller didn't supply anything keeps the aggregate
        interpretable as "expected resistance across operating envelope".
        """
        if operational_profile_weights:
            raw = [max(float(w), 0.0) for w in operational_profile_weights[: len(froudes)]]
            if len(raw) < len(froudes):
                raw.extend([0.0] * (len(froudes) - len(raw)))
            total = sum(raw)
            if total > 0:
                return [w / total for w in raw], "user_defined"
        equal = 1.0 / max(len(froudes), 1)
        return [equal for _ in froudes], "equal"

    def _specific_fuel_consumption_kg_per_kwh(self, load_ratio: float) -> float:
        normalized_ratio = min(max(float(load_ratio), 0.35), 1.1)
        off_design_penalty = (normalized_ratio - 0.82) ** 2
        low_load_penalty = max(0.72 - normalized_ratio, 0.0)
        return round(0.182 + (0.022 * off_design_penalty) + (0.01 * low_load_penalty), 6)
