"""High-fidelity cascade gate that drives the existing OpenFOAM adapter.

Design reference: 2026-04-22-bulbopt-night-optimization-design.md §9.

For each KrachtVector the gate:

1. Deforms the baseline mesh via :class:`BulbFFDDeformer`.
2. Writes the deformed STL to a per-candidate working directory.
3. Calls the injected ``build_case`` (normally
   :class:`OpenFOAMAdapter.build_case`) to materialise the OpenFOAM
   case tree around that STL.
4. Calls the injected ``run_case`` (normally
   :class:`OpenFOAMRunnerAdapter.run_case` with ``execute=True``) to
   invoke ``blockMesh`` + ``snappyHexMesh`` (and later ``simpleFoam``
   with ``forceCoeffs``).
5. Returns two objectives per candidate:
     * primary drag proxy (lower is better) — currently a geometric
       surrogate derived from the deformed mesh, upgraded to a real
       ``forceCoeffs`` reading once simpleFoam convergence detection
       lands in ``openfoam_runner``.
     * volume delta relative to baseline (lower is better).

On runner failure the gate returns a penalty fitness rather than
raising, so NSGA-II can still sort the population and the Pareto front
stays interpretable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Sequence
import uuid

import trimesh

from bulbopt.infrastructure.adapters.force_coeffs_parser import (
    ForceCoeffsNotFoundError,
    parse_drag_coefficient_dat,
)
from bulbopt.infrastructure.adapters.stl_sanity import validate_stl
from bulbopt.optimization.parametric.ffd_deformer import BulbFFDDeformer
from bulbopt.optimization.parametric.kracht_space import KrachtDesignSpace, KrachtVector


BuildCaseFn = Callable[..., dict]
RunCaseFn = Callable[..., dict]


@dataclass(slots=True)
class SimpleFoamHighFidelityGate:
    work_root: Path
    baseline_mesh: trimesh.Trimesh
    region: dict
    deformer: BulbFFDDeformer
    build_case: BuildCaseFn
    run_case: RunCaseFn
    # Reference state for Cd -> Newtons conversion (spec §9). All three
    # must be supplied to enable the conversion; otherwise the gate
    # falls back to the dimensionless coefficient and the proxy delta.
    reference_velocity_m_s: float | None = None
    reference_area_m2: float | None = None
    fluid_density_kg_m3: float | None = None
    evaluation_records: list[dict] = field(init=False, default_factory=list)
    _baseline_volume: float = field(init=False, default=0.0)
    _design_space: KrachtDesignSpace = field(init=False)

    def __post_init__(self) -> None:
        self._baseline_volume = _mesh_volume(self.baseline_mesh)
        self._design_space = KrachtDesignSpace()
        Path(self.work_root).mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        vectors: Sequence[KrachtVector],
        *,
        froude_numbers: Sequence[float] | None = None,
        froude_weights: Sequence[float] | None = None,
    ) -> List[List[float]]:
        """Evaluate ``vectors`` and return ``[aggregated_cd, vol_delta]`` rows.

        Parameters
        ----------
        vectors:
            KrachtVectors to evaluate.
        froude_numbers:
            Optional list of Froude numbers. When ``None`` (default) each
            candidate runs simpleFoam once at the case-template default Fr
            and the existing single-Fr behaviour is preserved bit-exactly.
            When provided, each candidate runs simpleFoam once per Fr and
            the per-Fr Cd values are aggregated as a weighted mean.
        froude_weights:
            Optional weights matching ``froude_numbers``. When ``None``,
            uniform weights are used. The weights are normalised to sum
            to 1.0 before aggregation.
        """
        normalised_weights = _normalise_froude_weights(
            froude_numbers, froude_weights
        )
        objectives: List[List[float]] = []
        for vector in vectors:
            if froude_numbers is None:
                row = self._evaluate_one(vector)
            else:
                row = self._evaluate_one_multi_fr(
                    vector,
                    list(froude_numbers),
                    list(normalised_weights or []),
                )
            objectives.append(row)
        return objectives

    def _evaluate_one_multi_fr(
        self,
        vector: KrachtVector,
        froude_numbers: list[float],
        froude_weights: list[float],
    ) -> List[float]:
        """Run simpleFoam once per Fr, aggregate Cd as a weighted mean.

        Volume delta is identical across Fr (geometry is invariant), so we
        pick whichever value the per-Fr runs report (they agree).
        """
        per_fr_rows: list[List[float]] = []
        for fr in froude_numbers:
            row = self._evaluate_one(vector, froude_number=float(fr))
            per_fr_rows.append(row)

        # If every per-Fr row hit the same up-front penalty (constraint
        # violation, STL invalid) the aggregate is just that penalty —
        # bail out rather than mix penalty Cd into a weighted mean.
        if all(row == [1e9, 1e9] for row in per_fr_rows):
            return [1e9, 1e9]

        cd_values = [row[0] for row in per_fr_rows]
        vol_deltas = [row[1] for row in per_fr_rows]
        aggregated_cd = sum(
            w * cd for w, cd in zip(froude_weights, cd_values)
        )
        # Volume delta is geometry-only; pick the first valid one (they
        # are all the same).
        return [float(aggregated_cd), float(vol_deltas[0])]

    def _evaluate_one(
        self,
        vector: KrachtVector,
        *,
        froude_number: float | None = None,
    ) -> List[float]:
        constraint_violations = self._design_space.constraint_violations(vector)
        if constraint_violations:
            self.evaluation_records.append(
                {
                    "parameters": dict(vector.values),
                    "solver_status": "skipped",
                    "solver_reason": "constraint_violation",
                    "constraint_violations": list(constraint_violations),
                    "objectives": [1e9, 1e9],
                }
            )
            return [1e9, 1e9]

        candidate_id = f"candidate-{uuid.uuid4().hex[:8]}"
        deformed = self.deformer.deform(self.baseline_mesh, self.region, vector)
        stl_report = validate_stl(deformed)
        if not stl_report["checks_passed"]:
            self.evaluation_records.append(
                {
                    "parameters": dict(vector.values),
                    "foam_candidate_id": candidate_id,
                    "solver_status": "skipped",
                    "solver_reason": "stl_invalid",
                    "stl_report": stl_report,
                    "objectives": [1e9, 1e9],
                }
            )
            return [1e9, 1e9]

        # Default single-Fr path puts the candidate at work_root/<id> so
        # the existing tests that watch per-candidate dirs still pass. For
        # multi-Fr we drop into a per-Fr sub-dir so each simpleFoam run
        # gets its own postProcessing tree.
        if froude_number is None:
            candidate_case_dir = Path(self.work_root) / candidate_id
        else:
            fr_label = f"fr_{froude_number:.4f}".replace(".", "p").replace("-", "m")
            candidate_case_dir = Path(self.work_root) / candidate_id / fr_label
        candidate_case_dir.mkdir(parents=True, exist_ok=True)

        geometry_path = candidate_case_dir / "input" / "candidate.stl"
        geometry_path.parent.mkdir(parents=True, exist_ok=True)
        geometry_path.write_bytes(trimesh.exchange.stl.export_stl(deformed))

        drag_proxy = _drag_proxy(deformed, self.region)
        volume_delta = _volume_delta(deformed, self._baseline_volume)
        record = {
            "parameters": dict(vector.values),
            "foam_candidate_id": candidate_id,
            "candidate_work_dir": str(candidate_case_dir),
            "input_geometry_path": str(geometry_path),
            "drag_proxy": float(drag_proxy),
            "volume_delta": float(volume_delta),
        }
        if froude_number is not None:
            record["froude_number"] = float(froude_number)

        try:
            build_kwargs: dict = {
                "best_candidate_id": candidate_id,
                "best_candidate_geometry_path": geometry_path,
            }
            if froude_number is not None:
                build_kwargs["froude_number"] = float(froude_number)
            manifest = self.build_case(candidate_case_dir, **build_kwargs)
            run_manifest = self.run_case(
                candidate_case_dir / "working" / "openfoam_case",
                case_manifest=manifest,
                execute=True,
            )
        except Exception:
            objectives = [_penalty_value(drag_proxy), volume_delta]
            record.update(
                {
                    "solver_status": "exception",
                    "objectives": list(objectives),
                }
            )
            self.evaluation_records.append(record)
            return objectives

        status = run_manifest.get("status", "unknown")
        record.update(
            {
                "solver_status": status,
                "solver_reason": run_manifest.get("reason"),
                "run_manifest": run_manifest,
            }
        )
        if status != "executed_ok":
            objectives = [_penalty_value(drag_proxy), volume_delta]
            record["objectives"] = list(objectives)
            self.evaluation_records.append(record)
            return objectives

        # Try to read real forceCoeffs drag; fall back to geometric proxy
        # when the file is missing (e.g. simpleFoam didn't run because the
        # case stops after snappyHexMesh).
        foam_case_dir = candidate_case_dir / "working" / "openfoam_case"
        cd_report = _read_force_coeffs(
            foam_case_dir,
            reference_velocity=self.reference_velocity_m_s,
            reference_area=self.reference_area_m2,
            fluid_density=self.fluid_density_kg_m3,
        )
        record["force_coeffs"] = cd_report
        if cd_report is not None and cd_report.get("drag_newtons") is not None:
            objectives = [float(cd_report["drag_newtons"]), volume_delta]
            record["objectives"] = list(objectives)
            self.evaluation_records.append(record)
            return objectives
        if cd_report is not None:
            objectives = [float(cd_report["final_cd"]), volume_delta]
            record["objectives"] = list(objectives)
            self.evaluation_records.append(record)
            return objectives
        objectives = [drag_proxy, volume_delta]
        record["objectives"] = list(objectives)
        self.evaluation_records.append(record)
        return objectives


def _normalise_froude_weights(
    froude_numbers: Sequence[float] | None,
    froude_weights: Sequence[float] | None,
) -> list[float] | None:
    """Validate and normalise the per-Fr weights to sum to 1.0.

    Returns ``None`` when ``froude_numbers`` is ``None`` (single-Fr path).
    Raises ``ValueError`` when the inputs are inconsistent.
    """
    if froude_numbers is None:
        return None
    fr_list = list(froude_numbers)
    if not fr_list:
        raise ValueError("froude_numbers must contain at least one entry")
    if froude_weights is None:
        uniform = 1.0 / len(fr_list)
        return [uniform for _ in fr_list]
    weights = [float(w) for w in froude_weights]
    if len(weights) != len(fr_list):
        raise ValueError(
            "froude_weights length must match froude_numbers length "
            f"({len(weights)} vs {len(fr_list)})"
        )
    if any(w < 0.0 for w in weights):
        raise ValueError("froude_weights entries must be non-negative")
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("froude_weights must have a positive sum")
    return [w / total for w in weights]


def _drag_proxy(mesh: trimesh.Trimesh, region: dict) -> float:
    extents = mesh.extents.astype(float)
    primary = int(region.get("axis_index", int(extents.argmax())))
    others = [i for i in range(3) if i != primary]
    axial = max(float(extents[primary]), 1e-9)
    beam = max(float(extents[others[0]]), 1e-9)
    draft = max(float(extents[others[1]]), 1e-9)
    return (beam * draft) / axial


def _volume_delta(mesh: trimesh.Trimesh, baseline_volume: float) -> float:
    candidate_volume = _mesh_volume(mesh)
    if baseline_volume <= 0:
        return 0.0
    return abs(candidate_volume - baseline_volume) / baseline_volume


def _mesh_volume(mesh: trimesh.Trimesh) -> float:
    if mesh.is_volume:
        return float(abs(mesh.volume))
    extents = mesh.extents.astype(float)
    return float(abs(extents[0] * extents[1] * extents[2]))


def _penalty_value(baseline_proxy: float) -> float:
    """Large finite penalty so NSGA-II demotes failed runs but still sorts."""
    return max(baseline_proxy * 1000.0, 1.0)


def _read_force_coeffs(
    foam_case_dir: Path,
    *,
    reference_velocity: float | None,
    reference_area: float | None,
    fluid_density: float | None,
) -> dict | None:
    """Locate and parse the latest forceCoeffs coefficient.dat, if any.

    OpenFOAM writes to ``postProcessing/forces/<startTime>/coefficient.dat``;
    for a steady-state simpleFoam run the directory name is usually ``0``
    but we pick the most recently modified one so the parser works for
    restarted cases too.
    """
    # OpenFOAM writes the output under a directory named after the
    # function object key in controlDict.functions. Our adapter uses
    # ``forceCoeffs`` (see openfoam_adapter._control_dict); earlier
    # tutorials used ``forces``. Check both, newest first.
    post_root = foam_case_dir / "postProcessing"
    if not post_root.exists():
        return None
    candidate_roots = [
        post_root / "forceCoeffs",
        post_root / "forces",
    ]
    forces_root = next((c for c in candidate_roots if c.exists()), None)
    if forces_root is None:
        return None
    subdirs = [p for p in forces_root.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    latest = max(subdirs, key=lambda p: p.stat().st_mtime)
    dat_path = next(
        (
            latest / filename
            for filename in ("forceCoeffs.dat", "coefficient.dat")
            if (latest / filename).exists()
        ),
        latest / "coefficient.dat",
    )
    try:
        return parse_drag_coefficient_dat(
            dat_path,
            reference_velocity_m_s=reference_velocity,
            reference_area_m2=reference_area,
            fluid_density_kg_m3=fluid_density,
        )
    except (ForceCoeffsNotFoundError, ValueError):
        return None
