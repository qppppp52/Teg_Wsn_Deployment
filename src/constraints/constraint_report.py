"""Immutable constraint evaluation contract for the deployment problem.

The evaluator in this module deliberately separates three concerns:

* raw hard predicates decide feasibility;
* normalized CV components measure the size of violations;
* physical snapshots are derived from the authoritative solution fields.

Repair code may refresh the compatibility caches on ``Solution`` through the
legacy wrapper in ``constraint_eval.py``, but it must not use those caches as
the source of truth.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Iterable

import numpy as np

from src.heatsink.sink_ownership import (
    SINK_OWNERSHIP_SEMANTICS_VERSION,
    SinkOwnership,
    build_sink_ownership,
)
from src.heatsink.sink_overlap_rules import SinkOverlapRules
from src.physics.numerical_tolerances import (
    ENERGY_ABS_TOL,
    ENERGY_REL_TOL,
    POWER_ABS_TOL,
    POWER_REL_TOL,
)


CV_SCHEMA_VERSION = 3
CONSTRAINT_SEMANTICS_VERSION = 3
COMPARATOR_VERSION = 2
SINK_CLASSIFICATION_VERSION = 2
ENERGY_TOLERANCE_VERSION = 2
POWER_FALLBACK_VERSION = 1
AUTHORITATIVE_FIELDS_VERSION = 1


@dataclass(frozen=True)
class CVComponents:
    deploy: float
    link: float
    power: float
    service: float
    sink: float
    energy: float

    def as_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class EnergyCVBreakdown:
    sensor: float
    ap: float
    aggregate: float


@dataclass(frozen=True)
class SinkHardViolations:
    invalid_index_or_type_count: int
    undeployed_owner_count: int
    outside_allowed_neighborhood_count: int
    illegal_node_overlap_excess: int
    duplicate_excess: int
    cross_owner_conflict_excess: int

    @property
    def satisfied(self) -> bool:
        return not any(
            int(value) > 0
            for value in (
                self.invalid_index_or_type_count,
                self.undeployed_owner_count,
                self.outside_allowed_neighborhood_count,
                self.illegal_node_overlap_excess,
                self.duplicate_excess,
                self.cross_owner_conflict_excess,
            )
        )

    @property
    def total(self) -> int:
        return int(sum(asdict(self).values()))


@dataclass(frozen=True)
class SinkDiagnostics:
    shortage_sensor: int
    shortage_ap: int
    required_sensor_total: int
    required_ap_total: int
    effective_sensor_total: int
    effective_ap_total: int
    unachievable_sensor_count: int
    unachievable_ap_count: int


@dataclass(frozen=True)
class ConsumptionSnapshot:
    sensor: tuple[float, ...]
    ap: tuple[float, ...]


@dataclass(frozen=True)
class EnergySnapshot:
    sensor_harvest: tuple[float, ...]
    ap_harvest: tuple[float, ...]
    sensor_deficit_physical: tuple[float, ...]
    ap_deficit_physical: tuple[float, ...]
    sensor_deficit_violation: tuple[float, ...]
    ap_deficit_violation: tuple[float, ...]
    sensor_deficit_node_count: int
    ap_deficit_node_count: int


@dataclass(frozen=True)
class SinkRequirementSnapshot:
    required_sensor: tuple[int, ...]
    required_ap: tuple[int, ...]
    unachievable_sensor: tuple[bool, ...]
    unachievable_ap: tuple[bool, ...]


@dataclass(frozen=True)
class PhysicalStateSnapshot:
    consumption: ConsumptionSnapshot
    energy: EnergySnapshot
    sink_requirement: SinkRequirementSnapshot


@dataclass(frozen=True)
class RawViolationBreakdown:
    deploy_sensor_excess: int
    deploy_ap_excess: int
    role_overlap_count: int
    invalid_endpoint_link_count: int
    physical_infeasible_link_count: int
    connection_degree_mismatch: int
    nonfinite_power_edge_count: int
    negative_power_edge_count: int
    inactive_nonzero_power_edge_count: int
    power_state_invalid_sensor_count: int
    power_lower_deficit_watt_physical: float
    power_lower_violation_watt: float
    power_upper_excess_watt_physical: float
    power_upper_violation_watt: float
    empty_ap_count: int
    sink_hard: SinkHardViolations
    sink_diagnostics: SinkDiagnostics


@dataclass(frozen=True)
class ConstraintReport:
    schema_version: int
    state_revision: int
    context_signature: str
    components: CVComponents
    energy_cv: EnergyCVBreakdown
    cv_total: float
    feasible: bool
    violated_components: tuple[str, ...]
    max_component_cv: float
    raw: RawViolationBreakdown
    physics: PhysicalStateSnapshot


@dataclass(frozen=True)
class ConstraintEvaluationSpec:
    cv_schema_version: int
    semantics_version: int
    comparator_version: int
    sink_classification_version: int
    ownership_semantics_version: int
    energy_tolerance_version: int
    power_fallback_version: int
    authoritative_fields_version: int
    numeric_zero_tol: float
    reward_metric_scaling_version: int
    objective_semantics_version: int
    power_abs_tol: float
    power_rel_tol: float
    energy_abs_tol: float
    energy_rel_tol: float
    energy_compare_tol: float
    cv_compare_tol: float
    max_sensors: int
    max_aps: int
    ap_service_enabled: bool
    max_ap_rebalance_moves: int
    ptx_max: float
    cv_weights: CVComponents

    @classmethod
    def from_context(cls, ctx) -> "ConstraintEvaluationSpec":
        config = getattr(ctx, "config", None) or {}
        constraints = config.get("constraints", {}) or {}
        deployment = config.get("deployment", {}) or {}
        channel = config.get("channel", {}) or {}
        weights = constraints.get("cv_weights", {}) or {}
        spec = cls(
            cv_schema_version=int(constraints.get("cv_schema_version", CV_SCHEMA_VERSION)),
            semantics_version=int(constraints.get("semantics_version", CONSTRAINT_SEMANTICS_VERSION)),
            comparator_version=int(constraints.get("comparator_version", COMPARATOR_VERSION)),
            sink_classification_version=int(constraints.get("sink_classification_version", SINK_CLASSIFICATION_VERSION)),
            ownership_semantics_version=int(constraints.get("ownership_semantics_version", SINK_OWNERSHIP_SEMANTICS_VERSION)),
            energy_tolerance_version=int(constraints.get("energy_tolerance_version", ENERGY_TOLERANCE_VERSION)),
            power_fallback_version=int(constraints.get("power_fallback_version", POWER_FALLBACK_VERSION)),
            authoritative_fields_version=int(constraints.get("authoritative_fields_version", AUTHORITATIVE_FIELDS_VERSION)),
            numeric_zero_tol=float(constraints.get("numeric_zero_tol", 1.0e-12)),
            reward_metric_scaling_version=int(constraints.get("reward_metric_scaling_version", 1)),
            objective_semantics_version=int(constraints.get("objective_semantics_version", 1)),
            power_abs_tol=float(constraints.get("power_abs_tol", POWER_ABS_TOL)),
            power_rel_tol=float(constraints.get("power_rel_tol", POWER_REL_TOL)),
            energy_abs_tol=float(constraints.get("energy_abs_tol", ENERGY_ABS_TOL)),
            energy_rel_tol=float(constraints.get("energy_rel_tol", ENERGY_REL_TOL)),
            energy_compare_tol=float(constraints.get("energy_compare_tol", 1.0e-12)),
            cv_compare_tol=float(constraints.get("cv_compare_tol", 1.0e-8)),
            max_sensors=int(deployment.get("max_sensors", 20)),
            max_aps=int(deployment.get("max_aps", 4)),
            ap_service_enabled=bool(
                constraints.get(
                    "ap_service_enabled",
                    (config.get("ap", {}) or {}).get("require_nonempty_service", True),
                )
            ),
            max_ap_rebalance_moves=int(constraints.get("max_ap_rebalance_moves", 0)),
            ptx_max=float(channel.get("p_tx_max", 0.5)),
            cv_weights=CVComponents(
                deploy=float(weights.get("deploy", 1.0)),
                link=float(weights.get("link", 1.0)),
                power=float(weights.get("power", 1.0)),
                service=float(weights.get("service", 1.0)),
                sink=float(weights.get("sink", 1.0)),
                energy=float(weights.get("energy", 1.0)),
            ),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        supported_versions = {
            "cv_schema_version": CV_SCHEMA_VERSION,
            "semantics_version": CONSTRAINT_SEMANTICS_VERSION,
            "comparator_version": COMPARATOR_VERSION,
            "sink_classification_version": SINK_CLASSIFICATION_VERSION,
            "ownership_semantics_version": SINK_OWNERSHIP_SEMANTICS_VERSION,
            "energy_tolerance_version": ENERGY_TOLERANCE_VERSION,
            "power_fallback_version": POWER_FALLBACK_VERSION,
            "authoritative_fields_version": AUTHORITATIVE_FIELDS_VERSION,
            "reward_metric_scaling_version": 1,
            "objective_semantics_version": 1,
        }
        for name, supported in supported_versions.items():
            if int(getattr(self, name)) != int(supported):
                raise ValueError(
                    f"unsupported constraint contract version for {name}: "
                    f"{getattr(self, name)} != {supported}"
                )
        for name in (
            "numeric_zero_tol",
            "power_abs_tol",
            "power_rel_tol",
            "energy_abs_tol",
            "energy_rel_tol",
            "energy_compare_tol",
            "cv_compare_tol",
            "ptx_max",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"constraint spec {name} must be finite and non-negative")
        if self.ptx_max <= 0.0:
            raise ValueError("channel.p_tx_max must be positive")
        if self.max_sensors < 0 or self.max_aps < 0:
            raise ValueError("deployment maxima must be non-negative")
        if self.max_ap_rebalance_moves < 0:
            raise ValueError("constraints.max_ap_rebalance_moves must be non-negative")
        if any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in self.cv_weights.as_dict().values()
        ):
            raise ValueError("all CV weights must be finite and positive")

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["cv_weights"] = self.cv_weights.as_dict()
        return data


def ensure_constraint_spec(ctx) -> ConstraintEvaluationSpec:
    """Attach and return the one specification used by a context."""
    spec = getattr(ctx, "constraint_evaluation_spec", None)
    if spec is None:
        spec = ConstraintEvaluationSpec.from_context(ctx)
        setattr(ctx, "constraint_evaluation_spec", spec)
    context_revision = int(getattr(ctx, "context_revision", 0))
    signature = getattr(ctx, "constraint_context_signature", None)
    signature_revision = getattr(ctx, "constraint_context_signature_revision", None)
    if not signature or signature_revision != context_revision:
        setattr(ctx, "constraint_context_signature", build_context_signature(ctx, spec))
        setattr(ctx, "constraint_context_signature_revision", context_revision)
    validate_context_physics(ctx)
    return spec


def mark_context_dirty(ctx) -> None:
    """Invalidate the cached context contract after an in-place physics/config edit."""
    ctx.context_revision = int(getattr(ctx, "context_revision", 0)) + 1
    ctx.constraint_evaluation_spec = None
    ctx.constraint_context_signature = None
    ctx.constraint_context_signature_revision = None


def build_context_signature(ctx, spec: ConstraintEvaluationSpec | None = None) -> str:
    spec = spec or ConstraintEvaluationSpec.from_context(ctx)
    payload: dict[str, Any] = {"spec": spec.as_dict()}
    for name in (
        "coverage_matrix",
        "ptx_min_matrix",
        "link_feasible_matrix",
        "P_grid",
        "channel_gain_matrix",
        "potential_rate_matrix",
        "optimistic_ptx_up_matrix",
    ):
        payload[name] = _array_digest(getattr(ctx, name, None))
    payload["neighbor_sets"] = [
        sorted(int(value) for value in neighbors)
        for neighbors in (getattr(ctx, "neighbor_sets", None) or [])
    ]
    mapping = getattr(ctx, "index_mapping", None)
    if mapping is not None:
        payload["index_mapping"] = {
            "Ls_local_to_global": [int(v) for v in getattr(mapping, "Ls_local_to_global", [])],
            "La_local_to_global": [int(v) for v in getattr(mapping, "La_local_to_global", [])],
        }
    config = getattr(ctx, "config", None) or {}
    payload["config"] = {
        "sensor": config.get("sensor", {}),
        "ap": config.get("ap", {}),
        "channel": config.get("channel", {}),
        "deployment": config.get("deployment", {}),
        "constraints": config.get("constraints", {}),
        "heatsink": config.get("heatsink", {}),
    }
    payload["role_masks"] = {
        "sensor": _array_digest(getattr(ctx, "sensor_mask", None)),
        "ap": _array_digest(getattr(ctx, "ap_mask", None)),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_context_physics(ctx) -> None:
    K = int(getattr(ctx, "num_candidates", 0) or 0)
    p_grid_value = getattr(ctx, "P_grid", None)
    if p_grid_value is not None:
        p_grid = np.asarray(p_grid_value, dtype=float)
        if p_grid.ndim != 1 or (K and p_grid.shape != (K,)):
            raise ValueError(f"context P_grid must have shape {(K,)}")
        if p_grid.size and (not np.all(np.isfinite(p_grid)) or np.any(p_grid < 0.0)):
            raise ValueError("P_grid must be finite and non-negative; zero is allowed")
    for name in ("ptx_min_matrix", "link_feasible_matrix", "optimistic_ptx_up_matrix"):
        value = getattr(ctx, name, None)
        if value is None:
            continue
        array = np.asarray(value)
        if array.ndim != 2:
            raise ValueError(f"context {name} must be a matrix")
        if K and array.shape != (K, K):
            raise ValueError(f"context {name} must have shape {(K, K)}")
    neighbor_sets = getattr(ctx, "neighbor_sets", None)
    if neighbor_sets is not None and K and len(neighbor_sets) != K:
        raise ValueError(f"context neighbor_sets must have length {K}")


def validate_solution_structure(solution, ctx) -> None:
    """Fail fast on shape/categorical corruption before normal CV diagnostics."""
    K = int(ctx.num_candidates)
    expected = {
        "x": (K,),
        "y": (K,),
        "c": (K, K),
        "p_tx": (K, K),
    }
    for name, shape in expected.items():
        value = getattr(solution, name, None)
        if value is None or np.asarray(value).shape != shape:
            raise ValueError(f"solution.{name} shape must be {shape}")
    for name in ("x", "y", "c"):
        value = np.asarray(getattr(solution, name))
        if not np.all(np.isfinite(value)) or not np.all(np.isin(value, (0, 1))):
            raise ValueError(f"solution.{name} must contain only finite binary values")
    for name in ("z_sink_sensor", "z_sink_ap"):
        positions = getattr(solution, name, None)
        if positions is None or len(positions) != K:
            raise ValueError(f"solution.{name} must have {K} owner lists")
        for owner_positions in positions:
            if owner_positions is None:
                raise ValueError(f"solution.{name} contains a null owner list")
            for raw_grid in owner_positions:
                try:
                    int(raw_grid)
                except (TypeError, ValueError, OverflowError):
                    raise ValueError(f"solution.{name} contains a non-integer sink claim")


def sync_state_revision(solution) -> int:
    """Detect uninstrumented edits while hashing only authoritative fields."""
    fingerprint = _solution_fingerprint(solution)
    previous = getattr(solution, "_physical_fingerprint", None)
    if previous is not None and previous != fingerprint:
        solution.state_revision = int(getattr(solution, "state_revision", 0)) + 1
        solution.constraint_report = None
    solution._physical_fingerprint = fingerprint
    return int(getattr(solution, "state_revision", 0))


def mark_physical_state_dirty(solution) -> None:
    solution.state_revision = int(getattr(solution, "state_revision", 0)) + 1
    solution.constraint_report = None
    solution._physical_fingerprint = _solution_fingerprint(solution)


def derive_consumption_snapshot(solution, ctx) -> ConsumptionSnapshot:
    """Derive power from raw x/y/c/p_tx, never from cached consumption arrays."""
    K = int(ctx.num_candidates)
    sensor_cfg = (ctx.config or {}).get("sensor", {})
    ap_cfg = (ctx.config or {}).get("ap", {})
    sensor_fixed = float(sensor_cfg.get("P_sens", 0.01)) + float(
        sensor_cfg.get("P_proc", 0.005)
    )
    ap_fixed = float(ap_cfg.get("P_idle", 0.05)) + float(
        ap_cfg.get("P_proc", 0.02)
    )
    p_rx = float(ap_cfg.get("P_rx", 0.003))
    sensor = np.zeros(K, dtype=float)
    ap = np.zeros(K, dtype=float)
    for sensor_id in range(K):
        if solution.x[sensor_id] != 1:
            continue
        ptx_sum = 0.0
        for ap_id in range(K):
            if solution.c[sensor_id, ap_id] != 1:
                continue
            value = float(solution.p_tx[sensor_id, ap_id])
            if math.isfinite(value):
                ptx_sum += max(0.0, value)
        sensor[sensor_id] = sensor_fixed + ptx_sum
    for ap_id in range(K):
        if solution.y[ap_id] != 1:
            continue
        connections = int(
            sum(
                solution.c[sensor_id, ap_id] == 1
                and solution.x[sensor_id] == 1
                and solution.y[ap_id] == 1
                for sensor_id in range(K)
            )
        )
        ap[ap_id] = ap_fixed + p_rx * connections
    return ConsumptionSnapshot(tuple(sensor.tolist()), tuple(ap.tolist()))


def energy_tolerance(consumption: float, spec: ConstraintEvaluationSpec) -> float:
    return float(spec.energy_abs_tol + spec.energy_rel_tol * max(1.0, abs(float(consumption))))


def required_sink_count_from_power(
    consumption: float, p_grid: float, spec: ConstraintEvaluationSpec
) -> tuple[int, bool]:
    """Return non-negative demand and a separate zero-harvest impossibility flag."""
    consumption = max(0.0, float(consumption))
    tolerance = energy_tolerance(consumption, spec)
    if p_grid > 0.0 and math.isfinite(p_grid):
        return int(math.ceil(max(0.0, consumption - tolerance) / float(p_grid))), False
    if p_grid == 0.0 and consumption <= tolerance:
        return 0, False
    return 0, True


def derive_sink_requirement_snapshot(
    consumption: ConsumptionSnapshot, solution, ctx, spec: ConstraintEvaluationSpec
) -> SinkRequirementSnapshot:
    p_grid = np.asarray(ctx.P_grid, dtype=float)
    required_sensor = []
    required_ap = []
    unachievable_sensor = []
    unachievable_ap = []
    for value, deployed in zip(consumption.sensor, solution.x):
        required, impossible = required_sink_count_from_power(
            value if deployed == 1 else 0.0, float(p_grid[len(required_sensor)]), spec
        )
        required_sensor.append(required if deployed == 1 else 0)
        unachievable_sensor.append(bool(impossible and deployed == 1))
    for value, deployed in zip(consumption.ap, solution.y):
        required, impossible = required_sink_count_from_power(
            value if deployed == 1 else 0.0, float(p_grid[len(required_ap)]), spec
        )
        required_ap.append(required if deployed == 1 else 0)
        unachievable_ap.append(bool(impossible and deployed == 1))
    return SinkRequirementSnapshot(
        tuple(required_sensor),
        tuple(required_ap),
        tuple(unachievable_sensor),
        tuple(unachievable_ap),
    )


def derive_energy_snapshot(
    solution, ctx, consumption: ConsumptionSnapshot, ownership: SinkOwnership, spec: ConstraintEvaluationSpec
) -> EnergySnapshot:
    p_grid = np.asarray(ctx.P_grid, dtype=float)
    sensor_harvest = []
    ap_harvest = []
    sensor_physical = []
    ap_physical = []
    sensor_violation = []
    ap_violation = []
    for index, value in enumerate(consumption.sensor):
        harvest = (
            len(ownership.effective_sensor_positions[index]) * float(p_grid[index])
            if solution.x[index] == 1
            else 0.0
        )
        deficit = max(0.0, float(value) - harvest)
        sensor_harvest.append(float(harvest))
        sensor_physical.append(deficit)
        sensor_violation.append(max(0.0, deficit - energy_tolerance(value, spec)))
    for index, value in enumerate(consumption.ap):
        harvest = (
            len(ownership.effective_ap_positions[index]) * float(p_grid[index])
            if solution.y[index] == 1
            else 0.0
        )
        deficit = max(0.0, float(value) - harvest)
        ap_harvest.append(float(harvest))
        ap_physical.append(deficit)
        ap_violation.append(max(0.0, deficit - energy_tolerance(value, spec)))
    return EnergySnapshot(
        tuple(sensor_harvest),
        tuple(ap_harvest),
        tuple(sensor_physical),
        tuple(ap_physical),
        tuple(sensor_violation),
        tuple(ap_violation),
        sum(value > 0.0 for value in sensor_physical),
        sum(value > 0.0 for value in ap_physical),
    )


def evaluate_constraints(solution, ctx) -> ConstraintReport:
    """Build a pure report from the authoritative solution and context."""
    validate_solution_structure(solution, ctx)
    spec = ensure_constraint_spec(ctx)
    state_revision = sync_state_revision(solution)
    context_signature = getattr(ctx, "constraint_context_signature", None)
    if not context_signature:
        context_signature = build_context_signature(ctx, spec)
        setattr(ctx, "constraint_context_signature", context_signature)
    cached = getattr(solution, "constraint_report", None)
    if (
        isinstance(cached, ConstraintReport)
        and cached.schema_version == CV_SCHEMA_VERSION
        and cached.state_revision == state_revision
        and cached.context_signature == context_signature
    ):
        return cached
    ownership = build_sink_ownership(solution, ctx)
    consumption = derive_consumption_snapshot(solution, ctx)
    requirements = derive_sink_requirement_snapshot(consumption, solution, ctx, spec)
    energy = derive_energy_snapshot(solution, ctx, consumption, ownership, spec)
    physics = PhysicalStateSnapshot(consumption, energy, requirements)
    raw = _raw_breakdown(solution, ctx, spec, ownership, requirements, energy)
    components, energy_cv = _cv_components(solution, ctx, spec, raw, physics)
    report = build_constraint_report(
        raw,
        components,
        physics,
        spec.cv_weights,
        state_revision,
        context_signature,
        energy_cv=energy_cv,
        spec=spec,
    )
    solution.constraint_report = report
    return report


def build_constraint_report(
    raw: RawViolationBreakdown,
    components: CVComponents,
    physics: PhysicalStateSnapshot,
    weights: CVComponents | dict[str, float],
    state_revision: int,
    context_signature: str,
    *,
    energy_cv: EnergyCVBreakdown | None = None,
    spec: ConstraintEvaluationSpec | None = None,
) -> ConstraintReport:
    """Create one validated report; all callers use this factory."""
    if energy_cv is None:
        energy_cv = EnergyCVBreakdown(components.energy, components.energy, components.energy)
    if abs(float(components.energy) - float(energy_cv.aggregate)) > 1.0e-12:
        raise ValueError("components.energy must equal energy_cv.aggregate")
    values = components.as_dict()
    if not all(math.isfinite(value) and value >= 0.0 for value in values.values()):
        raise ValueError("CV components must be finite and non-negative")
    weight_values = (
        weights.as_dict() if isinstance(weights, CVComponents) else dict(weights)
    )
    denominator = sum(float(weight_values[key]) for key in values)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("CV weights must have a positive finite sum")
    cv_total = sum(float(weight_values[key]) * value for key, value in values.items()) / denominator
    if spec is None:
        hard_feasible = _hard_feasible_without_spec(raw, physics)
    else:
        hard_feasible = _hard_feasible(raw, physics, spec)
    feasible = bool(hard_feasible)
    violated = tuple(key for key, value in values.items() if value > 0.0)
    return ConstraintReport(
        schema_version=CV_SCHEMA_VERSION,
        state_revision=int(state_revision),
        context_signature=str(context_signature),
        components=components,
        energy_cv=energy_cv,
        cv_total=float(cv_total),
        feasible=feasible,
        violated_components=violated,
        max_component_cv=float(max(values.values(), default=0.0)),
        raw=raw,
        physics=physics,
    )


def _cv_components(solution, ctx, spec, raw, physics):
    K = max(1, int(ctx.num_candidates))
    active_sensors = int(np.sum(solution.x == 1))
    active_aps = int(np.sum(solution.y == 1))
    deploy_sensor = max(0, active_sensors - spec.max_sensors) / max(1, spec.max_sensors)
    deploy_ap = max(0, active_aps - spec.max_aps) / max(1, spec.max_aps)
    deploy = (deploy_sensor + deploy_ap + raw.role_overlap_count / K) / 3.0

    declared, endpoint, physical = _declared_edge_counts(solution, ctx, spec)
    degree = raw.connection_degree_mismatch / max(1, active_sensors)
    endpoint_cv = abs(declared - endpoint) / max(1, declared)
    physical_cv = abs(endpoint - physical) / max(1, endpoint)
    link = (degree + endpoint_cv + physical_cv) / 3.0

    power_cv = _power_cv(solution, ctx, spec, raw, physical)
    service = (
        raw.empty_ap_count / max(1, active_aps)
        if spec.ap_service_enabled
        else 0.0
    )
    sink_denominator = max(
        1,
        sum(len(values) for values in solution.z_sink_sensor)
        + sum(len(values) for values in solution.z_sink_ap),
    )
    sink = raw.sink_hard.total / (6.0 * sink_denominator)
    sensor_values = [
        physics.energy.sensor_deficit_violation[i] / max(physics.consumption.sensor[i], 1.0e-12)
        for i in range(len(solution.x))
        if solution.x[i] == 1
    ]
    ap_values = [
        physics.energy.ap_deficit_violation[i] / max(physics.consumption.ap[i], 1.0e-12)
        for i in range(len(solution.y))
        if solution.y[i] == 1
    ]
    sensor_energy = float(np.mean(sensor_values)) if sensor_values else 0.0
    ap_energy = float(np.mean(ap_values)) if ap_values else 0.0
    aggregate = (
        (sensor_energy + ap_energy) / 2.0
        if sensor_values and ap_values
        else sensor_energy if sensor_values else ap_energy
    )
    components = CVComponents(deploy, link, power_cv, service, sink, aggregate)
    return components, EnergyCVBreakdown(sensor_energy, ap_energy, aggregate)


def _power_cv(solution, ctx, spec, raw, physical_edges):
    terms = []
    for sensor_id in range(ctx.num_candidates):
        for ap_id in range(ctx.num_candidates):
            if solution.c[sensor_id, ap_id] != 1:
                continue
            if not _physical_link(solution, ctx, sensor_id, ap_id, spec):
                continue
            value = float(solution.p_tx[sensor_id, ap_id])
            if not math.isfinite(value) or value < 0.0:
                continue
            pmin = float(ctx.ptx_min_matrix[sensor_id, ap_id])
            lower = max(0.0, pmin - value - spec.power_abs_tol - spec.power_rel_tol * max(1.0, abs(pmin)))
            edge_pmax = _edge_ptx_max(ctx, sensor_id, ap_id, spec)
            upper = max(0.0, value - edge_pmax - spec.power_abs_tol - spec.power_rel_tol * max(1.0, abs(edge_pmax)))
            terms.append((lower + upper) / max(edge_pmax, 1.0e-12))
    p_bound = float(np.mean(terms)) if terms else 0.0
    active_sensors = max(1, int(np.sum(solution.x == 1)))
    p_state = raw.power_state_invalid_sensor_count / active_sensors
    return (p_bound + p_state) / 2.0


def _raw_breakdown(solution, ctx, spec, ownership, requirements, energy):
    declared, endpoint, physical = _declared_edge_counts(solution, ctx, spec)
    nonfinite = negative = inactive_nonzero = 0
    lower_physical = lower_violation = upper_physical = upper_violation = 0.0
    invalid_sensor_rows = 0
    for sensor_id in range(ctx.num_candidates):
        row_invalid = False
        for ap_id in range(ctx.num_candidates):
            value = float(solution.p_tx[sensor_id, ap_id])
            active_edge = solution.c[sensor_id, ap_id] == 1
            if active_edge and not math.isfinite(value):
                nonfinite += 1
                row_invalid = True
            elif active_edge and value < 0.0:
                negative += 1
                row_invalid = True
            elif not active_edge and math.isfinite(value) and abs(value) > spec.numeric_zero_tol:
                inactive_nonzero += 1
                if solution.x[sensor_id] == 1:
                    row_invalid = True
            if not active_edge or not _physical_link(solution, ctx, sensor_id, ap_id, spec):
                continue
            if not math.isfinite(value):
                continue
            pmin = float(ctx.ptx_min_matrix[sensor_id, ap_id])
            lower_physical += max(0.0, pmin - value)
            lower_violation += max(
                0.0,
                pmin - value - spec.power_abs_tol - spec.power_rel_tol * max(1.0, abs(pmin)),
            )
            edge_pmax = _edge_ptx_max(ctx, sensor_id, ap_id, spec)
            upper_physical += max(0.0, value - edge_pmax)
            upper_violation += max(
                0.0,
                value - edge_pmax - spec.power_abs_tol - spec.power_rel_tol * max(1.0, abs(edge_pmax)),
            )
        if solution.x[sensor_id] == 1 and row_invalid:
            invalid_sensor_rows += 1
    sink_hard, sink_diag = _classify_sink_claims(solution, ctx, requirements, ownership)
    empty_ap = int(
        sum(solution.y[ap_id] == 1 and not np.any((solution.c[:, ap_id] == 1) & (solution.x == 1)) for ap_id in range(ctx.num_candidates))
    )
    raw = RawViolationBreakdown(
        deploy_sensor_excess=max(0, int(np.sum(solution.x)) - spec.max_sensors),
        deploy_ap_excess=max(0, int(np.sum(solution.y)) - spec.max_aps),
        role_overlap_count=int(np.sum((solution.x == 1) & (solution.y == 1))),
        invalid_endpoint_link_count=int(declared - endpoint),
        physical_infeasible_link_count=int(endpoint - physical),
        connection_degree_mismatch=_connection_degree_mismatch(solution),
        nonfinite_power_edge_count=nonfinite,
        negative_power_edge_count=negative,
        inactive_nonzero_power_edge_count=inactive_nonzero,
        power_state_invalid_sensor_count=invalid_sensor_rows,
        power_lower_deficit_watt_physical=float(lower_physical),
        power_lower_violation_watt=float(lower_violation),
        power_upper_excess_watt_physical=float(upper_physical),
        power_upper_violation_watt=float(upper_violation),
        empty_ap_count=empty_ap,
        sink_hard=sink_hard,
        sink_diagnostics=sink_diag,
    )
    return raw


def _declared_edge_counts(solution, ctx, spec):
    declared = endpoint = physical = 0
    for sensor_id in range(ctx.num_candidates):
        for ap_id in range(ctx.num_candidates):
            if solution.c[sensor_id, ap_id] != 1:
                continue
            declared += 1
            if solution.x[sensor_id] == 1 and solution.y[ap_id] == 1:
                endpoint += 1
                if _physical_link(solution, ctx, sensor_id, ap_id, spec):
                    physical += 1
    return declared, endpoint, physical


def _physical_link(solution, ctx, sensor_id, ap_id, spec):
    pmin = float(ctx.ptx_min_matrix[sensor_id, ap_id])
    edge_pmax = _edge_ptx_max(ctx, sensor_id, ap_id, spec)
    return bool(
        solution.x[sensor_id] == 1
        and solution.y[ap_id] == 1
        and ctx.link_feasible_matrix[sensor_id, ap_id] == 1
        and math.isfinite(pmin)
        and pmin <= edge_pmax + spec.power_abs_tol
    )


def _connection_degree_mismatch(solution):
    total = 0
    for sensor_id in np.where(solution.x == 1)[0]:
        degree = int(np.sum(solution.c[sensor_id] == 1))
        if degree == 0:
            total += 1
        elif degree > 1:
            total += degree - 1
    return int(total)


def edge_ptx_max(ctx, sensor_id, ap_id, fallback=None):
    matrix = getattr(ctx, "optimistic_ptx_up_matrix", None)
    if matrix is not None:
        array = np.asarray(matrix)
        if array.ndim == 2 and sensor_id < array.shape[0] and ap_id < array.shape[1]:
            value = float(array[sensor_id, ap_id])
            if math.isfinite(value) and value > 0.0:
                return value
    if fallback is None:
        config = getattr(ctx, "config", None) or {}
        fallback = (config.get("channel", {}) or {}).get("p_tx_max", 0.5)
    return float(fallback)


def _edge_ptx_max(ctx, sensor_id, ap_id, spec):
    return edge_ptx_max(ctx, sensor_id, ap_id, fallback=spec.ptx_max)


def _classify_sink_claims(solution, ctx, requirements, ownership):
    rules = SinkOverlapRules(ctx.config)
    counts = {
        "invalid": 0,
        "undeployed": 0,
        "outside": 0,
        "overlap": 0,
        "duplicate": 0,
        "cross": 0,
    }
    legal_claims: list[tuple[str, int, int]] = []
    all_claims = [
        ("sensor", owner, raw_grid)
        for owner, positions in enumerate(solution.z_sink_sensor)
        for raw_grid in positions
    ] + [
        ("ap", owner, raw_grid)
        for owner, positions in enumerate(solution.z_sink_ap)
        for raw_grid in positions
    ]
    for kind, owner, raw_grid in all_claims:
        try:
            grid = int(raw_grid)
        except (TypeError, ValueError, OverflowError):
            counts["invalid"] += 1
            continue
        deployed = solution.x[owner] == 1 if kind == "sensor" else solution.y[owner] == 1
        if not deployed:
            counts["undeployed"] += 1
            continue
        if grid < 0 or grid >= ctx.num_candidates:
            counts["invalid"] += 1
            continue
        if not rules.allow_sink_outside_neighborhood and grid not in {
            int(value) for value in ctx.neighbor_sets[owner]
        }:
            counts["outside"] += 1
            continue
        if grid == owner and not rules.count_self_grid_as_sink:
            counts["overlap"] += 1
            continue
        if not rules.allow_sink_on_other_node:
            occupied = solution.x[grid] == 1 or solution.y[grid] == 1
            own_node = grid == owner and deployed
            if occupied and not (own_node and rules.allow_self_node_sink_overlap):
                counts["overlap"] += 1
                continue
        legal_claims.append((kind, int(owner), grid))

    by_owner_grid: dict[tuple[str, int, int], int] = {}
    for kind, owner, grid in legal_claims:
        key = (kind, owner, grid)
        by_owner_grid[key] = by_owner_grid.get(key, 0) + 1
    counts["duplicate"] = sum(max(0, value - 1) for value in by_owner_grid.values())
    if not rules.allow_sink_sink_overlap:
        by_grid: dict[int, set[tuple[str, int]]] = {}
        for (kind, owner, grid), _count in by_owner_grid.items():
            by_grid.setdefault(grid, set()).add((kind, owner))
        counts["cross"] = sum(max(0, len(owners) - 1) for owners in by_grid.values())

    shortage_sensor = 0
    shortage_ap = 0
    effective_sensor_total = sum(ownership.effective_sensor_count)
    effective_ap_total = sum(ownership.effective_ap_count)
    for index, required in enumerate(requirements.required_sensor):
        shortage_sensor += max(0, int(required) - ownership.effective_sensor_count[index])
    for index, required in enumerate(requirements.required_ap):
        shortage_ap += max(0, int(required) - ownership.effective_ap_count[index])
    diagnostics = SinkDiagnostics(
        shortage_sensor=int(shortage_sensor),
        shortage_ap=int(shortage_ap),
        required_sensor_total=int(sum(requirements.required_sensor)),
        required_ap_total=int(sum(requirements.required_ap)),
        effective_sensor_total=int(effective_sensor_total),
        effective_ap_total=int(effective_ap_total),
        unachievable_sensor_count=int(sum(requirements.unachievable_sensor)),
        unachievable_ap_count=int(sum(requirements.unachievable_ap)),
    )
    return SinkHardViolations(
        counts["invalid"],
        counts["undeployed"],
        counts["outside"],
        counts["overlap"],
        counts["duplicate"],
        counts["cross"],
    ), diagnostics


def _hard_feasible(raw, physics, spec):
    sensor_energy_ok = all(value <= 0.0 for value in physics.energy.sensor_deficit_violation)
    ap_energy_ok = all(value <= 0.0 for value in physics.energy.ap_deficit_violation)
    no_invalid_power = (
        raw.nonfinite_power_edge_count == 0
        and raw.negative_power_edge_count == 0
        and raw.inactive_nonzero_power_edge_count == 0
        and raw.power_lower_violation_watt <= spec.numeric_zero_tol
        and raw.power_upper_violation_watt <= spec.numeric_zero_tol
        and raw.power_state_invalid_sensor_count == 0
    )
    return bool(
        raw.deploy_sensor_excess == 0
        and raw.deploy_ap_excess == 0
        and raw.role_overlap_count == 0
        and raw.invalid_endpoint_link_count == 0
        and raw.physical_infeasible_link_count == 0
        and raw.connection_degree_mismatch == 0
        and no_invalid_power
        and (not spec.ap_service_enabled or raw.empty_ap_count == 0)
        and raw.sink_hard.satisfied
        and not any(physics.sink_requirement.unachievable_sensor)
        and not any(physics.sink_requirement.unachievable_ap)
        and sensor_energy_ok
        and ap_energy_ok
    )


def _hard_feasible_without_spec(raw, physics):
    return bool(
        raw.deploy_sensor_excess == 0
        and raw.deploy_ap_excess == 0
        and raw.role_overlap_count == 0
        and raw.invalid_endpoint_link_count == 0
        and raw.physical_infeasible_link_count == 0
        and raw.connection_degree_mismatch == 0
        and raw.sink_hard.satisfied
        and all(value <= 0.0 for value in physics.energy.sensor_deficit_violation)
        and all(value <= 0.0 for value in physics.energy.ap_deficit_violation)
    )


def _array_digest(value):
    if value is None:
        return None
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _solution_fingerprint(solution):
    payload = {
        "x": _array_digest(getattr(solution, "x", None)),
        "y": _array_digest(getattr(solution, "y", None)),
        "c": _array_digest(getattr(solution, "c", None)),
        "p_tx": _array_digest(getattr(solution, "p_tx", None)),
        "z_sensor": [tuple(int(value) for value in positions) for positions in getattr(solution, "z_sink_sensor", [])],
        "z_ap": [tuple(int(value) for value in positions) for positions in getattr(solution, "z_sink_ap", [])],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def compare_constraint_reports(a: ConstraintReport, b: ConstraintReport, cv_compare_tol: float = 1.0e-8) -> int:
    """Return 1 when a is better, -1 when b is better, 0 for a tie."""
    if a.feasible != b.feasible:
        return 1 if a.feasible else -1
    if not a.feasible:
        for av, bv in (
            (a.cv_total, b.cv_total),
            (a.max_component_cv, b.max_component_cv),
            (len(a.violated_components), len(b.violated_components)),
        ):
            if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
                if av < bv - cv_compare_tol:
                    return 1
                if bv < av - cv_compare_tol:
                    return -1
            elif av != bv:
                return 1 if av < bv else -1
        return 0
    if a.cv_total < b.cv_total - cv_compare_tol:
        return 1
    if b.cv_total < a.cv_total - cv_compare_tol:
        return -1
    return 0


__all__ = [
    "AUTHORITATIVE_FIELDS_VERSION",
    "CVComponents",
    "CV_SCHEMA_VERSION",
    "COMPARATOR_VERSION",
    "ConsumptionSnapshot",
    "ConstraintEvaluationSpec",
    "ConstraintReport",
    "EnergyCVBreakdown",
    "EnergySnapshot",
    "PhysicalStateSnapshot",
    "RawViolationBreakdown",
    "SinkDiagnostics",
    "SinkHardViolations",
    "SinkRequirementSnapshot",
    "edge_ptx_max",
    "build_constraint_report",
    "build_context_signature",
    "compare_constraint_reports",
    "derive_consumption_snapshot",
    "derive_energy_snapshot",
    "derive_sink_requirement_snapshot",
    "energy_tolerance",
    "ensure_constraint_spec",
    "evaluate_constraints",
    "mark_physical_state_dirty",
    "mark_context_dirty",
    "required_sink_count_from_power",
    "sync_state_revision",
    "validate_context_physics",
    "validate_solution_structure",
]
