"""Energy-neutrality checks and bounded physical repair."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.constraints.constraint_report import (
    global_ptx_max,
    ensure_constraint_spec,
    evaluate_constraints,
    mark_physical_state_dirty,
)
from src.constraints.constraint_comparator import is_repair_candidate_better
from src.constraints.link_constraints import single_valid_connected_ap
from src.constraints.repair_config import resolve_repair_config
from src.heatsink.harvest_power import refresh_harvest_diagnostics
from src.heatsink.sink_allocator import rebuild_minimum_sink_allocation
from src.heatsink.sink_requirement import compute_sink_requirements
from src.physics.numerical_tolerances import POWER_ABS_TOL
from src.power.power_repair import repair_power


ENERGY_REPAIR_SEMANTICS_VERSION = 3


@dataclass(frozen=True)
class EnergyStabilizationResult:
    iterations: int
    stop_reason: str


def check_energy_constraints(solution, ctx):
    """Return role-balanced normalized energy CV without mutating the solution."""
    return float(evaluate_constraints(solution, ctx).components.energy)


def stabilize_energy_sink_state(solution, ctx):
    """Settle demand, deterministic minimum allocation and downward-only power repair."""
    configured = resolve_repair_config(ctx.config).max_energy_stabilization_iters
    active_count = int(solution.x.sum() + solution.y.sum())
    inner_limit = max(1, min(max(configured, 1), 1 + active_count))
    seen = set()
    stop_reason = "inner_limit"

    for iteration in range(1, inner_limit + 1):
        compute_sink_requirements(solution, ctx)
        rebuild_minimum_sink_allocation(solution, ctx)
        refresh_harvest_diagnostics(solution, ctx)
        repair_power(solution, ctx)
        compute_sink_requirements(solution, ctx)
        rebuild_minimum_sink_allocation(solution, ctx)
        refresh_harvest_diagnostics(solution, ctx)
        signature = _energy_sink_signature(solution)
        if signature in seen:
            stop_reason = "cycle"
            return EnergyStabilizationResult(iteration, stop_reason)
        seen.add(signature)
        repair_power(solution, ctx)
        compute_sink_requirements(solution, ctx)
        rebuild_minimum_sink_allocation(solution, ctx)
        refresh_harvest_diagnostics(solution, ctx)
        if _energy_sink_signature(solution) == signature:
            return EnergyStabilizationResult(iteration, "fixed_point")
    return EnergyStabilizationResult(inner_limit, stop_reason)


def repair_energy_constraints(solution, ctx):
    """Repair Sensor energy locally before changing AP service topology."""
    stabilize_energy_sink_state(solution, ctx)
    report = evaluate_constraints(solution, ctx)

    # First reduce a Sensor's own minimum TX demand without shifting AP load.
    reassigned, report = _reassign_sensor_energy_deficits(solution, ctx, report)
    if reassigned:
        solution.metadata["sensor_energy_reassignments"] = int(
            solution.metadata.get("sensor_energy_reassignments", 0)
        ) + reassigned

    # This deterministic rebuild is the AP-local, topology-preserving repair.
    stabilize_energy_sink_state(solution, ctx)
    report = evaluate_constraints(solution, ctx)
    _rebalance_ap_energy(solution, ctx, report)
    stabilize_energy_sink_state(solution, ctx)
    report = evaluate_constraints(solution, ctx)

    repair_config = resolve_repair_config(ctx.config)
    solution.metadata["energy_repair_variant"] = (
        "aggressive_energy_node_deletion"
        if repair_config.enable_energy_node_deletion_fallback
        else "baseline_no_energy_node_deletion"
    )
    if repair_config.enable_energy_node_deletion_fallback:
        _apply_energy_deletion_fallback(solution, ctx, report)
    return solution


def _apply_energy_deletion_fallback(solution, ctx, report):
    """Optional ablation only: preserve the historical destructive fallback."""
    for sensor_id in list(np.where(solution.x == 1)[0]):
        if report.physics.energy.sensor_deficit_violation[sensor_id] <= 0.0:
            continue
        _remove_sensor(solution, int(sensor_id))
        stabilize_energy_sink_state(solution, ctx)
        report = evaluate_constraints(solution, ctx)
    for ap_id in list(np.where(solution.y == 1)[0]):
        if report.physics.energy.ap_deficit_violation[ap_id] <= 0.0:
            continue
        _remove_ap(solution, int(ap_id))
        stabilize_energy_sink_state(solution, ctx)
        report = evaluate_constraints(solution, ctx)
    return report


def _reassign_sensor_energy_deficits(solution, ctx, report):
    """Try a physically valid lower-pmin AP before deleting an energy-deficit Sensor."""
    max_moves = resolve_repair_config(ctx.config).max_sensor_energy_reassignments
    if max_moves == 0:
        return 0, report
    spec = ensure_constraint_spec(ctx)
    compare_tol = float(spec.cv_compare_tol)
    energy_tol = float(spec.energy_compare_tol)
    accepted = 0

    while accepted < max_moves:
        best = None
        for sensor_id in np.where(solution.x == 1)[0]:
            sensor_id = int(sensor_id)
            current_deficit = float(report.physics.energy.sensor_deficit_violation[sensor_id])
            current_ap = single_valid_connected_ap(solution, sensor_id, ctx)
            if current_ap is None or current_deficit <= energy_tol:
                continue
            current_pmin = float(ctx.ptx_min_matrix[sensor_id, current_ap])
            for target_ap in np.where(solution.y == 1)[0]:
                target_ap = int(target_ap)
                if target_ap == current_ap or not _link_candidate_is_valid(sensor_id, target_ap, ctx):
                    continue
                target_pmin = float(ctx.ptx_min_matrix[sensor_id, target_ap])
                if target_pmin >= current_pmin - POWER_ABS_TOL:
                    continue
                candidate = solution.copy()
                candidate.c[sensor_id, :] = 0
                candidate.p_tx[sensor_id, :] = 0.0
                candidate.c[sensor_id, target_ap] = 1
                candidate.p_tx[sensor_id, target_ap] = target_pmin
                mark_physical_state_dirty(candidate)
                stabilize_energy_sink_state(candidate, ctx)
                candidate_report = evaluate_constraints(candidate, ctx)
                deficit_after = float(candidate_report.physics.energy.sensor_deficit_violation[sensor_id])
                if deficit_after >= current_deficit - energy_tol:
                    continue
                if not is_repair_candidate_better(candidate_report, report, compare_tol):
                    continue
                rank = (deficit_after, target_pmin, sensor_id, target_ap)
                if best is None or rank < best[0]:
                    best = (rank, candidate, candidate_report)
        if best is None:
            break
        _, candidate, report = best
        _copy_solution_in_place(solution, candidate)
        accepted += 1
    return accepted, report


def _rebalance_ap_energy(solution, ctx, report):
    spec = ensure_constraint_spec(ctx)
    max_moves = resolve_repair_config(ctx.config).max_ap_rebalance_moves
    if max_moves <= 0:
        return 0
    accepted = 0
    compare_tol = float(spec.cv_compare_tol)
    energy_tol = float(spec.energy_compare_tol)
    while accepted < max_moves:
        best = None
        for ap_id in np.where(solution.y == 1)[0]:
            current_residual = float(report.physics.energy.ap_deficit_violation[ap_id])
            if current_residual <= energy_tol:
                continue
            connected = [
                int(sensor_id)
                for sensor_id in np.where(solution.c[:, ap_id] == 1)[0]
                if solution.x[sensor_id] == 1
            ]
            for sensor_id in connected:
                for target_ap in np.where(solution.y == 1)[0]:
                    target_ap = int(target_ap)
                    if target_ap == int(ap_id):
                        continue
                    if not _link_candidate_is_valid(sensor_id, target_ap, ctx):
                        continue
                    candidate = solution.copy()
                    candidate.c[sensor_id, :] = 0
                    candidate.p_tx[sensor_id, :] = 0.0
                    candidate.c[sensor_id, target_ap] = 1
                    candidate.p_tx[sensor_id, target_ap] = float(
                        ctx.ptx_min_matrix[sensor_id, target_ap]
                    )
                    mark_physical_state_dirty(candidate)
                    stabilize_energy_sink_state(candidate, ctx)
                    candidate_report = evaluate_constraints(candidate, ctx)
                    residual_after = float(
                        candidate_report.physics.energy.ap_deficit_violation[int(ap_id)]
                    )
                    if not is_repair_candidate_better(candidate_report, report, compare_tol):
                        continue
                    rank = (
                        residual_after,
                        float(ctx.ptx_min_matrix[sensor_id, target_ap]),
                        sensor_id,
                        target_ap,
                    )
                    if best is None or rank < best[0]:
                        best = (rank, candidate, candidate_report)
        if best is None:
            break
        _, candidate, report = best
        _copy_solution_in_place(solution, candidate)
        accepted += 1
    return accepted


def _link_candidate_is_valid(sensor_id, ap_id, ctx):
    pmin = float(ctx.ptx_min_matrix[sensor_id, ap_id])
    return bool(
        ctx.link_feasible_matrix[sensor_id, ap_id] == 1
        and np.isfinite(pmin)
        and pmin <= global_ptx_max(ctx) + POWER_ABS_TOL
    )


def _copy_solution_in_place(target, source):
    for name in ("x", "y", "c", "p_tx", "n_sink_sensor", "n_sink_ap"):
        getattr(target, name)[:] = getattr(source, name)
    target.z_sink_sensor = [list(values) for values in source.z_sink_sensor]
    target.z_sink_ap = [list(values) for values in source.z_sink_ap]
    for name in (
        "sensor_power_consumption",
        "ap_power_consumption",
        "sensor_harvest_power",
        "ap_harvest_power",
    ):
        getattr(target, name)[:] = getattr(source, name)
    target.metadata = dict(source.metadata)
    mark_physical_state_dirty(target)


def _energy_sink_signature(solution):
    return (
        solution.p_tx.tobytes(),
        solution.n_sink_sensor.tobytes(),
        solution.n_sink_ap.tobytes(),
        tuple(tuple(sorted(int(grid) for grid in positions)) for positions in solution.z_sink_sensor),
        tuple(tuple(sorted(int(grid) for grid in positions)) for positions in solution.z_sink_ap),
    )


def _remove_sensor(solution, sensor_id):
    solution.x[sensor_id] = 0
    solution.c[sensor_id] = 0
    solution.p_tx[sensor_id] = 0.0
    solution.n_sink_sensor[sensor_id] = 0
    solution.z_sink_sensor[sensor_id] = []
    solution.sensor_power_consumption[sensor_id] = 0.0
    solution.sensor_harvest_power[sensor_id] = 0.0
    solution.metadata["forced_sensor_removals"] = int(
        solution.metadata.get("forced_sensor_removals", 0)
    ) + 1
    mark_physical_state_dirty(solution)


def _remove_ap(solution, ap_id):
    solution.y[ap_id] = 0
    solution.c[:, ap_id] = 0
    solution.p_tx[:, ap_id] = 0.0
    solution.n_sink_ap[ap_id] = 0
    solution.z_sink_ap[ap_id] = []
    solution.ap_power_consumption[ap_id] = 0.0
    solution.ap_harvest_power[ap_id] = 0.0
    solution.metadata["forced_ap_removals"] = int(
        solution.metadata.get("forced_ap_removals", 0)
    ) + 1
    mark_physical_state_dirty(solution)
