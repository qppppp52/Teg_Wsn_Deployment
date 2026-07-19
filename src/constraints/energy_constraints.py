"""Energy-neutrality checks and bounded repair stabilization."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.constraints.link_constraints import single_valid_connected_ap
from src.heatsink.harvest_power import (
    ap_harvest_power,
    refresh_harvest_diagnostics,
    sensor_harvest_power,
)
from src.heatsink.sink_allocator import rebuild_minimum_sink_allocation
from src.heatsink.sink_ownership import build_sink_ownership
from src.heatsink.sink_requirement import (
    InvalidHarvestPower,
    compute_sink_requirements,
)
from src.physics.numerical_tolerances import ENERGY_ABS_TOL
from src.power.power_repair import repair_power


@dataclass(frozen=True)
class EnergyStabilizationResult:
    iterations: int
    stop_reason: str


def check_energy_constraints(solution, ctx):
    """Compute energy CV without mutating any physical solution field."""
    ownership = build_sink_ownership(solution, ctx)
    cv = 0.0
    for sensor_id in np.where(solution.x == 1)[0]:
        p_grid = float(ctx.P_grid[sensor_id])
        if p_grid <= 0.0:
            raise InvalidHarvestPower("P_grid must be positive for a deployed node")
        harvested = sensor_harvest_power(solution, sensor_id, ctx, ownership)
        consumption = float(solution.sensor_power_consumption[sensor_id])
        if harvested + ENERGY_ABS_TOL < consumption:
            cv += (consumption - harvested) / p_grid
    for ap_id in np.where(solution.y == 1)[0]:
        p_grid = float(ctx.P_grid[ap_id])
        if p_grid <= 0.0:
            raise InvalidHarvestPower("P_grid must be positive for a deployed node")
        harvested = ap_harvest_power(solution, ap_id, ctx, ownership)
        consumption = float(solution.ap_power_consumption[ap_id])
        if harvested + ENERGY_ABS_TOL < consumption:
            cv += (consumption - harvested) / p_grid
    return float(cv)


def stabilize_energy_sink_state(solution, ctx):
    """Settle demand, minimum allocation and downward-only power repair."""
    configured = int(ctx.config.get("constraints", {}).get("max_repair_iter", 5))
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
            stop_reason = "cycle" if seen else "fixed_point"
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
    """Stabilize first, then let this outer repair operator remove impossible nodes."""
    stabilize_energy_sink_state(solution, ctx)

    for sensor_id in list(np.where(solution.x == 1)[0]):
        ap_id = single_valid_connected_ap(solution, sensor_id, ctx)
        if ap_id is None:
            continue
        ownership = build_sink_ownership(solution, ctx)
        harvested = sensor_harvest_power(solution, sensor_id, ctx, ownership)
        if harvested + ENERGY_ABS_TOL < solution.sensor_power_consumption[sensor_id]:
            _remove_sensor(solution, sensor_id)
            stabilize_energy_sink_state(solution, ctx)

    for ap_id in list(np.where(solution.y == 1)[0]):
        connected = np.where(solution.c[:, ap_id] == 1)[0]
        if any(single_valid_connected_ap(solution, sensor_id, ctx) != ap_id for sensor_id in connected):
            continue
        ownership = build_sink_ownership(solution, ctx)
        harvested = ap_harvest_power(solution, ap_id, ctx, ownership)
        if harvested + ENERGY_ABS_TOL < solution.ap_power_consumption[ap_id]:
            _remove_ap(solution, ap_id)
            stabilize_energy_sink_state(solution, ctx)
    return solution


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


def _remove_ap(solution, ap_id):
    solution.y[ap_id] = 0
    solution.c[:, ap_id] = 0
    solution.p_tx[:, ap_id] = 0.0
    solution.n_sink_ap[ap_id] = 0
    solution.z_sink_ap[ap_id] = []
    solution.ap_power_consumption[ap_id] = 0.0
    solution.ap_harvest_power[ap_id] = 0.0
