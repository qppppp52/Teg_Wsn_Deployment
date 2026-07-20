"""Deterministic downward-only transmit-power repair."""
from __future__ import annotations

import numpy as np

from src.constraints.link_constraints import single_valid_connected_ap
from src.constraints.constraint_report import edge_ptx_max
from src.constraints.constraint_report import mark_physical_state_dirty
from src.heatsink.harvest_power import refresh_harvest_diagnostics
from src.physics.node_power_model import compute_ap_consumption, compute_sensor_consumption
from src.physics.numerical_tolerances import POWER_ABS_TOL
from src.power.power_bounds import max_energy_feasible_ptx


POWER_REPAIR_SEMANTICS_VERSION = 3


def repair_power(solution, ctx):
    """Retain supported current power and downshift only when energy requires it."""
    previous_power = solution.p_tx.copy()
    solution.p_tx.fill(0.0)
    ownership = refresh_harvest_diagnostics(solution, ctx)

    for sensor_id in range(ctx.num_candidates):
        ap_id = single_valid_connected_ap(solution, sensor_id, ctx)
        if ap_id is None:
            solution.sensor_power_consumption[sensor_id] = (
                compute_sensor_consumption(sensor_id, 0.0, ctx.config)
                if solution.x[sensor_id] == 1
                else 0.0
            )
            continue
        ptx_min = float(ctx.ptx_min_matrix[sensor_id, ap_id])
        current = float(previous_power[sensor_id, ap_id])
        if not np.isfinite(current) or current <= POWER_ABS_TOL:
            current = ptx_min
        upper = max_energy_feasible_ptx(
            solution, sensor_id, ap_id, ctx, ownership=ownership
        )
        target = max(ptx_min, min(current, upper))
        target = min(edge_ptx_max(ctx, sensor_id, ap_id), target)
        solution.p_tx[sensor_id, ap_id] = target
        solution.sensor_power_consumption[sensor_id] = compute_sensor_consumption(
            sensor_id, target, ctx.config
        )

    for ap_id in range(ctx.num_candidates):
        if solution.y[ap_id] == 1:
            connections = int(
                np.sum((solution.c[:, ap_id] == 1) & (solution.x == 1))
            )
            solution.ap_power_consumption[ap_id] = compute_ap_consumption(
                connections, ctx.config
            )
        else:
            solution.ap_power_consumption[ap_id] = 0.0
    if not np.array_equal(previous_power, solution.p_tx):
        mark_physical_state_dirty(solution)
    return solution