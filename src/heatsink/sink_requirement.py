"""Minimum theoretical heatsink demand from current node consumption."""
from __future__ import annotations

import math

from src.constraints.link_constraints import single_valid_connected_ap
from src.physics.node_power_model import compute_ap_consumption, compute_sensor_consumption
from src.physics.numerical_tolerances import ENERGY_ABS_TOL


class InvalidHarvestPower(ValueError):
    pass


def required_sink_count(consumption: float, p_grid: float) -> int:
    """Return the untruncated minimum theoretical sink demand."""
    p_grid = float(p_grid)
    if p_grid <= 0.0:
        raise InvalidHarvestPower("P_grid must be positive for a deployed node")
    adjusted = max(0.0, float(consumption) - ENERGY_ABS_TOL)
    return int(math.ceil(adjusted / p_grid))


def compute_sink_requirements(solution, ctx):
    K = ctx.num_candidates
    for sensor_id in range(K):
        if solution.x[sensor_id] == 1:
            ap_id = single_valid_connected_ap(solution, sensor_id, ctx)
            p_tx = solution.p_tx[sensor_id, ap_id] if ap_id is not None else 0.0
            consumption = compute_sensor_consumption(sensor_id, p_tx, ctx.config)
            solution.sensor_power_consumption[sensor_id] = consumption
            solution.n_sink_sensor[sensor_id] = required_sink_count(
                consumption, ctx.P_grid[sensor_id]
            )
        else:
            solution.n_sink_sensor[sensor_id] = 0
            solution.sensor_power_consumption[sensor_id] = 0.0

    for ap_id in range(K):
        if solution.y[ap_id] == 1:
            connection_count = int(solution.c[:, ap_id].sum())
            consumption = compute_ap_consumption(connection_count, ctx.config)
            solution.ap_power_consumption[ap_id] = consumption
            solution.n_sink_ap[ap_id] = required_sink_count(
                consumption, ctx.P_grid[ap_id]
            )
        else:
            solution.n_sink_ap[ap_id] = 0
            solution.ap_power_consumption[ap_id] = 0.0
