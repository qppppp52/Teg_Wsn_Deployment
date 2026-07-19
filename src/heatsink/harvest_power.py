"""Actual TEG harvesting derived from effective physical ownership."""
from __future__ import annotations

from src.heatsink.sink_ownership import build_sink_ownership


def actual_sink_count(sink_positions) -> int:
    """Count unique declared sink grids for one node."""
    return len({int(grid_id) for grid_id in sink_positions})


def sensor_harvest_power(solution, sensor_global_id: int, ctx, ownership=None) -> float:
    """Return sensor harvesting power from effective heatsink positions."""
    sensor_global_id = int(sensor_global_id)
    ownership = ownership or build_sink_ownership(solution, ctx)
    return float(
        len(ownership.effective_sensor_positions[sensor_global_id])
        * float(ctx.P_grid[sensor_global_id])
    )


def ap_harvest_power(solution, ap_global_id: int, ctx, ownership=None) -> float:
    """Return AP harvesting power; AP IDs are global candidate-point IDs."""
    ap_global_id = int(ap_global_id)
    ownership = ownership or build_sink_ownership(solution, ctx)
    return float(
        len(ownership.effective_ap_positions[ap_global_id])
        * float(ctx.P_grid[ap_global_id])
    )


def refresh_harvest_diagnostics(solution, ctx, ownership=None):
    """Refresh cached harvest arrays outside pure constraint checks."""
    ownership = ownership or build_sink_ownership(solution, ctx)
    solution.sensor_harvest_power.fill(0.0)
    solution.ap_harvest_power.fill(0.0)
    for sensor_id in range(ctx.num_candidates):
        if solution.x[sensor_id] == 1:
            solution.sensor_harvest_power[sensor_id] = sensor_harvest_power(
                solution, sensor_id, ctx, ownership
            )
    for ap_id in range(ctx.num_candidates):
        if solution.y[ap_id] == 1:
            solution.ap_harvest_power[ap_id] = ap_harvest_power(
                solution, ap_id, ctx, ownership
            )
    return ownership
