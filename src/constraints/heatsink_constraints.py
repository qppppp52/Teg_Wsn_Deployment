"""Heatsink constraints from one effective-ownership interpretation."""
from __future__ import annotations

from src.heatsink.sink_ownership import build_sink_ownership


def heatsink_constraint_components(solution, ctx):
    ownership = build_sink_ownership(solution, ctx)
    duplicate_cv = float(
        sum(ownership.duplicate_sensor_count)
        + sum(ownership.duplicate_ap_count)
    )
    invalid_cv = float(
        sum(ownership.invalid_sensor_count)
        + sum(ownership.invalid_ap_count)
    )
    conflict_cv = float(ownership.forbidden_conflict_count)
    shortage_cv = 0.0

    for sensor_id in range(ctx.num_candidates):
        if solution.x[sensor_id] == 1:
            effective = len(ownership.effective_sensor_positions[sensor_id])
            shortage_cv += max(
                0, int(solution.n_sink_sensor[sensor_id]) - effective
            )
    for ap_id in range(ctx.num_candidates):
        if solution.y[ap_id] == 1:
            effective = len(ownership.effective_ap_positions[ap_id])
            shortage_cv += max(0, int(solution.n_sink_ap[ap_id]) - effective)

    return {
        "duplicate_cv": duplicate_cv,
        "invalid_cv": invalid_cv,
        "sink_conflict_cv": conflict_cv,
        "sink_shortage_cv": float(shortage_cv),
        "total": duplicate_cv + invalid_cv + conflict_cv + float(shortage_cv),
    }


def check_heatsink_constraints(solution, ctx):
    """Return total sink CV without modifying the solution."""
    return heatsink_constraint_components(solution, ctx)["total"]
