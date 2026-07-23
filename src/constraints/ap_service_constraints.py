"""AP service constraints without an AP sensor-count upper bound."""
from __future__ import annotations

import numpy as np

from src.constraints.constraint_report import evaluate_constraints, mark_physical_state_dirty
from src.constraints.link_constraints import single_valid_connected_ap


def check_ap_service(solution, ctx):
    """Return the service component from the authoritative physical report."""
    return float(evaluate_constraints(solution, ctx).components.service)


def repair_empty_aps(solution, ctx):
    """Remove deployed APs that have no physically valid Sensor connection."""
    if not bool(ctx.config.get("constraints", {}).get("ap_service_enabled", True)):
        return solution
    served_ap_ids = {
        ap_id
        for sensor_id in np.where(solution.x == 1)[0]
        if (ap_id := single_valid_connected_ap(solution, int(sensor_id), ctx)) is not None
    }
    changed = False
    for ap_id in np.where(solution.y == 1)[0]:
        ap_id = int(ap_id)
        if ap_id in served_ap_ids:
            continue
        solution.y[ap_id] = 0
        solution.c[:, ap_id] = 0
        solution.p_tx[:, ap_id] = 0.0
        solution.n_sink_ap[ap_id] = 0
        solution.z_sink_ap[ap_id] = []
        solution.ap_power_consumption[ap_id] = 0.0
        solution.ap_harvest_power[ap_id] = 0.0
        changed = True
    if changed:
        mark_physical_state_dirty(solution)
    return solution
