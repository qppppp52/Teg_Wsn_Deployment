"""AP service constraints without an AP sensor-count upper bound."""
from __future__ import annotations

import numpy as np

from src.constraints.constraint_report import mark_physical_state_dirty


def check_ap_service(solution, ctx):
    """Measure the optional non-empty-service rule for deployed APs."""
    if not bool(ctx.config.get("constraints", {}).get("ap_service_enabled", True)):
        return 0.0
    active_aps = max(1, int(np.sum(solution.y == 1)))
    empty = sum(
        solution.y[ap_id] == 1 and not np.any((solution.c[:, ap_id] == 1) & (solution.x == 1))
        for ap_id in range(ctx.num_candidates)
    )
    return float(empty / active_aps)


def repair_empty_aps(solution, ctx):
    """Remove deployed APs with no real sensor connection when enabled."""
    if not bool(ctx.config.get("constraints", {}).get("ap_service_enabled", True)):
        return solution
    changed = False
    for ap_id in np.where(solution.y == 1)[0]:
        if np.any((solution.c[:, ap_id] == 1) & (solution.x == 1)):
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
