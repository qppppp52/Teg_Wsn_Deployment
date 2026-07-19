"""Deterministic initial Sensor-AP connection decoding."""
from __future__ import annotations

import numpy as np

from src.physics.numerical_tolerances import POWER_ABS_TOL


def assign_connections(x, y, ctx):
    K = ctx.num_candidates
    c = np.zeros((K, K), dtype=np.int8)
    ap_capacity = int(ctx.config["ap"]["C_max"])
    ptx_max = float(ctx.config.get("channel", {}).get("p_tx_max", 0.5))
    ap_loads = np.zeros(K, dtype=int)
    sensor_ids = np.where(x == 1)[0]
    ap_ids = np.where(y == 1)[0]
    for sensor_id in sensor_ids:
        feasible_aps = [
            (int(ap_id), float(ctx.ptx_min_matrix[sensor_id, ap_id]))
            for ap_id in ap_ids
            if ctx.link_feasible_matrix[sensor_id, ap_id] == 1
            and np.isfinite(ctx.ptx_min_matrix[sensor_id, ap_id])
            and ctx.ptx_min_matrix[sensor_id, ap_id]
            <= ptx_max + POWER_ABS_TOL
            and ap_loads[ap_id] < ap_capacity
        ]
        if not feasible_aps:
            continue
        feasible_aps.sort(key=lambda item: (item[1], item[0]))
        best_ap = feasible_aps[0][0]
        c[sensor_id, best_ap] = 1
        ap_loads[best_ap] += 1
    return c
