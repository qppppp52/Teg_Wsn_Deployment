"""Deterministic initial Sensor-AP connection decoding."""
from __future__ import annotations

import numpy as np

from src.constraints.constraint_report import global_ptx_max
from src.physics.numerical_tolerances import POWER_ABS_TOL


def assign_connections(x, y, ctx):
    """Assign every feasible Sensor to its lowest-pmin active AP.

    AP load is retained as a physical power quantity, but it is not capped by
    an AP maximum-service hard constraint.
    """
    K = int(ctx.num_candidates)
    c = np.zeros((K, K), dtype=np.int8)
    sensor_ids = np.where(x == 1)[0]
    ap_ids = np.where(y == 1)[0]
    for sensor_id in sensor_ids:
        feasible_aps = [
            (int(ap_id), float(ctx.ptx_min_matrix[sensor_id, ap_id]))
            for ap_id in ap_ids
            if ctx.link_feasible_matrix[sensor_id, ap_id] == 1
            and np.isfinite(ctx.ptx_min_matrix[sensor_id, ap_id])
            and ctx.ptx_min_matrix[sensor_id, ap_id] <= global_ptx_max(ctx) + POWER_ABS_TOL
        ]
        if not feasible_aps:
            continue
        best_ap = min(feasible_aps, key=lambda item: (item[1], item[0]))[0]
        c[sensor_id, best_ap] = 1
    return c