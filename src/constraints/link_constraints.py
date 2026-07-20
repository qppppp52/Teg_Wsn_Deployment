"""Sensor-AP link predicates and capacity-free deterministic repair."""
from __future__ import annotations

import numpy as np

from src.constraints.constraint_report import edge_ptx_max, evaluate_constraints, mark_physical_state_dirty
from src.physics.numerical_tolerances import POWER_ABS_TOL


def single_valid_connected_ap(solution, sensor_id, ctx):
    """Return the sole endpoint/link-valid AP global ID, otherwise None."""
    sensor_id = int(sensor_id)
    if solution.x[sensor_id] != 1:
        return None
    connected = np.where(solution.c[sensor_id] == 1)[0]
    if len(connected) != 1:
        return None
    ap_id = int(connected[0])
    ptx_min = float(ctx.ptx_min_matrix[sensor_id, ap_id])
    if (
        solution.y[ap_id] != 1
        or ctx.link_feasible_matrix[sensor_id, ap_id] != 1
        or not np.isfinite(ptx_min)
        or ptx_min > edge_ptx_max(ctx, sensor_id, ap_id) + POWER_ABS_TOL
    ):
        return None
    return ap_id


def check_link_constraints(solution, ctx):
    """Return the normalized link CV from the unified evaluator."""
    return float(evaluate_constraints(solution, ctx).components.link)


def repair_link_constraints(solution, ctx):
    """Normalize connections without imposing an AP Sensor-count limit."""
    K = int(ctx.num_candidates)
    ap_ids = [int(ap_id) for ap_id in np.where(solution.y == 1)[0]]
    previous_connections = solution.c.copy()
    previous_power = solution.p_tx.copy()

    for sensor_id in range(K):
        for ap_id in range(K):
            if solution.c[sensor_id, ap_id] != 1:
                continue
            ptx_min = float(ctx.ptx_min_matrix[sensor_id, ap_id])
            if (
                solution.x[sensor_id] != 1
                or solution.y[ap_id] != 1
                or ctx.link_feasible_matrix[sensor_id, ap_id] != 1
                or not np.isfinite(ptx_min)
                or ptx_min > edge_ptx_max(ctx, sensor_id, ap_id) + POWER_ABS_TOL
            ):
                solution.c[sensor_id, ap_id] = 0

    for sensor_id in np.where(solution.x == 1)[0]:
        connected = np.where(solution.c[sensor_id] == 1)[0]
        if len(connected) > 1:
            best_ap = min(
                (int(ap_id) for ap_id in connected),
                key=lambda ap_id: (float(ctx.ptx_min_matrix[sensor_id, ap_id]), ap_id),
            )
            solution.c[sensor_id] = 0
            solution.c[sensor_id, best_ap] = 1
        if not np.any(solution.c[sensor_id]):
            feasible = [
                ap_id
                for ap_id in ap_ids
                if ctx.link_feasible_matrix[sensor_id, ap_id] == 1
                and np.isfinite(ctx.ptx_min_matrix[sensor_id, ap_id])
                and ctx.ptx_min_matrix[sensor_id, ap_id] <= edge_ptx_max(ctx, sensor_id, ap_id) + POWER_ABS_TOL
            ]
            if feasible:
                best_ap = min(
                    feasible,
                    key=lambda ap_id: (float(ctx.ptx_min_matrix[sensor_id, ap_id]), ap_id),
                )
                solution.c[sensor_id, best_ap] = 1

    solution.p_tx.fill(0.0)
    for sensor_id in np.where(solution.x == 1)[0]:
        ap_id = single_valid_connected_ap(solution, sensor_id, ctx)
        if ap_id is None:
            continue
        ptx_min = float(ctx.ptx_min_matrix[sensor_id, ap_id])
        previous = float(previous_power[sensor_id, ap_id])
        if not np.isfinite(previous) or previous <= 0.0:
            previous = ptx_min
        solution.p_tx[sensor_id, ap_id] = min(
            edge_ptx_max(ctx, sensor_id, ap_id), max(ptx_min, previous)
        )
    if (
        not np.array_equal(previous_connections, solution.c)
        or not np.array_equal(previous_power, solution.p_tx)
    ):
        mark_physical_state_dirty(solution)
    return solution