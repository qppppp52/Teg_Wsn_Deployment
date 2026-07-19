"""Sensor-AP link constraints and deterministic normalization."""
from __future__ import annotations

import numpy as np

from src.physics.numerical_tolerances import POWER_ABS_TOL


def single_valid_connected_ap(solution, sensor_id, ctx):
    """Return the sole fully valid AP global ID, otherwise return None."""
    sensor_id = int(sensor_id)
    if solution.x[sensor_id] != 1:
        return None
    connected = np.where(solution.c[sensor_id] == 1)[0]
    if len(connected) != 1:
        return None
    ap_id = int(connected[0])
    ptx_max = float(ctx.config.get("channel", {}).get("p_tx_max", 0.5))
    ptx_min = float(ctx.ptx_min_matrix[sensor_id, ap_id])
    if (
        solution.y[ap_id] != 1
        or ctx.link_feasible_matrix[sensor_id, ap_id] != 1
        or not np.isfinite(ptx_min)
        or ptx_min > ptx_max + POWER_ABS_TOL
    ):
        return None
    return ap_id


def check_link_constraints(solution, ctx):
    cv = 0.0
    K = ctx.num_candidates
    ptx_max = float(ctx.config.get("channel", {}).get("p_tx_max", 0.5))

    for sensor_id in range(K):
        for ap_id in range(K):
            if solution.c[sensor_id, ap_id] != 1:
                continue
            if solution.x[sensor_id] != 1:
                cv += 1.0
            if solution.y[ap_id] != 1:
                cv += 1.0
            if ctx.link_feasible_matrix[sensor_id, ap_id] == 0:
                cv += 1.0
            power = float(solution.p_tx[sensor_id, ap_id])
            if power > ptx_max + POWER_ABS_TOL:
                cv += 1.0
            ptx_min = float(ctx.ptx_min_matrix[sensor_id, ap_id])
            if power + POWER_ABS_TOL < ptx_min:
                cv += 1.0

    for sensor_id in np.where(solution.x == 1)[0]:
        connection_count = int(np.sum(solution.c[sensor_id]))
        if connection_count < 1:
            cv += 1.0
        elif connection_count > 1:
            cv += connection_count - 1
    return cv


def repair_link_constraints(solution, ctx):
    """Normalize connections and clear all stale power entries."""
    K = ctx.num_candidates
    ap_ids = np.where(solution.y == 1)[0]
    capacity = int(ctx.config["ap"]["C_max"])
    ptx_max = float(ctx.config.get("channel", {}).get("p_tx_max", 0.5))
    previous_power = solution.p_tx.copy()

    for sensor_id in range(K):
        for ap_id in range(K):
            if solution.c[sensor_id, ap_id] != 1:
                continue
            if (
                solution.x[sensor_id] != 1
                or solution.y[ap_id] != 1
                or ctx.link_feasible_matrix[sensor_id, ap_id] != 1
                or ctx.ptx_min_matrix[sensor_id, ap_id] > ptx_max + POWER_ABS_TOL
            ):
                solution.c[sensor_id, ap_id] = 0

    for sensor_id in np.where(solution.x == 1)[0]:
        connected = np.where(solution.c[sensor_id] == 1)[0]
        if len(connected) > 1:
            best_ap = min(
                (int(ap_id) for ap_id in connected),
                key=lambda ap_id: (
                    float(ctx.ptx_min_matrix[sensor_id, ap_id]),
                    ap_id,
                ),
            )
            solution.c[sensor_id] = 0
            solution.c[sensor_id, best_ap] = 1

        if not np.any(solution.c[sensor_id]):
            loads = np.sum(solution.c, axis=0)
            feasible = [
                int(ap_id)
                for ap_id in ap_ids
                if ctx.link_feasible_matrix[sensor_id, ap_id] == 1
                and ctx.ptx_min_matrix[sensor_id, ap_id] <= ptx_max + POWER_ABS_TOL
                and loads[ap_id] < capacity
            ]
            if feasible:
                best_ap = min(
                    feasible,
                    key=lambda ap_id: (
                        float(ctx.ptx_min_matrix[sensor_id, ap_id]),
                        ap_id,
                    ),
                )
                solution.c[sensor_id, best_ap] = 1

    solution.p_tx.fill(0.0)
    for sensor_id in np.where(solution.x == 1)[0]:
        ap_id = single_valid_connected_ap(solution, sensor_id, ctx)
        if ap_id is None:
            continue
        ptx_min = float(ctx.ptx_min_matrix[sensor_id, ap_id])
        previous = float(previous_power[sensor_id, ap_id])
        solution.p_tx[sensor_id, ap_id] = min(
            ptx_max, max(ptx_min, previous if previous > 0.0 else ptx_min)
        )
    return solution
