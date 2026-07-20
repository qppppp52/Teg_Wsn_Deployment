"""Deployment-count and Sensor/AP role-exclusion constraints."""
from __future__ import annotations

import numpy as np

from src.constraints.constraint_report import mark_physical_state_dirty


def check_deployment_constraints(solution, ctx):
    """Return the normalized deployment CV from the shared contract."""
    scfg = ctx.config.get("deployment", {})
    max_sensors = int(scfg.get("max_sensors", 20))
    max_aps = int(scfg.get("max_aps", 4))
    active_sensors = int(np.sum(solution.x == 1))
    active_aps = int(np.sum(solution.y == 1))
    overlap = int(np.sum((solution.x == 1) & (solution.y == 1)))
    sensor_excess = max(0, active_sensors - max_sensors) / max(1, max_sensors)
    ap_excess = max(0, active_aps - max_aps) / max(1, max_aps)
    role_overlap = overlap / max(1, int(ctx.num_candidates))
    return float((sensor_excess + ap_excess + role_overlap) / 3.0)


def _sensor_coverage_contribution(gid, ctx):
    return int(np.sum(ctx.coverage_matrix[gid]))


def _ap_communication_contribution(gid, ctx):
    reachable = ctx.link_feasible_matrix[:, gid] == 1
    count = int(np.sum(reachable))
    if count == 0:
        return 0.0
    return float(count * np.mean(ctx.potential_rate_matrix[:, gid][reachable]))


def repair_deployment(solution, ctx):
    """Repair deployment maxima and role overlap without AP service-count semantics."""
    scfg = ctx.config.get("deployment", {})
    max_sensors = int(scfg.get("max_sensors", 20))
    max_aps = int(scfg.get("max_aps", 4))
    changed = False

    overlap = np.where((solution.x == 1) & (solution.y == 1))[0]
    for gid in overlap:
        if _sensor_coverage_contribution(int(gid), ctx) >= _ap_communication_contribution(int(gid), ctx):
            _deactivate_ap(solution, int(gid))
        else:
            _deactivate_sensor(solution, int(gid))
        changed = True

    sensor_ids = np.where(solution.x == 1)[0]
    if len(sensor_ids) > max_sensors:
        scores = np.asarray([_sensor_coverage_contribution(int(gid), ctx) for gid in sensor_ids])
        for gid in sensor_ids[np.argsort(scores)[: len(sensor_ids) - max_sensors]]:
            _deactivate_sensor(solution, int(gid))
            changed = True

    ap_ids = np.where(solution.y == 1)[0]
    if len(ap_ids) > max_aps:
        scores = np.asarray([_ap_communication_contribution(int(gid), ctx) for gid in ap_ids])
        for gid in ap_ids[np.argsort(scores)[: len(ap_ids) - max_aps]]:
            _deactivate_ap(solution, int(gid))
            changed = True

    if changed:
        mark_physical_state_dirty(solution)
    return solution

def _deactivate_sensor(solution, sensor_id):
    solution.x[sensor_id] = 0
    solution.c[sensor_id, :] = 0
    solution.p_tx[sensor_id, :] = 0.0
    solution.z_sink_sensor[sensor_id] = []

def _deactivate_ap(solution, ap_id):
    solution.y[ap_id] = 0
    solution.c[:, ap_id] = 0
    solution.p_tx[:, ap_id] = 0.0
    solution.z_sink_ap[ap_id] = []
