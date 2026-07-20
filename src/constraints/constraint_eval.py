"""Unified entry points for the immutable constraint report."""
from __future__ import annotations

import numpy as np

from src.constraints.constraint_report import (
    evaluate_constraints,
    ensure_constraint_spec,
)


def apply_constraint_report(solution, report):
    """Refresh compatibility caches without making them authoritative."""
    components = report.components
    solution.constraint_report = report
    solution.cv = float(report.cv_total)
    solution.feasible = bool(report.feasible)
    solution.cv_deploy = float(components.deploy)
    solution.cv_link = float(components.link)
    solution.cv_power = float(components.power)
    solution.cv_service = float(components.service)
    solution.cv_sink = float(components.sink)
    solution.cv_energy = float(components.energy)
    solution.cv_energy_sensor = float(report.energy_cv.sensor)
    solution.cv_energy_ap = float(report.energy_cv.ap)
    solution.cv_sink_conflict = float(report.raw.sink_hard.cross_owner_conflict_excess)
    solution.cv_sink_shortage = float(
        report.raw.sink_diagnostics.shortage_sensor
        + report.raw.sink_diagnostics.shortage_ap
    )
    solution.cv_sink_invalid = float(
        report.raw.sink_hard.invalid_index_or_type_count
        + report.raw.sink_hard.undeployed_owner_count
        + report.raw.sink_hard.outside_allowed_neighborhood_count
        + report.raw.sink_hard.illegal_node_overlap_excess
        + report.raw.sink_hard.duplicate_excess
    )

    consumption = report.physics.consumption
    requirements = report.physics.sink_requirement
    energy = report.physics.energy
    solution.sensor_power_consumption[:] = np.asarray(consumption.sensor, dtype=float)
    solution.ap_power_consumption[:] = np.asarray(consumption.ap, dtype=float)
    solution.n_sink_sensor[:] = np.asarray(requirements.required_sensor, dtype=np.int32)
    solution.n_sink_ap[:] = np.asarray(requirements.required_ap, dtype=np.int32)
    solution.sensor_harvest_power[:] = np.asarray(energy.sensor_harvest, dtype=float)
    solution.ap_harvest_power[:] = np.asarray(energy.ap_harvest, dtype=float)
    return solution


def evaluate_report(solution, ctx):
    """Pure evaluator used by new code."""
    return evaluate_constraints(solution, ctx)


def evaluate_all_constraints(solution, ctx):
    """Evaluate once, refresh legacy scalar caches, and return total CV."""
    report = evaluate_report(solution, ctx)
    apply_constraint_report(solution, report)
    return float(report.cv_total)
