"""Minimum theoretical heatsink demand from authoritative node consumption."""
from __future__ import annotations

import math

from src.constraints.constraint_report import (
    derive_consumption_snapshot,
    derive_sink_requirement_snapshot,
    ensure_constraint_spec,
)
from src.physics.numerical_tolerances import ENERGY_ABS_TOL


class InvalidHarvestPower(ValueError):
    """Compatibility exception for callers that still validate P_grid explicitly."""


def required_sink_count(consumption: float, p_grid: float) -> int:
    """Return a non-negative compatibility demand for one positive P_grid."""
    p_grid = float(p_grid)
    if p_grid <= 0.0:
        return 0
    adjusted = max(0.0, float(consumption) - ENERGY_ABS_TOL)
    return int(math.ceil(adjusted / p_grid))


def compute_sink_requirements(solution, ctx):
    """Refresh derived demand caches from raw x/y/c/p_tx only."""
    spec = ensure_constraint_spec(ctx)
    consumption = derive_consumption_snapshot(solution, ctx)
    requirements = derive_sink_requirement_snapshot(consumption, solution, ctx, spec)
    solution.sensor_power_consumption[:] = consumption.sensor
    solution.ap_power_consumption[:] = consumption.ap
    solution.n_sink_sensor[:] = requirements.required_sensor
    solution.n_sink_ap[:] = requirements.required_ap
    solution.metadata["unachievable_sensor"] = list(requirements.unachievable_sensor)
    solution.metadata["unachievable_ap"] = list(requirements.unachievable_ap)
    return solution