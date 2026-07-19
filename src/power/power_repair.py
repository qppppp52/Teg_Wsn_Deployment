"""Feasibility-only transmit-power capping and downshift."""
from __future__ import annotations

from src.constraints.link_constraints import single_valid_connected_ap
from src.heatsink.harvest_power import refresh_harvest_diagnostics
from src.physics.node_power_model import compute_ap_consumption, compute_sensor_consumption
from src.physics.numerical_tolerances import POWER_ABS_TOL
from src.power.power_bounds import max_energy_feasible_ptx


POWER_REPAIR_SEMANTICS_VERSION = 2
_VALID_POLICIES = {"retain_current", "conservative"}


def repair_power(solution, ctx):
    """Apply only feasibility semantics; never raise an established valid power."""
    policy = getattr(ctx, "current_power_policy", None)
    if policy is None:
        policy = solution.metadata.get("power_policy")
    if policy is None:
        policy = "retain_current"
    policy = _normalize_policy(policy)
    previous_power = solution.p_tx.copy()
    solution.p_tx.fill(0.0)
    ownership = refresh_harvest_diagnostics(solution, ctx)

    for sensor_id in range(ctx.num_candidates):
        ap_id = single_valid_connected_ap(solution, sensor_id, ctx)
        if ap_id is None:
            solution.sensor_power_consumption[sensor_id] = (
                compute_sensor_consumption(sensor_id, 0.0, ctx.config)
                if solution.x[sensor_id] == 1
                else 0.0
            )
            continue

        ptx_min = float(ctx.ptx_min_matrix[sensor_id, ap_id])
        current = float(previous_power[sensor_id, ap_id])
        if current <= POWER_ABS_TOL:
            current = ptx_min
        upper = max_energy_feasible_ptx(
            solution, sensor_id, ap_id, ctx, ownership=ownership
        )
        if upper + POWER_ABS_TOL < ptx_min:
            target = ptx_min
        elif policy == "conservative":
            target = ptx_min
        else:
            target = max(ptx_min, min(current, upper))
        solution.p_tx[sensor_id, ap_id] = target
        solution.sensor_power_consumption[sensor_id] = compute_sensor_consumption(
            sensor_id, target, ctx.config
        )

    for ap_id in range(ctx.num_candidates):
        if solution.y[ap_id] == 1:
            solution.ap_power_consumption[ap_id] = compute_ap_consumption(
                int(solution.c[:, ap_id].sum()), ctx.config
            )
        else:
            solution.ap_power_consumption[ap_id] = 0.0
    return solution


def _normalize_policy(policy):
    aliases = {
        "balanced": "retain_current",
        "energy_balanced": "retain_current",
        "rsum_capacity_priority": "retain_current",
        "sink_limited": "conservative",
    }
    normalized = aliases.get(str(policy), str(policy))
    if normalized not in _VALID_POLICIES:
        raise ValueError(f"Unknown power repair policy: {policy}")
    return normalized
