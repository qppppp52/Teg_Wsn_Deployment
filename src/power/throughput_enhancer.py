"""Deterministic post-feasibility enhancement of theoretical capacity."""
from __future__ import annotations

import hashlib
import time

import numpy as np

from src.constraints.constraint_eval import evaluate_all_constraints
from src.constraints.link_constraints import single_valid_connected_ap
from src.heatsink.harvest_power import refresh_harvest_diagnostics
from src.heatsink.sink_overlap_rules import SinkOverlapRules
from src.heatsink.sink_ownership import build_sink_ownership
from src.heatsink.sink_requirement import compute_sink_requirements, required_sink_count
from src.objectives.objective_eval import evaluate_objectives
from src.objectives.rsum_capacity_objective import compute_link_capacity
from src.physics.node_power_model import compute_sensor_consumption
from src.physics.numerical_tolerances import (
    CAPACITY_ABS_GAIN_BPS,
    CAPACITY_REL_TOL,
    POWER_ABS_TOL,
)
from src.power.power_bounds import max_energy_feasible_ptx


THROUGHPUT_ENHANCER_VERSION = 2


class UnsupportedEnhancementConfiguration(ValueError):
    pass


def enhance_rsum_capacity_greedily(solution, ctx):
    """Add legal Sensor sinks with positive marginal Shannon-capacity gain."""
    config = ctx.config.get("throughput_enhancement", {})
    enabled = bool(config.get("enabled", False))
    before = solution.copy()
    if not enabled:
        return _record_no_boost(before, "disabled")
    if not bool(solution.feasible):
        return _record_no_boost(before, "infeasible")

    physical_signature = _physical_signature(solution)
    enhancement_contract = _enhancement_contract(config)
    if (
        solution.metadata.get("boost_finalized_signature") == physical_signature
        and solution.metadata.get("boost_finalized_contract")
        == enhancement_contract
    ):
        return before

    rules = SinkOverlapRules(ctx.config)
    if rules.allow_sink_sink_overlap:
        raise UnsupportedEnhancementConfiguration(
            "Capacity enhancement requires exclusive heatsink ownership"
        )

    started = time.perf_counter()
    candidate = solution.copy()
    evaluate_all_constraints(candidate, ctx)
    evaluate_objectives(candidate, ctx)
    if not candidate.feasible:
        return _record_no_boost(before, "infeasible_after_recheck")

    initial_coverage = float(candidate.coverage)
    initial_rsum = float(candidate.rsum_capacity)
    initial_x = candidate.x.copy()
    initial_y = candidate.y.copy()
    initial_c = candidate.c.copy()
    initial_ownership = build_sink_ownership(candidate, ctx)
    initial_ap_counts = np.asarray(initial_ownership.effective_ap_count, dtype=int)
    natural_step_cap = _current_global_legal_free_grid_count(
        candidate, ctx, rules
    )
    configured_cap = int(config.get("max_boost_steps", ctx.num_candidates))
    max_steps = max(0, min(configured_cap, natural_step_cap))

    accepted = 0
    candidate_evaluations = 0
    stop_reason = "no_positive_gain"
    for _ in range(max_steps):
        ownership = build_sink_ownership(candidate, ctx)
        options = []
        threshold = max(
            CAPACITY_ABS_GAIN_BPS,
            CAPACITY_REL_TOL * max(1.0, abs(candidate.rsum_capacity)),
        )
        for sensor_id in range(ctx.num_candidates):
            ap_id = single_valid_connected_ap(candidate, sensor_id, ctx)
            if ap_id is None:
                continue
            current_power = float(candidate.p_tx[sensor_id, ap_id])
            ptx_max = float(ctx.config.get("channel", {}).get("p_tx_max", 0.5))
            if current_power >= ptx_max - POWER_ABS_TOL:
                continue
            owned = set(candidate.z_sink_sensor[sensor_id])
            for grid_id in sorted(set(int(g) for g in ctx.neighbor_sets[sensor_id])):
                if grid_id in owned:
                    continue
                if not rules.can_place_sink_on_grid(
                    grid_id,
                    sensor_id,
                    candidate,
                    ctx,
                    owner_kind="sensor",
                ):
                    continue
                upper = max_energy_feasible_ptx(
                    candidate,
                    sensor_id,
                    ap_id,
                    ctx,
                    extra_effective_sinks=1,
                    ownership=ownership,
                )
                candidate_evaluations += 1
                if upper <= current_power + POWER_ABS_TOL:
                    continue
                current_rate = compute_link_capacity(
                    sensor_id, ap_id, current_power, ctx
                )
                next_rate = compute_link_capacity(sensor_id, ap_id, upper, ctx)
                gain = next_rate - current_rate
                if gain < threshold:
                    continue
                options.append(
                    (
                        -gain,
                        _boost_conflict_degree(
                            grid_id, sensor_id, candidate, ctx, rules
                        ),
                        sensor_id,
                        grid_id,
                        ap_id,
                        upper,
                    )
                )
        if not options:
            break

        _, _, sensor_id, grid_id, ap_id, upper = min(options)
        old_power = float(candidate.p_tx[sensor_id, ap_id])
        candidate.z_sink_sensor[sensor_id].append(int(grid_id))
        candidate.p_tx[sensor_id, ap_id] = float(upper)
        consumption = compute_sensor_consumption(sensor_id, upper, ctx.config)
        candidate.sensor_power_consumption[sensor_id] = consumption
        candidate.n_sink_sensor[sensor_id] = required_sink_count(
            consumption, ctx.P_grid[sensor_id]
        )
        step_ownership = build_sink_ownership(candidate, ctx)
        effective_positions = step_ownership.effective_sensor_positions[sensor_id]
        if (
            grid_id not in effective_positions
            or len(effective_positions) < candidate.n_sink_sensor[sensor_id]
        ):
            candidate.z_sink_sensor[sensor_id].remove(int(grid_id))
            candidate.p_tx[sensor_id, ap_id] = old_power
            stop_reason = "step_validation_failed"
            break
        accepted += 1
    else:
        stop_reason = "step_cap"

    compute_sink_requirements(candidate, ctx)
    refresh_harvest_diagnostics(candidate, ctx)
    evaluate_all_constraints(candidate, ctx)
    evaluate_objectives(candidate, ctx)
    final_ownership = build_sink_ownership(candidate, ctx)
    final_ap_counts = np.asarray(final_ownership.effective_ap_count, dtype=int)
    valid = (
        candidate.feasible
        and np.array_equal(candidate.x, initial_x)
        and np.array_equal(candidate.y, initial_y)
        and np.array_equal(candidate.c, initial_c)
        and abs(candidate.coverage - initial_coverage) <= 1.0e-12
        and np.all(final_ap_counts >= initial_ap_counts)
        and candidate.rsum_capacity + CAPACITY_ABS_GAIN_BPS >= initial_rsum
        and sum(final_ownership.duplicate_sensor_count) == 0
        and sum(final_ownership.duplicate_ap_count) == 0
        and sum(final_ownership.invalid_sensor_count) == 0
        and sum(final_ownership.invalid_ap_count) == 0
        and final_ownership.forbidden_conflict_count == 0
    )
    runtime_ms = (time.perf_counter() - started) * 1000.0
    if not valid:
        rolled_back = before.copy()
        rolled_back.metadata.update(
            {
                "boost_applied": False,
                "boost_rollback": True,
                "boost_stop_reason": "final_validation_failed",
                "boost_runtime_ms": runtime_ms,
                "boost_candidate_evaluations": candidate_evaluations,
            }
        )
        return rolled_back

    candidate.metadata.update(
        {
            "boost_applied": bool(accepted),
            "boost_sinks_added": int(accepted),
            "boost_iterations": int(accepted),
            "boost_candidate_evaluations": int(candidate_evaluations),
            "boost_runtime_ms": float(runtime_ms),
            "rsum_capacity_before_boost": initial_rsum,
            "rsum_capacity_after_boost": float(candidate.rsum_capacity),
            "rsum_capacity_boost_gain": float(candidate.rsum_capacity - initial_rsum),
            "boost_stop_reason": stop_reason,
        }
    )
    candidate.metadata["boost_finalized_signature"] = _physical_signature(candidate)
    candidate.metadata["boost_finalized_contract"] = enhancement_contract
    return candidate


def _current_global_legal_free_grid_count(solution, ctx, rules):
    legal_grids = set()
    for sensor_id in range(ctx.num_candidates):
        if single_valid_connected_ap(solution, sensor_id, ctx) is None:
            continue
        owned = set(int(grid) for grid in solution.z_sink_sensor[sensor_id])
        for grid_id in set(int(grid) for grid in ctx.neighbor_sets[sensor_id]):
            if grid_id not in owned and rules.can_place_sink_on_grid(
                grid_id,
                sensor_id,
                solution,
                ctx,
                owner_kind="sensor",
            ):
                legal_grids.add(grid_id)
    return len(legal_grids)

def _boost_conflict_degree(grid_id, sensor_id, solution, ctx, rules):
    degree = 0
    for other_id in range(ctx.num_candidates):
        if other_id == sensor_id:
            continue
        if single_valid_connected_ap(solution, other_id, ctx) is None:
            continue
        if grid_id in set(int(g) for g in ctx.neighbor_sets[other_id]):
            if rules.can_place_sink_on_grid(
                grid_id,
                other_id,
                solution,
                ctx,
                owner_kind="sensor",
            ):
                degree += 1
    return degree


def _record_no_boost(solution, reason):
    solution.metadata.update(
        {
            "boost_applied": False,
            "boost_sinks_added": 0,
            "boost_iterations": 0,
            "boost_candidate_evaluations": 0,
            "boost_runtime_ms": 0.0,
            "rsum_capacity_before_boost": float(solution.rsum_capacity),
            "rsum_capacity_after_boost": float(solution.rsum_capacity),
            "rsum_capacity_boost_gain": 0.0,
            "boost_stop_reason": reason,
        }
    )
    return solution


def _physical_signature(solution):
    digest = hashlib.sha256()
    for array in (
        solution.x,
        solution.y,
        solution.c,
        solution.p_tx,
        solution.n_sink_sensor,
        solution.n_sink_ap,
    ):
        digest.update(np.ascontiguousarray(array).tobytes())
    for all_positions in (solution.z_sink_sensor, solution.z_sink_ap):
        digest.update(
            repr(tuple(tuple(sorted(int(g) for g in positions)) for positions in all_positions)).encode(
                "ascii"
            )
        )
    return digest.hexdigest()


def _enhancement_contract(config):
    payload = {
        "version": THROUGHPUT_ENHANCER_VERSION,
        "enabled": bool(config.get("enabled", False)),
        "max_boost_steps": int(config.get("max_boost_steps", 0)),
    }
    encoded = repr(sorted(payload.items())).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()