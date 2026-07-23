"""Decode, repair, evaluate and finalize one optimization individual."""
from __future__ import annotations

import numpy as np

from src.constraints.constraint_eval import evaluate_all_constraints
from src.constraints.constraint_comparator import is_repair_candidate_better
from src.constraints.repair_config import resolve_repair_config
from src.constraints.repair_operator import repair_solution, repair_solution_with_strategy
from src.decoder.solution_decoder import decode_solution
from src.objectives.objective_eval import evaluate_objectives
from src.power.throughput_enhancer import enhance_rsum_capacity_greedily


def repair_loop(individual, ctx, max_iter=None, repair_strategy=None):
    """Keep the best constraint report observed across bounded repair rounds."""
    repair_config = resolve_repair_config(ctx.config)
    if max_iter is None:
        max_iter = repair_config.max_outer_repair_rounds
    sol = decode_solution(individual, ctx)
    evaluate_all_constraints(sol, ctx)
    sol.cv_before_repair = sol.cv
    sol.feasible_before_repair = bool(sol.feasible)
    if sol.feasible:
        sol = _finalize_feasible_solution(sol, ctx)
        sol.cv_after_repair = sol.cv
        sol.feasible_after_repair = True
        sol.repair_success = True
        return sol, None

    best_sol = sol.copy()
    best_report = sol.constraint_report
    patience = repair_config.repair_patience
    no_improvement = 0
    seen = {_solution_signature(sol)}
    strategy_name = _strategy_get(repair_strategy, "name", None)
    repair_order = _strategy_get(repair_strategy, "repair_order", None)
    tolerance = float(ctx.config.get("constraints", {}).get("cv_improvement_tol", 0.0))

    for iter_idx in range(max_iter):
        if repair_strategy is None:
            repair_solution(sol, ctx)
        else:
            repair_solution_with_strategy(
                sol, ctx, repair_order=repair_order, strategy_name=strategy_name
            )
        evaluate_all_constraints(sol, ctx)
        sol.repair_iter = iter_idx + 1
        sol.repair_success = bool(sol.feasible)
        sol.repair_strategy = strategy_name
        sol.repair_order = list(repair_order or [])
        if sol.feasible:
            sol = _finalize_feasible_solution(sol, ctx)

        if is_repair_candidate_better(sol.constraint_report, best_report, tolerance):
            best_sol = sol.copy()
            best_report = sol.constraint_report
            no_improvement = 0
        else:
            no_improvement += 1
        if sol.feasible:
            break
        signature = _solution_signature(sol)
        if (
            (repair_config.enable_repair_cycle_detection and signature in seen)
            or no_improvement >= patience
        ):
            break
        seen.add(signature)

    if not best_sol.feasible:
        evaluate_objectives(best_sol, ctx)
    best_sol.repair_success = bool(best_sol.feasible)
    best_sol.repair_strategy = strategy_name
    best_sol.repair_order = list(repair_order or [])
    best_sol.cv_after_repair = best_sol.cv
    best_sol.feasible_after_repair = bool(best_sol.feasible)
    repaired_individual = (
        _build_repaired_individual(individual, best_sol, ctx)
        if _deployment_changed(individual, best_sol, ctx)
        else None
    )
    return best_sol, repaired_individual


def evaluate_decoded_solution_without_repair(individual, ctx):
    """Evaluate a decoded MODE individual without entering the repair loop.

    MODE still uses the common deterministic decoder, constraint evaluator, and
    objective definition. It deliberately skips ``repair_solution`` so that
    comparisons against CR-MODE isolate the effect of constraint repair.
    """
    solution = decode_solution(individual, ctx)
    evaluate_all_constraints(solution, ctx)
    solution.cv_before_repair = solution.cv
    solution.feasible_before_repair = bool(solution.feasible)
    solution.repair_iter = 0
    solution.repair_success = bool(solution.feasible)
    solution.repair_strategy = "disabled"
    solution.repair_order = []

    if solution.feasible:
        solution = _finalize_feasible_solution(solution, ctx)
    else:
        evaluate_objectives(solution, ctx)

    solution.cv_after_repair = solution.cv
    solution.feasible_after_repair = bool(solution.feasible)
    solution.metadata["repair_enabled"] = False
    return solution, None


def _finalize_feasible_solution(solution, ctx):
    """Submit only a feasible, deployment-preserving strict Rsum improvement."""
    evaluate_all_constraints(solution, ctx)
    if not solution.feasible:
        evaluate_objectives(solution, ctx)
        return solution
    baseline = solution.copy()
    evaluate_objectives(baseline, ctx)
    finalized = enhance_rsum_capacity_greedily(baseline, ctx)
    evaluate_all_constraints(finalized, ctx)
    evaluate_objectives(finalized, ctx)
    gain_tol = float(ctx.config.get("objectives", {}).get("rsum_gain_tolerance", 1.0e-12))
    coverage_tol = float(
        ctx.config.get("objectives", {}).get("coverage_compare_tolerance", 1.0e-12)
    )
    valid = (
        finalized.feasible
        and np.array_equal(finalized.x, baseline.x)
        and np.array_equal(finalized.y, baseline.y)
        and np.array_equal(finalized.c, baseline.c)
        and abs(finalized.coverage - baseline.coverage) <= coverage_tol
        and finalized.rsum_capacity > baseline.rsum_capacity + gain_tol
    )
    if not valid:
        return baseline
    finalized.metadata["boost_invocations"] = int(
        finalized.metadata.get("boost_invocations", 0)
    ) + 1
    return finalized


def _strategy_get(strategy, key, default=None):
    if strategy is None:
        return default
    if isinstance(strategy, dict):
        return strategy.get(key, default)
    return getattr(strategy, key, default)


def _deployment_changed(individual, solution, ctx):
    from src.decoder.deployment_decoder import decode_deployment

    initial_x, initial_y = decode_deployment(individual, ctx)
    return not (
        np.array_equal(initial_x, solution.x)
        and np.array_equal(initial_y, solution.y)
    )


def _build_repaired_individual(individual, solution, ctx):
    mapping = ctx.index_mapping
    new_individual = individual.copy()
    for local_id in range(len(new_individual.rho_s)):
        global_id = mapping.Ls_local_to_global[local_id]
        if solution.x[global_id] == 1:
            new_individual.rho_s[local_id] = min(1.0, new_individual.rho_s[local_id] * 1.2)
        else:
            new_individual.rho_s[local_id] = max(0.0, new_individual.rho_s[local_id] * 0.5)
    for local_id in range(len(new_individual.rho_a)):
        global_id = mapping.La_local_to_global[local_id]
        if solution.y[global_id] == 1:
            new_individual.rho_a[local_id] = min(1.0, new_individual.rho_a[local_id] * 1.2)
        else:
            new_individual.rho_a[local_id] = max(0.0, new_individual.rho_a[local_id] * 0.5)
    new_individual.clip()
    return new_individual


def _solution_signature(solution):
    return (
        solution.x.tobytes(),
        solution.y.tobytes(),
        solution.c.tobytes(),
        solution.p_tx.tobytes(),
        tuple(tuple(sorted(int(grid) for grid in positions)) for positions in solution.z_sink_sensor),
        tuple(tuple(sorted(int(grid) for grid in positions)) for positions in solution.z_sink_ap),
    )
