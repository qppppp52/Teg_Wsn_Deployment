import numpy as np
from src.constraints.constraint_eval import evaluate_all_constraints
from src.constraints.repair_operator import repair_solution, repair_solution_with_strategy
from src.decoder.solution_decoder import decode_solution
from src.objectives.objective_eval import evaluate_objectives


def repair_loop(individual, ctx, max_iter=5, repair_strategy=None):
    sol = decode_solution(individual, ctx)
    evaluate_all_constraints(sol, ctx)
    sol.cv_before_repair = sol.cv
    sol.feasible_before_repair = bool(sol.feasible)
    evaluate_objectives(sol, ctx)
    if sol.feasible:
        sol.cv_after_repair = sol.cv
        sol.feasible_after_repair = True
        sol.repair_success = True
        return sol, None

    best_sol = sol.copy()
    best_cv = float(sol.cv)
    patience = int(ctx.config.get("constraints", {}).get("repair_patience", 2))
    no_improvement = 0
    seen = {_solution_signature(sol)}
    strategy_name = _strategy_get(repair_strategy, "name", None)
    repair_order = _strategy_get(repair_strategy, "repair_order", None)
    power_policy = _strategy_get(repair_strategy, "power_policy", None)

    for iter_idx in range(max_iter):
        if repair_strategy is None:
            repair_solution(sol, ctx)
        else:
            repair_solution_with_strategy(
                sol, ctx,
                repair_order=repair_order,
                power_policy=power_policy,
                strategy_name=strategy_name,
            )
        evaluate_all_constraints(sol, ctx)
        evaluate_objectives(sol, ctx)
        sol.repair_iter = iter_idx + 1
        sol.repair_success = bool(sol.feasible)
        sol.repair_strategy = strategy_name
        sol.repair_order = list(repair_order or [])
        tolerance = float(ctx.config.get("constraints", {}).get("cv_improvement_tol", 0.0))
        if sol.feasible or sol.cv < best_cv - tolerance:
            best_sol = sol.copy()
            best_cv = float(sol.cv)
            no_improvement = 0
        else:
            no_improvement += 1
        if sol.feasible:
            break
        signature = _solution_signature(sol)
        if signature in seen or no_improvement >= patience:
            break
        seen.add(signature)
    best_sol.repair_iter = sol.repair_iter
    best_sol.repair_success = bool(best_sol.feasible)
    best_sol.repair_strategy = strategy_name
    best_sol.repair_order = list(repair_order or [])
    best_sol.cv_after_repair = best_sol.cv
    best_sol.feasible_after_repair = bool(best_sol.feasible)
    rep_ind = _build_repaired_individual(individual, best_sol, ctx) if _deployment_changed(individual, best_sol, ctx) else None
    return best_sol, rep_ind


def _strategy_get(strategy, key, default=None):
    if strategy is None:
        return default
    if isinstance(strategy, dict):
        return strategy.get(key, default)
    return getattr(strategy, key, default)


def _deployment_changed(ind, sol, ctx):
    from src.decoder.deployment_decoder import decode_deployment
    x0, y0 = decode_deployment(ind, ctx)
    return not (np.array_equal(x0, sol.x) and np.array_equal(y0, sol.y))


def _build_repaired_individual(ind, sol, ctx):
    im = ctx.index_mapping
    new_ind = ind.copy()
    for ls in range(len(new_ind.rho_s)):
        gid = im.Ls_local_to_global[ls]
        new_ind.rho_s[ls] = min(1.0, new_ind.rho_s[ls]*1.2) if sol.x[gid] == 1 else max(0.0, new_ind.rho_s[ls]*0.5)
    for la in range(len(new_ind.rho_a)):
        gid = im.La_local_to_global[la]
        new_ind.rho_a[la] = min(1.0, new_ind.rho_a[la]*1.2) if sol.y[gid] == 1 else max(0.0, new_ind.rho_a[la]*0.5)
    new_ind.clip()
    return new_ind


def _solution_signature(sol):
    return (
        sol.x.tobytes(),
        sol.y.tobytes(),
        sol.c.tobytes(),
        sol.p_tx.tobytes(),
        sol.n_sink_sensor.tobytes(),
        sol.n_sink_ap.tobytes(),
    )
