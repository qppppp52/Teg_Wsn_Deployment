import numpy as np
from src.model.individual import Individual
from src.constraints.constraint_eval import evaluate_all_constraints
from src.constraints.repair_operator import repair_solution
from src.decoder.solution_decoder import decode_solution
from src.objectives.objective_eval import evaluate_objectives

def repair_loop(individual, ctx, max_iter=5):
    sol = decode_solution(individual, ctx)
    evaluate_all_constraints(sol, ctx)
    evaluate_objectives(sol, ctx)
    if sol.feasible:
        return sol, None
    rep_ind = None
    prev_cv = sol.cv
    for _ in range(max_iter):
        repair_solution(sol, ctx)
        evaluate_all_constraints(sol, ctx)
        evaluate_objectives(sol, ctx)
        if sol.feasible:
            if rep_ind is None and _deployment_changed(individual, sol, ctx):
                rep_ind = _build_repaired_individual(individual, sol, ctx)
            break
        if sol.cv >= prev_cv:
            break
        prev_cv = sol.cv
    return sol, rep_ind

def _deployment_changed(ind, sol, ctx):
    from src.decoder.deployment_decoder import decode_deployment
    x0, y0 = decode_deployment(ind, ctx)
    return not (np.array_equal(x0, sol.x) and np.array_equal(y0, sol.y))

def _build_repaired_individual(ind, sol, ctx):
    im = ctx.index_mapping
    new_ind = ind.copy()
    for ls in range(len(new_ind.rho_s)):
        gid = im.Ls_local_to_global[ls]
        new_ind.rho_s[ls] = min(1.0, new_ind.rho_s[ls]*1.2) if sol.x[gid]==1 else max(0.0, new_ind.rho_s[ls]*0.5)
    for la in range(len(new_ind.rho_a)):
        gid = im.La_local_to_global[la]
        new_ind.rho_a[la] = min(1.0, new_ind.rho_a[la]*1.2) if sol.y[gid]==1 else max(0.0, new_ind.rho_a[la]*0.5)
    new_ind.clip()
    return new_ind
