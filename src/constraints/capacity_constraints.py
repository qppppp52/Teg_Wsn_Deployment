import numpy as np

def check_capacity_constraints(solution, ctx):
    cv = 0.0
    Cmax = ctx.config["ap"]["C_max"]
    loads = np.sum(solution.c, axis=0)
    for ai in np.where(solution.y == 1)[0]:
        if loads[ai] > Cmax:
            cv += loads[ai] - Cmax
    return cv

def repair_capacity(solution, ctx):
    Cmax = ctx.config["ap"]["C_max"]
    for ai in np.where(solution.y == 1)[0]:
        sensors = np.where(solution.c[:, ai] == 1)[0]
        if len(sensors) <= Cmax:
            continue
        rates = [ctx.potential_rate_matrix[si, ai] for si in sensors]
        order = np.argsort(rates)[::-1]
        for si in sensors[order[Cmax:]]:
            solution.c[si, ai] = 0
