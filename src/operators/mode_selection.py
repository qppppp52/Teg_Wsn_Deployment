def select_better(trial_ind, trial_sol, target_ind, target_sol):
    if trial_sol.feasible and not target_sol.feasible:
        return trial_ind, trial_sol, True
    if not trial_sol.feasible and target_sol.feasible:
        return target_ind, target_sol, False
    if not trial_sol.feasible and not target_sol.feasible:
        if trial_sol.cv < target_sol.cv:
            return trial_ind, trial_sol, True
        return target_ind, target_sol, False
    if (trial_sol.coverage >= target_sol.coverage and
        trial_sol.throughput >= target_sol.throughput and
        (trial_sol.coverage > target_sol.coverage or
         trial_sol.throughput > target_sol.throughput)):
        return trial_ind, trial_sol, True
    return target_ind, target_sol, False
