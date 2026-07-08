import numpy as np

def check_ap_service(solution, ctx):
    cv = 0.0
    for ai in np.where(solution.y == 1)[0]:
        if np.sum(solution.c[:, ai]) == 0:
            cv += 1.0
    return cv

def repair_empty_aps(solution, ctx):
    for ai in np.where(solution.y == 1)[0]:
        if np.sum(solution.c[:, ai]) == 0:
            solution.y[ai] = 0
