import numpy as np

def iterative_filter(sensor_mask, ap_mask, link_feasible, max_rounds=10):
    sm, am = sensor_mask.copy(), ap_mask.copy()
    for _ in range(max_rounds):
        prev_sm, prev_am = sm.copy(), am.copy()
        for r in range(len(sm)):
            if sm[r] and not np.any(link_feasible[r] & am):
                sm[r] = False
        for r in range(len(am)):
            if am[r] and not np.any(link_feasible[:, r] & sm):
                am[r] = False
        if np.array_equal(sm, prev_sm) and np.array_equal(am, prev_am):
            break
    return sm, am
