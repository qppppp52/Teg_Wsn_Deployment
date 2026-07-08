import numpy as np

def build_candidate_sets(sensor_mask, ap_mask, link_feasible):
    sm = sensor_mask.copy()
    am = ap_mask.copy()
    for r in range(len(sm)):
        if sm[r] and not np.any(link_feasible[r] & am):
            sm[r] = False
    for r in range(len(am)):
        if am[r] and not np.any(link_feasible[:, r] & sm):
            am[r] = False
    return sm, am
