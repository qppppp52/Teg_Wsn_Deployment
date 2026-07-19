"""AP service constraints."""
import numpy as np


def check_ap_service(solution, ctx):
    cv = 0.0
    for ap_id in np.where(solution.y == 1)[0]:
        if np.sum(solution.c[:, ap_id]) == 0:
            cv += 1.0
    return cv


def repair_empty_aps(solution, ctx):
    for ap_id in np.where(solution.y == 1)[0]:
        if np.sum(solution.c[:, ap_id]) != 0:
            continue
        solution.y[ap_id] = 0
        solution.c[:, ap_id] = 0
        solution.p_tx[:, ap_id] = 0.0
        solution.n_sink_ap[ap_id] = 0
        solution.z_sink_ap[ap_id] = []
        solution.ap_power_consumption[ap_id] = 0.0
        solution.ap_harvest_power[ap_id] = 0.0
