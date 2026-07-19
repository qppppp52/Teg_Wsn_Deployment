"""Reward calculation for DQN-CR-MODE."""
from __future__ import annotations

import numpy as np


def compute_reward(metrics_t, metrics_t1, max_repair_iter=5, reward_clip=(-1.0, 1.0)):
    """Compute clipped scalar reward and component dictionary."""
    eps = 1e-12
    cv0 = float(metrics_t.get("CV_mean", 0.0))
    cv1 = float(metrics_t1.get("CV_mean", cv0))
    r_cv = (cv0 - cv1) / (abs(cv0) + eps)

    pressure0 = metrics_t.get("pressure", {})
    pressure1 = metrics_t1.get("pressure", {})
    keys = ["deploy", "link", "capacity", "energy", "sink", "service"]
    r_pressure = sum(float(pressure0.get(k, 0.0)) * (float(pressure0.get(k, 0.0)) - float(pressure1.get(k, 0.0))) for k in keys)

    r_fr = float(metrics_t1.get("FR", 0.0)) - float(metrics_t.get("FR", 0.0))
    r_hv = float(metrics_t1.get("HV", 0.0)) - float(metrics_t.get("HV", 0.0))
    r_obj = 0.5 * (float(metrics_t1.get("best_coverage", 0.0)) - float(metrics_t.get("best_coverage", 0.0))) + 0.5 * (float(metrics_t1.get("best_rsum_norm", 0.0)) - float(metrics_t.get("best_rsum_norm", 0.0)))
    r_div = float(metrics_t1.get("diversity", 0.0)) - float(metrics_t.get("diversity", 0.0))
    r_cost = float(metrics_t1.get("mean_repair_iter", 0.0)) / max(float(max_repair_iter), 1.0)

    reward = 0.25*r_cv + 0.20*r_pressure + 0.20*r_fr + 0.25*r_hv + 0.05*r_obj + 0.05*r_div - 0.10*r_cost
    reward = float(np.clip(reward, *reward_clip))
    return reward, {
        "R_CV": r_cv, "R_pressure": r_pressure, "R_FR": r_fr,
        "R_HV": r_hv, "R_obj": r_obj, "R_div": r_div, "R_cost": r_cost,
    }
