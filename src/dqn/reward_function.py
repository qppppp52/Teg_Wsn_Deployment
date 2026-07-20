"""Fixed reward-v2 for DQN-CR-MODE."""
from __future__ import annotations

import numpy as np


REWARD_V2_VERSION = 2
REWARD_WEIGHTS = {
    "cv": 0.20,
    "pressure": 0.15,
    "fr": 0.10,
    "hv": 0.25,
    "obj": 0.25,
    "div": 0.05,
    "cost": 0.03,
}
PRESSURE_KEYS = ("deploy", "link", "power", "energy", "sink", "service")


def compute_reward(metrics_t, metrics_t1, max_repair_iter=5, reward_clip=(-1.0, 1.0)):
    """Return a clipped incremental reward with fixed experiment weights."""
    cv_t = float(metrics_t.get("CV_mean", 0.0))
    cv_t1 = float(metrics_t1.get("CV_mean", cv_t))
    r_cv = _clip((cv_t - cv_t1) / (1.0 + abs(cv_t)))
    pressure_t = metrics_t.get("pressure", {}) or {}
    pressure_t1 = metrics_t1.get("pressure", {}) or {}
    r_pressure = _clip(float(np.mean([
        float(pressure_t.get(key, 0.0)) - float(pressure_t1.get(key, 0.0))
        for key in PRESSURE_KEYS
    ])))
    r_fr = _clip(float(metrics_t1.get("FR", 0.0)) - float(metrics_t.get("FR", 0.0)))
    r_hv = _clip(float(metrics_t1.get("HV", 0.0)) - float(metrics_t.get("HV", 0.0)))
    r_obj = _clip(
        0.5 * (float(metrics_t1.get("best_coverage", 0.0)) - float(metrics_t.get("best_coverage", 0.0)))
        + 0.5 * (float(metrics_t1.get("best_rsum_norm", 0.0)) - float(metrics_t.get("best_rsum_norm", 0.0)))
    )
    r_div = _clip(float(metrics_t1.get("diversity", 0.0)) - float(metrics_t.get("diversity", 0.0)))
    r_cost = float(np.clip(
        float(metrics_t1.get("mean_repair_iter", 0.0)) / max(float(max_repair_iter), 1.0),
        0.0,
        1.0,
    ))
    reward = (
        REWARD_WEIGHTS["cv"] * r_cv
        + REWARD_WEIGHTS["pressure"] * r_pressure
        + REWARD_WEIGHTS["fr"] * r_fr
        + REWARD_WEIGHTS["hv"] * r_hv
        + REWARD_WEIGHTS["obj"] * r_obj
        + REWARD_WEIGHTS["div"] * r_div
        - REWARD_WEIGHTS["cost"] * r_cost
    )
    reward = float(np.clip(reward, *reward_clip))
    return reward, {
        "R_CV": r_cv,
        "R_pressure": r_pressure,
        "R_FR": r_fr,
        "R_HV": r_hv,
        "R_obj": r_obj,
        "R_div": r_div,
        "R_cost": r_cost,
        "reward_v2_version": REWARD_V2_VERSION,
    }


def _clip(value):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("reward components must be finite")
    return float(np.clip(value, -1.0, 1.0))