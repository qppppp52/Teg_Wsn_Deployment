"""Fixed reward-v3 for DQN-CR-MODE."""
from __future__ import annotations

import numpy as np

from src.constraints.cv_schema import CV_COMPONENT_KEYS, REWARD_SCHEMA_VERSION


REWARD_VERSION = 3
REWARD_WEIGHTS = {
    "cv": 0.20,
    "pressure_mean": 0.10,
    "pressure_max": 0.05,
    "pressure_regression": 0.05,
    "fr": 0.10,
    "hv": 0.25,
    "obj": 0.25,
    "div": 0.05,
    "cost": 0.03,
}
PRESSURE_KEYS = CV_COMPONENT_KEYS


def compute_reward(
    metrics_t,
    metrics_t1,
    max_repair_iter=5,
    reward_clip=(-1.0, 1.0),
    pressure_regression_tolerance=0.02,
):
    """Return a clipped incremental reward with shared pressure semantics."""
    if pressure_regression_tolerance < 0.0:
        raise ValueError("pressure_regression_tolerance must be nonnegative")
    cv_t = float(metrics_t.get("CV_mean", 0.0))
    cv_t1 = float(metrics_t1.get("CV_mean", cv_t))
    r_cv = _clip((cv_t - cv_t1) / (1.0 + abs(cv_t)))
    pressure_t = _pressure_vector(metrics_t)
    pressure_t1 = _pressure_vector(metrics_t1)
    pressure_delta = {
        key: _clip(pressure_t[key] - pressure_t1[key])
        for key in PRESSURE_KEYS
    }
    r_pressure_mean = _clip(float(np.mean(list(pressure_delta.values()))))
    r_pressure_max = _clip(max(pressure_t.values()) - max(pressure_t1.values()))
    pressure_regression = float(np.mean([
        max(0.0, pressure_t1[key] - pressure_t[key] - pressure_regression_tolerance)
        for key in PRESSURE_KEYS
    ]))
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
        + REWARD_WEIGHTS["pressure_mean"] * r_pressure_mean
        + REWARD_WEIGHTS["pressure_max"] * r_pressure_max
        - REWARD_WEIGHTS["pressure_regression"] * pressure_regression
        + REWARD_WEIGHTS["fr"] * r_fr
        + REWARD_WEIGHTS["hv"] * r_hv
        + REWARD_WEIGHTS["obj"] * r_obj
        + REWARD_WEIGHTS["div"] * r_div
        - REWARD_WEIGHTS["cost"] * r_cost
    )
    reward = float(np.clip(reward, *reward_clip))
    parts = {
        "R_CV": r_cv,
        "R_pressure_mean": r_pressure_mean,
        "R_pressure_max": r_pressure_max,
        "penalty_pressure_regression": pressure_regression,
        "R_FR": r_fr,
        "R_HV": r_hv,
        "R_obj": r_obj,
        "R_div": r_div,
        "R_cost": r_cost,
        "reward_version": REWARD_VERSION,
        "reward_schema_version": REWARD_SCHEMA_VERSION,
    }
    for key in PRESSURE_KEYS:
        parts[f"delta_p_{key}"] = pressure_delta[key]
    return reward, parts


def _pressure_vector(metrics: dict) -> dict[str, float]:
    raw = metrics.get("pressure", {}) or {}
    values = {}
    for key in PRESSURE_KEYS:
        try:
            value = float(raw.get(key, 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Pressure {key!r} must be numeric") from exc
        if not np.isfinite(value):
            raise ValueError(f"Pressure {key!r} must be finite")
        values[key] = float(np.clip(value, 0.0, 1.0))
    return values


def _clip(value):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("reward components must be finite")
    return float(np.clip(value, -1.0, 1.0))
