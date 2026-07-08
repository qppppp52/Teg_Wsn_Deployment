"""Action masking rules for DQN-CR-MODE."""
from __future__ import annotations

import numpy as np


def build_dqn_action_mask(state_metrics: dict, actions: list, config: dict) -> np.ndarray:
    """
    Build a boolean mask for the discrete DQN action space.

    True means the action may be selected. The rules are conservative: they
    suppress actions that are clearly mismatched with the current repair
    pressure and always leave at least one fallback action.
    """
    num_actions = len(actions)
    mask = np.ones(num_actions, dtype=bool)
    if num_actions == 0:
        return mask

    dcfg = config.get("dqn", {}) if isinstance(config, dict) else {}
    if not dcfg.get("action_mask_enabled", True):
        return mask

    fr = _metric(state_metrics, "FR", "FR_after_repair", "feasible_ratio", default=0.0)
    cv = _metric(state_metrics, "CV_mean", "CV_after_repair_mean", "cv", default=0.0)
    hv_stall = int(_metric(state_metrics, "hv_stall_generations", "HV_stall", default=0))
    pressure = state_metrics.get("pressure", {}) if isinstance(state_metrics, dict) else {}

    thresholds = dcfg.get("mask", {})
    low_fr = float(thresholds.get("low_fr", 0.2))
    feasible_fr = float(thresholds.get("feasible_fr", 0.8))
    low_cv = float(thresholds.get("low_cv", 1.0e-3))
    high_pressure = float(thresholds.get("high_pressure", 0.35))
    stall_generations = int(thresholds.get("hv_stall_generations", 5))

    name_to_idx = {str(action.get("name", "")): idx for idx, action in enumerate(actions)}

    if fr < low_fr:
        _disable(mask, name_to_idx, "throughput_priority")

    if _pressure(pressure, "energy", state_metrics) >= high_pressure:
        _disable(mask, name_to_idx, "throughput_priority")
        _enable(mask, name_to_idx, "energy_first", "conservative_repair")

    if _pressure(pressure, "link", state_metrics) >= high_pressure:
        _enable(mask, name_to_idx, "link_first")

    if _pressure(pressure, "capacity", state_metrics) >= high_pressure:
        _enable(mask, name_to_idx, "capacity_first")

    if _pressure(pressure, "sink", state_metrics) >= high_pressure:
        _enable(mask, name_to_idx, "sink_first")

    if hv_stall >= stall_generations:
        _enable(mask, name_to_idx, "diversity_boost", "exploration_high_F")

    if fr >= feasible_fr and cv <= low_cv:
        _enable(mask, name_to_idx, "throughput_priority", "exploration_high_F", "diversity_boost")

    if not bool(np.any(mask)):
        if "balanced" in name_to_idx:
            mask[name_to_idx["balanced"]] = True
        else:
            mask[:] = True
    return mask


def _metric(metrics, *names, default=0.0):
    if not isinstance(metrics, dict):
        return default
    for name in names:
        if name in metrics:
            try:
                return float(metrics[name])
            except (TypeError, ValueError):
                return default
    return default


def _pressure(pressure, key, metrics):
    if isinstance(pressure, dict) and key in pressure:
        try:
            return float(pressure[key])
        except (TypeError, ValueError):
            return 0.0
    return _metric(metrics, f"cv_{key}_mean", f"P_{key}_mean", default=0.0)


def _disable(mask, name_to_idx, *names):
    for name in names:
        idx = name_to_idx.get(name)
        if idx is not None:
            mask[idx] = False


def _enable(mask, name_to_idx, *names):
    for name in names:
        idx = name_to_idx.get(name)
        if idx is not None:
            mask[idx] = True
