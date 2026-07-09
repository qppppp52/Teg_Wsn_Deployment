"""Action masking rules for DQN-CR-MODE."""
from __future__ import annotations

import numpy as np


def build_dqn_action_mask(state_metrics: dict, actions: list, config: dict) -> np.ndarray:
    """
    Build a pressure-aware boolean mask for the discrete DQN action space.

    True means the action may be selected. The mask uses conservative
    whitelists for clearly constrained phases and falls back to balanced/all
    actions if a local rule would otherwise block every action.
    """
    num_actions = len(actions)
    mask = np.ones(num_actions, dtype=bool)
    if num_actions == 0:
        return mask

    dcfg = config.get("dqn", {}) if isinstance(config, dict) else {}
    if not dcfg.get("action_mask_enabled", True):
        return mask

    fr = _metric(state_metrics, "FR_after_repair", "FR", "feasible_ratio", default=0.0)
    cv = _metric(state_metrics, "CV_after_repair_mean", "CV_mean", "cv", default=0.0)
    delta_hv = _metric(state_metrics, "delta_HV", "delta_HV_norm", default=0.0)
    hv_stall = int(_metric(state_metrics, "hv_stall_generations", "HV_stall", default=0))
    pressure = pressure_values(state_metrics)

    thresholds = dcfg.get("mask", {})
    low_fr = float(thresholds.get("low_fr", 0.2))
    feasible_fr = float(thresholds.get("feasible_fr", 0.8))
    low_cv = float(thresholds.get("low_cv", 1.0e-3))
    high_pressure = float(thresholds.get("high_pressure", 0.4))
    stall_generations = int(thresholds.get("hv_stall_generations", 5))
    small_delta_hv = float(thresholds.get("small_delta_hv", 1.0e-5))

    names = [str(action.get("name", "")) for action in actions]
    dominant = dominant_pressure_name(state_metrics)
    dominant_value = pressure.get(dominant, 0.0) if dominant else 0.0

    if fr < low_fr:
        mask = _allow_only(names, {
            "balanced", "energy_first", "link_first", "capacity_first", "sink_first", "conservative_repair",
        })
    elif dominant == "energy" and dominant_value > high_pressure:
        mask = _allow_only(names, {"balanced", "energy_first", "conservative_repair", "sink_first"})
    elif dominant == "link" and dominant_value > high_pressure:
        mask = _allow_only(names, {"balanced", "link_first", "capacity_first", "throughput_priority"})
    elif dominant == "capacity" and dominant_value > high_pressure:
        mask = _allow_only(names, {"balanced", "capacity_first", "link_first", "conservative_repair"})
    elif dominant == "sink" and dominant_value > high_pressure:
        mask = _allow_only(names, {"balanced", "sink_first", "energy_first", "conservative_repair"})

    if fr >= feasible_fr and cv <= low_cv:
        _enable(mask, names, "throughput_priority", "exploration_high_F", "diversity_boost", "exploitation_low_F")

    if hv_stall >= stall_generations or abs(delta_hv) <= small_delta_hv:
        _enable(mask, names, "diversity_boost", "exploration_high_F")

    return _ensure_any(mask, names)


def action_mask_diagnostics(mask, actions, state_metrics: dict) -> dict:
    names = [str(action.get("name", "")) for action in actions]
    allowed = [name for name, allowed_flag in zip(names, np.asarray(mask, dtype=bool)) if allowed_flag]
    return {
        "num_allowed_actions": len(allowed),
        "allowed_action_names": "|".join(allowed),
        "dominant_pressure": dominant_pressure_name(state_metrics) or "none",
    }


def pressure_values(metrics: dict) -> dict:
    nested = metrics.get("pressure", {}) if isinstance(metrics, dict) else {}
    return {
        key: _pressure(nested, key, metrics)
        for key in ["deploy", "link", "capacity", "energy", "sink", "service"]
    }


def dominant_pressure_name(metrics: dict) -> str:
    values = pressure_values(metrics)
    if not values:
        return ""
    return max(values, key=values.get)


def _allow_only(names, allowed_names):
    return np.asarray([name in allowed_names for name in names], dtype=bool)


def _ensure_any(mask, names):
    mask = np.asarray(mask, dtype=bool)
    if bool(np.any(mask)):
        return mask
    if "balanced" in names:
        mask[names.index("balanced")] = True
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
    return _metric(metrics, f"P_{key}_mean", f"cv_{key}_mean", default=0.0)


def _enable(mask, names, *action_names):
    for name in action_names:
        if name in names:
            mask[names.index(name)] = True
