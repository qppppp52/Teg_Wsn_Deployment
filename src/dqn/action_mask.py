"""Action masking rules for DQN-CR-MODE."""
from __future__ import annotations

import numpy as np

LOW_FR_ACTIONS = (
    "balanced", "energy_first", "link_first", "sink_first", "conservative_repair",
)
PRESSURE_ACTIONS = {
    "energy": ("balanced", "energy_first", "conservative_repair", "sink_first"),
    "power": ("balanced", "energy_first", "link_first", "rsum_search_priority"),
    "link": ("balanced", "link_first", "energy_first", "rsum_search_priority"),
    "sink": ("balanced", "sink_first", "energy_first", "conservative_repair"),
    "service": ("balanced", "link_first", "energy_first", "conservative_repair"),
    "deploy": ("balanced", "exploration_high_F", "diversity_boost"),
}
FEASIBLE_ACTIONS = (
    "rsum_search_priority", "exploration_high_F", "diversity_boost",
    "exploitation_low_F",
)
STAGNATION_ACTIONS = ("diversity_boost", "exploration_high_F")
MASK_RULE_ACTION_NAMES = frozenset(
    LOW_FR_ACTIONS
    + tuple(name for actions in PRESSURE_ACTIONS.values() for name in actions)
    + FEASIBLE_ACTIONS
    + STAGNATION_ACTIONS
)


def build_dqn_action_mask(state_metrics: dict, actions: list, config: dict) -> np.ndarray:
    """Build a conservative pressure-aware mask with a non-empty fallback."""
    num_actions = len(actions)
    mask = np.ones(num_actions, dtype=bool)
    if num_actions == 0:
        return mask
    dcfg = config.get("dqn", {}) if isinstance(config, dict) else {}
    action_mask_cfg = dcfg.get("action_mask", {})
    if not action_mask_cfg.get("enabled", True):
        return mask
    fr = _metric(state_metrics, "FR_after_repair", "FR", "feasible_ratio", default=0.0)
    cv = _metric(state_metrics, "CV_after_repair_mean", "CV_mean", "cv", default=0.0)
    delta_hv = _metric(state_metrics, "delta_HV", "delta_HV_norm", default=None)
    hv_stall = int(_metric(state_metrics, "hv_stall_generations", "HV_stall", default=0))
    pressure = pressure_values(state_metrics)
    thresholds = action_mask_cfg.get("thresholds", {})
    low_fr = float(thresholds.get("low_fr", 0.2))
    feasible_fr = float(thresholds.get("feasible_fr", 0.8))
    low_cv = float(thresholds.get("low_cv", 1.0e-3))
    high_pressure = float(thresholds.get("high_pressure", 0.4))
    stall_generations = int(thresholds.get("hv_stall_generations", 5))
    small_delta_hv = float(thresholds.get("small_delta_hv", 1.0e-5))
    names = [str(action.get("name", "")) for action in actions]
    dominant = dominant_pressure_name(state_metrics)
    if fr < low_fr:
        mask = _allow_only(names, LOW_FR_ACTIONS)
    elif dominant in PRESSURE_ACTIONS and pressure.get(dominant, 0.0) > high_pressure:
        mask = _allow_only(names, PRESSURE_ACTIONS[dominant])
    if fr >= feasible_fr and cv <= low_cv:
        _enable(mask, names, *FEASIBLE_ACTIONS)
    hv_change_is_small = delta_hv is not None and abs(delta_hv) <= small_delta_hv
    if fr >= low_fr and (hv_stall >= stall_generations or hv_change_is_small):
        _enable(mask, names, *STAGNATION_ACTIONS)
    return _ensure_any(mask, names)


def action_mask_diagnostics(mask, actions, state_metrics: dict) -> dict:
    names = [str(action.get("name", "")) for action in actions]
    allowed = [name for name, flag in zip(names, np.asarray(mask, dtype=bool)) if flag]
    return {
        "num_allowed_actions": len(allowed),
        "allowed_action_names": "|".join(allowed),
        "dominant_pressure": dominant_pressure_name(state_metrics) or "none",
    }


def pressure_values(metrics: dict) -> dict:
    nested = metrics.get("pressure", {}) if isinstance(metrics, dict) else {}
    return {
        key: _pressure(nested, key, metrics)
        for key in ("deploy", "link", "power", "energy", "sink", "service")
    }


def dominant_pressure_name(metrics: dict) -> str:
    values = pressure_values(metrics)
    dominant = max(values, key=values.get) if values else ""
    return dominant if dominant and values[dominant] > 0.0 else ""


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
                value = float(metrics[name])
                return value if np.isfinite(value) else default
            except (TypeError, ValueError):
                return default
    return default


def _pressure(pressure, key, metrics):
    if isinstance(pressure, dict) and key in pressure:
        try:
            value = float(pressure[key])
            return value if np.isfinite(value) else 0.0
        except (TypeError, ValueError):
            return 0.0
    return _metric(metrics, f"P_{key}_mean", f"cv_{key}_mean", default=0.0)


def _enable(mask, names, *action_names):
    for name in action_names:
        if name in names:
            mask[names.index(name)] = True