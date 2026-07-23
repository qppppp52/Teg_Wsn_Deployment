"""Action masking rules for DQN-CR-MODE."""
from __future__ import annotations

import numpy as np

from src.constraints.cv_schema import CV_COMPONENT_KEYS


LOW_FR_ACTIONS = (
    "balanced", "energy_first", "link_first", "sink_first", "conservative_repair",
)
PRESSURE_ACTIONS = {
    "energy": ("balanced", "energy_first", "link_first", "sink_first", "conservative_repair"),
    "power": ("balanced", "energy_first", "link_first", "conservative_repair"),
    "link": ("balanced", "link_first", "energy_first", "conservative_repair"),
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
    return build_dqn_action_mask_details(state_metrics, actions, config)[0]


def build_dqn_action_mask_details(
    state_metrics: dict,
    actions: list,
    config: dict,
) -> tuple[np.ndarray, dict]:
    """Return the mask and an auditable explanation of the decision."""
    num_actions = len(actions)
    mask = np.ones(num_actions, dtype=bool)
    names = [str(action.get("name", "")) for action in actions]
    if num_actions == 0:
        return mask, _mask_details(names, state_metrics, False, "empty_action_space")
    dcfg = config.get("dqn", {}) if isinstance(config, dict) else {}
    action_mask_cfg = dcfg.get("action_mask", {})
    if not action_mask_cfg.get("enabled", True):
        return mask, _mask_details(names, state_metrics, False, "mask_disabled")

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
    dominant_margin = float(thresholds.get("dominant_margin", 0.05))
    stall_generations = int(thresholds.get("hv_stall_generations", 5))
    small_delta_hv = float(thresholds.get("small_delta_hv", 1.0e-5))

    ranked = _rank_pressures(pressure)
    top_name, top_value = ranked[0]
    second_value = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = max(top_value - second_value, 0.0)
    clear_dominant = bool(
        top_value >= high_pressure and margin >= dominant_margin and top_value > 0.0
    )
    reason = "unrestricted"
    if fr < low_fr:
        mask = _allow_only(names, LOW_FR_ACTIONS)
        reason = "low_feasible_rate"
    elif clear_dominant and top_name in PRESSURE_ACTIONS:
        mask = _allow_only(names, PRESSURE_ACTIONS[top_name])
        reason = f"clear_dominant_pressure:{top_name}"
    elif top_value >= high_pressure:
        reason = "ambiguous_dominant_pressure"

    if fr >= feasible_fr and cv <= low_cv:
        _enable(mask, names, *FEASIBLE_ACTIONS)
        reason += "|feasible"
    hv_change_is_small = delta_hv is not None and abs(delta_hv) <= small_delta_hv
    if fr >= low_fr and (hv_stall >= stall_generations or hv_change_is_small):
        _enable(mask, names, *STAGNATION_ACTIONS)
        reason += "|stagnation"

    fallback_used = False
    if not bool(np.any(mask)):
        fallback_used = True
        reason += "|fallback"
        if "balanced" in names:
            mask[names.index("balanced")] = True
        else:
            mask[:] = True
    details = _mask_details(
        names,
        state_metrics,
        fallback_used,
        reason,
        pressure=pressure,
        top_name=top_name,
        top_value=top_value,
        second_value=second_value,
        margin=margin,
        clear_dominant=clear_dominant,
    )
    details["num_allowed_actions"] = int(np.count_nonzero(mask))
    details["allowed_action_names"] = "|".join(
        name for name, flag in zip(names, mask) if flag
    )
    return mask, details


def action_mask_diagnostics(mask, actions, state_metrics: dict) -> dict:
    """Compatibility diagnostic helper for callers that only have a final mask."""
    names = [str(action.get("name", "")) for action in actions]
    details = _mask_details(names, state_metrics, False, "external_mask")
    details["num_allowed_actions"] = int(np.count_nonzero(np.asarray(mask, dtype=bool)))
    details["allowed_action_names"] = "|".join(
        name for name, flag in zip(names, np.asarray(mask, dtype=bool)) if flag
    )
    return details


def pressure_values(metrics: dict) -> dict:
    nested = metrics.get("pressure", {}) if isinstance(metrics, dict) else {}
    return {
        key: _pressure(nested, key, metrics)
        for key in CV_COMPONENT_KEYS
    }


def dominant_pressure_name(metrics: dict) -> str:
    ranked = _rank_pressures(pressure_values(metrics))
    return ranked[0][0] if ranked and ranked[0][1] > 0.0 else ""


def _mask_details(
    names,
    metrics,
    fallback_used,
    reason,
    *,
    pressure=None,
    top_name=None,
    top_value=None,
    second_value=None,
    margin=None,
    clear_dominant=None,
):
    pressure = pressure if pressure is not None else pressure_values(metrics)
    ranked = _rank_pressures(pressure)
    top_name = ranked[0][0] if top_name is None else top_name
    top_value = ranked[0][1] if top_value is None else top_value
    second_value = ranked[1][1] if second_value is None and len(ranked) > 1 else (
        0.0 if second_value is None else second_value
    )
    margin = max(float(top_value) - float(second_value), 0.0) if margin is None else margin
    return {
        "num_allowed_actions": len(names),
        "allowed_action_names": "|".join(names),
        "dominant_pressure": top_name if top_value > 0.0 else "none",
        "dominant_pressure_value": float(top_value),
        "second_pressure_value": float(second_value),
        "dominant_pressure_margin": float(margin),
        "dominant_pressure_clear": bool(clear_dominant) if clear_dominant is not None else False,
        "mask_fallback_used": bool(fallback_used),
        "mask_reason": str(reason),
    }


def _rank_pressures(pressure: dict) -> list[tuple[str, float]]:
    order = {key: index for index, key in enumerate(CV_COMPONENT_KEYS)}
    return sorted(
        ((key, float(pressure.get(key, 0.0))) for key in CV_COMPONENT_KEYS),
        key=lambda item: (-item[1], order[item[0]]),
    )


def _allow_only(names, allowed_names):
    return np.asarray([name in allowed_names for name in names], dtype=bool)


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
