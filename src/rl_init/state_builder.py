"""State builder for the DRL initialization environment."""
from __future__ import annotations

import numpy as np
from src.rl_init.action_mask import build_joint_action_mask
from src.rl_init.candidate_features import build_candidate_features

GLOBAL_FEATURE_NAMES = [
    "num_selected_sensors_norm", "num_selected_aps_norm", "remaining_sensor_slots_norm",
    "remaining_ap_slots_norm", "estimated_coverage_norm", "estimated_mean_pgrid_norm",
    "estimated_link_quality_norm", "estimated_energy_pressure_norm", "estimated_sink_pressure_norm",
    "step_progress", "last_cv_norm", "last_feasible_flag", "last_coverage",
    "last_rsum_norm", "last_repair_iter_norm",
]


def build_init_state(env) -> dict:
    cfg = env.cfg
    max_s = max(int(cfg.get("max_selected_sensors", env._deployment_limit("max_sensors", 20))), 1)
    max_a = max(int(cfg.get("max_selected_aps", env._deployment_limit("max_aps", 4))), 1)
    max_steps = max(int(cfg.get("max_steps_per_episode", 30)), 1)
    selected = list(env.selected_sensors | env.selected_aps)
    pgrid = np.asarray(getattr(env.ctx, "P_grid", np.zeros(env.ctx.num_candidates)), dtype=float)
    mean_p = float(np.mean(pgrid[selected])) if selected else 0.0
    p_norm = mean_p / (float(np.max(pgrid)) + 1.0e-12) if pgrid.size else 0.0

    estimated_coverage = _estimated_coverage(env)
    estimated_link = _estimated_link_quality(env)
    sink_pressure = _estimated_sink_pressure(env)
    last = env.last_eval_metrics or {}
    cv_ref = float(cfg.get("normalization", {}).get("cv_ref", 10.0))
    rsum_ref = float(cfg.get("normalization", {}).get("rsum_ref_max", 2.0e7))
    repair_ref = float(cfg.get("normalization", {}).get("repair_iter_ref", 5.0))

    global_features = np.asarray([
        len(env.selected_sensors) / max_s,
        len(env.selected_aps) / max_a,
        max(max_s - len(env.selected_sensors), 0) / max_s,
        max(max_a - len(env.selected_aps), 0) / max_a,
        estimated_coverage,
        p_norm,
        estimated_link,
        max(0.0, 1.0 - p_norm),
        sink_pressure,
        env.step_count / max_steps,
        min(float(last.get("cv", last.get("CV", 0.0))) / max(cv_ref, 1.0e-12), 1.0),
        1.0 if bool(last.get("feasible", False)) else 0.0,
        float(last.get("coverage", 0.0)),
        min(float(last.get("rsum", 0.0)) / max(rsum_ref, 1.0e-12), 1.0),
        min(float(last.get("repair_iter", 0.0)) / max(repair_ref, 1.0e-12), 1.0),
    ], dtype=np.float32)

    return {
        "candidate_features": build_candidate_features(env.ctx, env),
        "global_features": global_features,
        "joint_action_mask": build_joint_action_mask(env),
    }


def _estimated_coverage(env):
    mat = getattr(env.ctx, "coverage_matrix", None)
    if mat is None or not env.selected_sensors:
        return 0.0
    rows = np.asarray(list(env.selected_sensors), dtype=int)
    covered = np.any(np.asarray(mat)[rows] > 0, axis=0)
    return float(np.mean(covered)) if covered.size else 0.0


def _estimated_link_quality(env):
    if not env.selected_sensors or not env.selected_aps:
        return 0.0
    mat = getattr(env.ctx, "potential_rate_matrix", None)
    if mat is None:
        mat = getattr(env.ctx, "channel_gain_matrix", None)
    if mat is None:
        return 0.0
    mat = np.asarray(mat, dtype=float)
    vals = [mat[s, a] for s in env.selected_sensors for a in env.selected_aps]
    if not vals:
        return 0.0
    vmax = float(np.max(mat)) + 1.0e-12
    return float(np.mean(vals) / vmax)


def _estimated_sink_pressure(env):
    selected = env.selected_sensors | env.selected_aps
    if not selected:
        return 0.0
    neighbors = getattr(env.ctx, "neighbor_sets", None) or []
    risks = []
    for gid in selected:
        ns = neighbors[int(gid)] if int(gid) < len(neighbors) else []
        risks.append(sum(1 for item in ns if int(item) in selected) / max(len(ns), 1))
    return float(np.mean(risks)) if risks else 0.0
