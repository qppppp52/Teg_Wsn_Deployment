"""Reward functions for PPO deployment initialization."""
from __future__ import annotations

import numpy as np
from src.evaluator.individual_evaluator import evaluate_individual


def compute_step_reward(prev_summary: dict, new_summary: dict, info: dict, reward_cfg: dict) -> float:
    w_cov = float(reward_cfg.get("coverage_step", 0.10))
    w_energy = float(reward_cfg.get("energy_step", 0.05))
    w_link = float(reward_cfg.get("link_step", 0.05))
    w_invalid = float(reward_cfg.get("invalid_action_penalty", 0.20))
    w_duplicate = float(reward_cfg.get("duplicate_penalty", 0.10))
    w_sink = float(reward_cfg.get("sink_risk", 0.05))
    reward = 0.0
    reward += w_cov * (float(new_summary.get("estimated_coverage", 0.0)) - float(prev_summary.get("estimated_coverage", 0.0)))
    reward += w_energy * (float(new_summary.get("estimated_energy", 0.0)) - float(prev_summary.get("estimated_energy", 0.0)))
    reward += w_link * (float(new_summary.get("estimated_link", 0.0)) - float(prev_summary.get("estimated_link", 0.0)))
    reward -= w_sink * max(0.0, float(new_summary.get("estimated_sink_pressure", 0.0)) - float(prev_summary.get("estimated_sink_pressure", 0.0)))
    if info.get("invalid_action"):
        reward -= w_invalid
    if info.get("duplicate_or_conflict"):
        reward -= w_duplicate
    return float(reward)


def compute_terminal_reward(env, reward_cfg: dict, norm_cfg: dict) -> tuple[float, dict]:
    individual = env.build_current_individual()
    solution, _ = evaluate_individual(
        individual,
        env.ctx,
        max_repair_iter=int(env.cfg.get("max_repair_iter", env.ctx.config.get("constraints", {}).get("max_repair_iter", 5))),
    )
    coverage = float(getattr(solution, "coverage", 0.0))
    rsum_actual = float(solution.metadata.get("throughput_actual", getattr(solution, "throughput", 0.0)))
    rsum_capacity = float(solution.metadata.get("throughput_capacity", getattr(solution, "throughput", 0.0)))
    metric = env.config.get("objectives", {}).get("throughput_metric", "actual")
    rsum = rsum_capacity if metric == "capacity" else rsum_actual
    rsum_ref = max(float(norm_cfg.get("rsum_ref_max", 2.0e7)), 1.0e-12)
    cv = float(getattr(solution, "cv", 0.0))
    cv_ref = max(float(norm_cfg.get("cv_ref", 10.0)), 1.0e-12)
    repair_iter = float(getattr(solution, "repair_iter", 0.0))
    repair_ref = max(float(norm_cfg.get("repair_iter_ref", 5.0)), 1.0e-12)
    feasible = bool(getattr(solution, "feasible", False))
    too_few = env.too_few_nodes()
    invalid_rate = env.invalid_action_count / max(env.step_count, 1)
    diversity_bonus = 1.0 if env.selection_order_sensors and env.selection_order_aps else 0.0

    reward = 0.0
    reward += float(reward_cfg.get("coverage", 0.25)) * coverage
    reward += float(reward_cfg.get("rsum", 0.25)) * min(rsum / rsum_ref, 1.0)
    reward += float(reward_cfg.get("feasible_bonus", 0.30)) * (1.0 if feasible else 0.0)
    reward -= float(reward_cfg.get("cv_penalty", 0.35)) * min(cv / cv_ref, 1.0)
    reward -= float(reward_cfg.get("repair_cost_penalty", 0.10)) * min(repair_iter / repair_ref, 1.0)
    reward += float(reward_cfg.get("diversity_bonus", 0.08)) * diversity_bonus
    reward -= float(reward_cfg.get("too_few_nodes_penalty", 0.15)) * (1.0 if too_few else 0.0)
    reward -= float(reward_cfg.get("invalid_action_penalty", 0.20)) * invalid_rate

    metrics = {
        "coverage": coverage,
        "rsum": rsum,
        "rsum_actual": rsum_actual,
        "rsum_capacity": rsum_capacity,
        "throughput_metric": metric,
        "cv": cv,
        "feasible": feasible,
        "repair_iter": repair_iter,
        "repair_success": bool(getattr(solution, "repair_success", feasible)),
        "cv_deploy": float(getattr(solution, "cv_deploy", 0.0)),
        "cv_link": float(getattr(solution, "cv_link", 0.0)),
        "cv_capacity": float(getattr(solution, "cv_capacity", 0.0)),
        "cv_energy": float(getattr(solution, "cv_energy", 0.0)),
        "cv_sink": float(getattr(solution, "cv_sink", 0.0)),
        "cv_service": float(getattr(solution, "cv_service", 0.0)),
        "invalid_action_rate": invalid_rate,
        "too_few_nodes": too_few,
        "solution": solution,
    }
    return float(np.clip(reward, -2.0, 2.0)), metrics
