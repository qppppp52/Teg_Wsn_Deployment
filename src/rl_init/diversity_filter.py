"""Diversity filters for DRL-generated initial populations."""
from __future__ import annotations

import numpy as np


def hamming_distance_deployment(sol_a, sol_b) -> float:
    xa = np.asarray(getattr(sol_a, "x", []), dtype=int)
    xb = np.asarray(getattr(sol_b, "x", []), dtype=int)
    ya = np.asarray(getattr(sol_a, "y", []), dtype=int)
    yb = np.asarray(getattr(sol_b, "y", []), dtype=int)
    if xa.size == 0 or xb.size == 0:
        return 0.0
    return float(np.mean(np.concatenate([xa, ya]) != np.concatenate([xb, yb])))


def priority_l2_distance(ind_a, ind_b) -> float:
    a = np.concatenate([np.asarray(ind_a.rho_s), np.asarray(ind_a.rho_a)])
    b = np.concatenate([np.asarray(ind_b.rho_s), np.asarray(ind_b.rho_a)])
    if a.size == 0:
        return 0.0
    return float(np.linalg.norm(a - b) / np.sqrt(a.size))


def objective_space_distance(sol_a, sol_b, rsum_scale=2.0e8) -> float:
    coverage_delta = float(getattr(sol_a, "coverage", 0.0)) - float(getattr(sol_b, "coverage", 0.0))
    rsum_a = float(getattr(sol_a, "rsum_capacity", 0.0))
    rsum_b = float(getattr(sol_b, "rsum_capacity", 0.0))
    rsum_delta = (rsum_a - rsum_b) / max(float(rsum_scale), 1.0)
    return float(np.hypot(coverage_delta, rsum_delta))


def filter_population_by_diversity(
    candidates,
    min_distance=None,
    target_count=None,
    *,
    config=None,
):
    """Select diverse high-quality candidates using deployment-first rules."""
    if target_count is None:
        raise ValueError("target_count is required")
    cfg = (config or {}).get("drl_init", config or {})
    diversity = cfg.get("diversity", {}) if isinstance(cfg, dict) else {}
    legacy_distance = float(min_distance if min_distance is not None else 0.05)
    hard_hamming = float(diversity.get("min_hamming_distance", legacy_distance))
    weak_hamming = float(diversity.get("weak_hamming_distance", min(hard_hamming, 0.05)))
    min_priority = float(diversity.get("min_priority_l2_distance", legacy_distance))
    min_objective = float(diversity.get("min_objective_distance", 0.03))
    rsum_scale = float(cfg.get("normalization", {}).get("rsum_ref_max", 2.0e8)) if isinstance(cfg, dict) else 2.0e8

    ordered = sorted(
        candidates,
        key=lambda item: float(item.get("quality_score", 0.0)),
        reverse=True,
    )
    kept = []
    for item in ordered:
        if len(kept) >= int(target_count):
            item["diversity_reject_reason"] = "target_already_reached"
            continue
        if not kept:
            _record_decision(item, np.nan, np.nan, np.nan, "first_candidate", "")
            kept.append(item)
            continue

        hamming = [
            hamming_distance_deployment(item["solution"], previous["solution"])
            for previous in kept
        ]
        priority = [
            priority_l2_distance(item["individual"], previous["individual"])
            for previous in kept
        ]
        objective = [
            objective_space_distance(item["solution"], previous["solution"], rsum_scale)
            for previous in kept
        ]
        min_hamming = min(hamming)
        min_priority_value = min(priority)
        min_objective_value = min(objective)

        accept_reason = ""
        if min_hamming >= hard_hamming:
            accept_reason = "deployment_hamming"
        elif min_objective_value >= min_objective:
            accept_reason = "objective_space"
        elif min_priority_value >= min_priority and min_hamming >= weak_hamming:
            accept_reason = "priority_with_weak_deployment"

        if accept_reason:
            _record_decision(
                item,
                min_hamming,
                min_priority_value,
                min_objective_value,
                accept_reason,
                "",
            )
            kept.append(item)
        else:
            _record_decision(
                item,
                min_hamming,
                min_priority_value,
                min_objective_value,
                "",
                "insufficient_diversity",
            )
    return kept


def _record_decision(item, hamming, priority, objective, accept_reason, reject_reason):
    item["min_deployment_hamming"] = float(hamming)
    item["min_priority_l2"] = float(priority)
    item["min_objective_distance"] = float(objective)
    item["diversity_accept_reason"] = accept_reason
    item["diversity_reject_reason"] = reject_reason
