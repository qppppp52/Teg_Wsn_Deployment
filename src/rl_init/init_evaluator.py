"""Evaluation helpers for generated initial individuals."""
from __future__ import annotations

from src.evaluator.individual_evaluator import evaluate_individual


def evaluate_init_individual(individual, ctx, config, source_type="unknown") -> dict:
    sol, rep_ind = evaluate_individual(individual, ctx)
    if rep_ind is not None:
        individual = rep_ind
    return {"individual": individual, "solution": sol, "source_type": source_type, "quality_score": quality_score(sol, config)}


def get_selected_rsum_capacity(solution) -> float:
    return float(solution.rsum_capacity)


def quality_score(solution, config) -> float:
    norm = config.get("drl_init", config).get("normalization", {})
    rsum_min = float(norm.get("rsum_ref_min", 0.0))
    rsum_max = max(float(norm.get("rsum_ref_max", 2.0e7)), rsum_min + 1.0e-12)
    cv_ref = max(float(norm.get("cv_ref", 10.0)), 1.0e-12)
    repair_ref = max(float(norm.get("repair_iter_ref", 5.0)), 1.0e-12)
    coverage = float(getattr(solution, "coverage", 0.0))
    rsum_norm = min(max((get_selected_rsum_capacity(solution) - rsum_min) / (rsum_max - rsum_min), 0.0), 1.0)
    feasible = 1.0 if bool(getattr(solution, "feasible", False)) else 0.0
    cv_norm = min(float(getattr(solution, "cv", 0.0)) / cv_ref, 1.0)
    repair_norm = min(float(getattr(solution, "repair_iter", 0.0)) / repair_ref, 1.0)
    return 0.30 * coverage + 0.30 * rsum_norm + 0.25 * feasible - 0.30 * cv_norm - 0.10 * repair_norm
