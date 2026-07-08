"""Evaluation helpers for generated initial individuals."""
from __future__ import annotations

from src.evaluator.individual_evaluator import evaluate_individual


def evaluate_init_individual(individual, ctx, config, source_type="unknown") -> dict:
    cfg = config.get("drl_init", config)
    sol, rep_ind = evaluate_individual(
        individual,
        ctx,
        max_repair_iter=int(cfg.get("max_repair_iter", ctx.config.get("constraints", {}).get("max_repair_iter", 5))),
    )
    if rep_ind is not None:
        individual = rep_ind
    return {"individual": individual, "solution": sol, "source_type": source_type, "quality_score": quality_score(sol, config)}


def quality_score(solution, config) -> float:
    norm = config.get("drl_init", config).get("normalization", {})
    rsum_ref = max(float(norm.get("rsum_ref_max", 2.0e7)), 1.0e-12)
    cv_ref = max(float(norm.get("cv_ref", 10.0)), 1.0e-12)
    repair_ref = max(float(norm.get("repair_iter_ref", 5.0)), 1.0e-12)
    coverage = float(getattr(solution, "coverage", 0.0))
    metric = config.get("objectives", {}).get("throughput_metric", "actual")
    metadata_key = "throughput_capacity" if metric == "capacity" else "throughput_actual"
    rsum = float(solution.metadata.get(metadata_key, solution.throughput))
    rsum_norm = min(rsum / rsum_ref, 1.0)
    feasible = 1.0 if bool(getattr(solution, "feasible", False)) else 0.0
    cv_norm = min(float(getattr(solution, "cv", 0.0)) / cv_ref, 1.0)
    repair_norm = min(float(getattr(solution, "repair_iter", 0.0)) / repair_ref, 1.0)
    return 0.30 * coverage + 0.30 * rsum_norm + 0.25 * feasible - 0.30 * cv_norm - 0.10 * repair_norm
