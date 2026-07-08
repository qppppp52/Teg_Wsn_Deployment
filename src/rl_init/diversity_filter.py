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
    a = np.concatenate([xa, ya])
    b = np.concatenate([xb, yb])
    return float(np.mean(a != b))


def priority_l2_distance(ind_a, ind_b) -> float:
    a = np.concatenate([np.asarray(ind_a.rho_s), np.asarray(ind_a.rho_a)])
    b = np.concatenate([np.asarray(ind_b.rho_s), np.asarray(ind_b.rho_a)])
    if a.size == 0:
        return 0.0
    return float(np.linalg.norm(a - b) / np.sqrt(a.size))


def filter_population_by_diversity(candidates, min_distance, target_count):
    ordered = sorted(candidates, key=lambda item: float(item.get("quality_score", 0.0)), reverse=True)
    kept = []
    for item in ordered:
        if len(kept) >= target_count:
            break
        if not kept:
            kept.append(item)
            continue
        distances = []
        for prev in kept:
            if item.get("solution") is not None and prev.get("solution") is not None:
                distances.append(hamming_distance_deployment(item["solution"], prev["solution"]))
            distances.append(priority_l2_distance(item["individual"], prev["individual"]))
        if max(distances or [0.0]) >= float(min_distance):
            kept.append(item)
    return kept
