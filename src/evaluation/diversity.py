"""Diversity metrics for objective and decision spaces."""
from __future__ import annotations

import numpy as np


def objective_space_diversity(solutions):
    """Return mean normalized neighbor distance in objective space."""
    feasible = [s for s in solutions if getattr(s, "feasible", False)]
    if len(feasible) < 2:
        return 0.0
    objs = np.array([[s.coverage, s.rsum_capacity] for s in feasible], dtype=float)
    lo = objs.min(axis=0)
    hi = objs.max(axis=0)
    norm = (objs - lo) / (hi - lo + 1e-12)
    norm = norm[np.argsort(norm[:, 0])]
    dists = np.linalg.norm(np.diff(norm, axis=0), axis=1)
    return float(np.mean(dists)) if len(dists) else 0.0


def decision_space_diversity(individuals):
    """Return mean nearest-neighbor distance among encoded individuals."""
    if len(individuals) < 2:
        return 0.0
    arr = np.array([np.concatenate([ind.rho_s, ind.rho_a]) for ind in individuals], dtype=float)
    nearest = []
    for idx, row in enumerate(arr):
        others = np.delete(arr, idx, axis=0)
        nearest.append(float(np.min(np.linalg.norm(others - row, axis=1))))
    return float(np.mean(nearest))


def compute_diversity(solutions):
    """Backward-compatible objective diversity alias."""
    return objective_space_diversity(solutions)
