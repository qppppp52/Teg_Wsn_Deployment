"""External Pareto archive with constrained dominance semantics."""
import copy

import numpy as np
from src.constraints.report_freshness import require_fresh_constraint_report
from src.evaluation.constrained_dominance import constrained_dominates


PARETO_ARCHIVE_SEMANTICS_VERSION = 2


def _dominates(a, b):
    return constrained_dominates(a, b)


def _equivalent(a, b):
    """Archive entries are fresh and feasible, so equality is objective equality."""
    return (
        abs(float(a.coverage) - float(b.coverage)) < 1.0e-9
        and abs(float(a.rsum_capacity) - float(b.rsum_capacity)) < 1.0e-9
    )


def _crowding_distance(objs):
    n = len(objs)
    if n <= 2:
        return np.full(n, float("inf"))
    dist = np.zeros(n)
    for m in range(objs.shape[1]):
        idx = np.argsort(objs[:, m])
        dist[idx[0]] = float("inf")
        dist[idx[-1]] = float("inf")
        span = objs[idx[-1], m] - objs[idx[0], m]
        if span < 1e-12:
            continue
        for i in range(1, n - 1):
            dist[idx[i]] += (objs[idx[i + 1], m] - objs[idx[i - 1], m]) / span
    return dist


class ParetoArchive:
    def __init__(self, max_size=200):
        self.solutions = []
        self.max_size = max_size

    def update(self, solutions):
        for sol in solutions:
            self._insert(sol)
        self._prune()

    def _insert(self, sol):
        report = require_fresh_constraint_report(sol, "Pareto archive insertion")
        if not report.feasible:
            return False
        dominated = False
        to_remove = []
        for i, existing in enumerate(self.solutions):
            if _dominates(existing, sol) or _equivalent(existing, sol):
                dominated = True
                break
            if _dominates(sol, existing):
                to_remove.append(i)
        if not dominated:
            for i in sorted(to_remove, reverse=True):
                self.solutions.pop(i)
            self.solutions.append(copy.deepcopy(sol))
            return True
        return False

    def _prune(self):
        if len(self.solutions) <= self.max_size:
            return
        objs = self.get_objectives()
        crowding = _crowding_distance(objs)
        idx = np.argsort(crowding)[::-1]
        self.solutions = [self.solutions[i] for i in idx[: self.max_size]]


    def get_objectives(self):
        if not self.solutions:
            return np.zeros((0, 2))
        return np.array(
            [[solution.coverage, solution.rsum_capacity] for solution in self.solutions]
        )

    def get_feasible_objectives(self):
        feasible = [solution for solution in self.solutions if solution.feasible]
        if not feasible:
            return np.zeros((0, 2))
        return np.array(
            [[solution.coverage, solution.rsum_capacity] for solution in feasible]
        )

    def save(self, path, ctx=None):
        from src.io.result_io import save_pareto

        save_pareto(self.solutions, path, ctx)

    def __len__(self):
        return len(self.solutions)
