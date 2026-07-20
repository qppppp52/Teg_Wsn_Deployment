"""External Pareto archive with constrained dominance semantics."""
import copy

import numpy as np
from src.evaluation.constrained_dominance import compare_constraint_state, constrained_dominates


def _dominates(a, b):
    return constrained_dominates(a, b)


def _equivalent(a, b):
    report_a = getattr(a, "constraint_report", None)
    report_b = getattr(b, "constraint_report", None)
    feasible_a = bool(report_a.feasible) if report_a is not None else bool(getattr(a, "feasible", False))
    feasible_b = bool(report_b.feasible) if report_b is not None else bool(getattr(b, "feasible", False))
    if feasible_a != feasible_b:
        return False
    same_objectives = (
        abs(float(a.coverage) - float(b.coverage)) < 1.0e-9
        and abs(float(a.rsum_capacity) - float(b.rsum_capacity)) < 1.0e-9
    )
    if not same_objectives:
        return False
    if feasible_a:
        return True
    return compare_constraint_state(a, b) == 0


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
        if not sol.feasible and self._has_any_feasible():
            return
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

    def _prune(self):
        if len(self.solutions) <= self.max_size:
            return
        objs = self.get_objectives()
        crowding = _crowding_distance(objs)
        idx = np.argsort(crowding)[::-1]
        self.solutions = [self.solutions[i] for i in idx[: self.max_size]]

    def _has_any_feasible(self):
        return any(solution.feasible for solution in self.solutions)

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
