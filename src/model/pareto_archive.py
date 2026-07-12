"""外部Pareto档案"""
import copy
import numpy as np


def _dominates(a, b):
    if a.feasible and not b.feasible:
        return True
    if not a.feasible and b.feasible:
        return False
    if not a.feasible and not b.feasible:
        return a.cv < b.cv
    return (a.coverage >= b.coverage and a.rsum_capacity >= b.rsum_capacity and
            (a.coverage > b.coverage or a.rsum_capacity > b.rsum_capacity))


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
        # Skip exact duplicates: same (coverage, rsum) pair already stored
        for existing in self.solutions:
            if (abs(existing.coverage - sol.coverage) < 1e-9 and
                abs(existing.rsum_capacity - sol.rsum_capacity) < 1e-9):
                return  # duplicate, skip
        dominated = False
        to_remove = []
        for i, existing in enumerate(self.solutions):
            if _dominates(existing, sol):
                dominated = True
                break
            elif _dominates(sol, existing):
                to_remove.append(i)
        if not dominated:
            for i in sorted(to_remove, reverse=True):
                self.solutions.pop(i)
            # 深拷贝：防止后续repair变异导致归档解被污染
            self.solutions.append(copy.deepcopy(sol))

    def _prune(self):
        if len(self.solutions) <= self.max_size:
            return
        objs = self.get_objectives()
        crowding = _crowding_distance(objs)
        # 降序排列：保留拥挤距离大的解（极值点保持多样性）
        idx = np.argsort(crowding)[::-1]
        self.solutions = [self.solutions[i] for i in idx[:self.max_size]]

    def _has_any_feasible(self):
        return any(s.feasible for s in self.solutions)

    def get_objectives(self):
        if not self.solutions:
            return np.zeros((0, 2))
        return np.array([[s.coverage, s.rsum_capacity] for s in self.solutions])

    def get_feasible_objectives(self):
        feasible = [s for s in self.solutions if s.feasible]
        if not feasible:
            return np.zeros((0, 2))
        return np.array([[s.coverage, s.rsum_capacity] for s in feasible])

    def save(self, path):
        from src.io.result_io import save_pareto
        save_pareto(self.solutions, path)

    def __len__(self):
        return len(self.solutions)
