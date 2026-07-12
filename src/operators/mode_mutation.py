import numpy as np
from src.model.individual import Individual


def mutate(individual, F, strategy="standard_rand", population=None, idx=None, ctx=None):
    if population is None or len(population) < 3:
        return individual.copy()
    strategy = _alias_strategy(strategy)
    m = Individual(len(individual.rho_s), len(individual.rho_a))

    if strategy == "rand_2" and len(population) >= 5:
        r = _select_many(idx, len(population), 5)
        X1, X2, X3, X4, X5 = [population[j] for j in r]
        m.rho_s = X1.rho_s + F*(X2.rho_s - X3.rho_s) + F*(X4.rho_s - X5.rho_s)
        m.rho_a = X1.rho_a + F*(X2.rho_a - X3.rho_a) + F*(X4.rho_a - X5.rho_a)
    else:
        r1, r2, r3 = _select_three(idx, len(population))
        X1, X2, X3 = population[r1], population[r2], population[r3]
        if strategy == "standard_rand":
            m.rho_s = X1.rho_s + F*(X2.rho_s - X3.rho_s)
            m.rho_a = X1.rho_a + F*(X2.rho_a - X3.rho_a)
        elif strategy in {"coverage_guided", "throughput_guided", "best_1", "current_to_best_1"}:
            key = (lambda s: getattr(s, "rsum_capacity", 0.0)) if strategy == "throughput_guided" else _score
            best = _get_best(population, key)
            base = X1 if strategy == "best_1" else individual
            m.rho_s = base.rho_s + F*(best.rho_s - base.rho_s) + F*(X2.rho_s - X3.rho_s)
            m.rho_a = base.rho_a + F*(best.rho_a - base.rho_a) + F*(X2.rho_a - X3.rho_a)
        else:
            m.rho_s = X1.rho_s + F*(X2.rho_s - X3.rho_s)
            m.rho_a = X1.rho_a + F*(X2.rho_a - X3.rho_a)
    m.clip()
    return m


def _alias_strategy(strategy):
    aliases = {"rand_1": "standard_rand"}
    return aliases.get(strategy, strategy)


def _select_three(cur, sz):
    r = _select_many(cur, sz, 3)
    return r[0], r[1], r[2]


def _select_many(cur, sz, n):
    cand = list(range(sz))
    if cur is not None and cur in cand:
        cand.remove(cur)
    replace = len(cand) < n
    return np.random.choice(cand, n, replace=replace)


def _get_best(pop, key):
    return max(pop, key=key)


def _score(ind):
    return getattr(ind, "coverage", 0.0) + getattr(ind, "rsum_capacity", 0.0) / 1e9
