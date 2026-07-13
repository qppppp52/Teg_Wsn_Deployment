import numpy as np
from src.model.individual import Individual


def mutate(individual, F, strategy="standard_rand", population=None, idx=None, ctx=None, guide=None):
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
        elif strategy in {"coverage_guided", "rsum_capacity_guided", "best_1", "current_to_best_1"}:
            selected_guide = guide or select_pareto_guide(population)
            if strategy == "best_1":
                m.rho_s = selected_guide.rho_s + F*(X1.rho_s - X2.rho_s)
                m.rho_a = selected_guide.rho_a + F*(X1.rho_a - X2.rho_a)
            else:
                m.rho_s = individual.rho_s + F*(selected_guide.rho_s - individual.rho_s) + F*(X1.rho_s - X2.rho_s)
                m.rho_a = individual.rho_a + F*(selected_guide.rho_a - individual.rho_a) + F*(X1.rho_a - X2.rho_a)
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


def select_pareto_guide(population):
    """Sample a guide from the current constrained Pareto leading front."""
    if not population:
        raise ValueError("population must not be empty")
    feasible = [individual for individual in population if getattr(individual, "feasible", False)]
    candidates = feasible or list(population)
    if not feasible:
        best_cv = min(float(getattr(individual, "cv", np.inf)) for individual in candidates)
        candidates = [
            individual for individual in candidates
            if abs(float(getattr(individual, "cv", np.inf)) - best_cv) <= 1.0e-12
        ]
    else:
        candidates = [
            candidate for candidate in candidates
            if not any(
                _objective_dominates(other, candidate)
                for other in candidates
                if other is not candidate
            )
        ]
    return candidates[int(np.random.randint(len(candidates)))]


def _objective_dominates(a, b):
    return (
        a.coverage >= b.coverage
        and a.rsum_capacity >= b.rsum_capacity
        and (a.coverage > b.coverage or a.rsum_capacity > b.rsum_capacity)
    )
