import numpy as np
from src.model.individual import Individual

def mutate(individual, F, strategy="standard_rand", population=None, idx=None, ctx=None):
    if population is None or len(population)<3:
        return individual.copy()
    r1,r2,r3 = _select_three(idx, len(population))
    m = Individual(len(individual.rho_s), len(individual.rho_a))
    X1,X2,X3 = population[r1], population[r2], population[r3]
    if strategy == "standard_rand":
        m.rho_s = X1.rho_s + F*(X2.rho_s - X3.rho_s)
        m.rho_a = X1.rho_a + F*(X2.rho_a - X3.rho_a)
    elif strategy == "coverage_guided" and ctx is not None:
        best = _get_best(population, lambda s: s.coverage)
        m.rho_s = individual.rho_s + F*(best.rho_s-individual.rho_s) + F*(X2.rho_s-X3.rho_s)
        m.rho_a = individual.rho_a + F*(best.rho_a-individual.rho_a) + F*(X2.rho_a-X3.rho_a)
    elif strategy == "throughput_guided" and ctx is not None:
        best = _get_best(population, lambda s: s.throughput)
        m.rho_s = individual.rho_s + F*(best.rho_s-individual.rho_s) + F*(X2.rho_s-X3.rho_s)
        m.rho_a = individual.rho_a + F*(best.rho_a-individual.rho_a) + F*(X2.rho_a-X3.rho_a)
    else:
        m.rho_s = X1.rho_s + F*(X2.rho_s - X3.rho_s)
        m.rho_a = X1.rho_a + F*(X2.rho_a - X3.rho_a)
    m.clip()
    return m

def _select_three(cur, sz):
    cand = list(range(sz))
    if cur is not None: cand.remove(cur)
    c = np.random.choice(cand, 3, replace=False)
    return c[0],c[1],c[2]

def _get_best(pop, key):
    return max(pop, key=key)
