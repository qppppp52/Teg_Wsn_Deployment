import numpy as np
from src.model.individual import Individual

def crossover(target, mutant, crossover_rate):
    t = Individual(len(target.rho_s), len(target.rho_a))
    m_s = np.random.rand(len(target.rho_s)) <= crossover_rate
    m_a = np.random.rand(len(target.rho_a)) <= crossover_rate
    t.rho_s = np.where(m_s, mutant.rho_s, target.rho_s)
    t.rho_a = np.where(m_a, mutant.rho_a, target.rho_a)
    t.clip()
    return t
