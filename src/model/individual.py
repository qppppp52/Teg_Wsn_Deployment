"""MODE个体编码：X = [rho_s, rho_a]"""
import numpy as np


class Individual:
    def __init__(self, num_s, num_a):
        self.rho_s = np.random.rand(num_s)
        self.rho_a = np.random.rand(num_a)
        self.coverage = 0.0
        self.throughput = 0.0
        self.cv = float("inf")
        self.feasible = False

    def copy(self):
        new = Individual(len(self.rho_s), len(self.rho_a))
        new.rho_s = self.rho_s.copy()
        new.rho_a = self.rho_a.copy()
        new.coverage = self.coverage
        new.throughput = self.throughput
        new.cv = self.cv
        new.feasible = self.feasible
        return new

    def clip(self):
        np.clip(self.rho_s, 0.0, 1.0, out=self.rho_s)
        np.clip(self.rho_a, 0.0, 1.0, out=self.rho_a)


def create_random_individual(num_s, num_a):
    ind = Individual(num_s, num_a)
    ind.rho_s = np.random.rand(num_s)
    ind.rho_a = np.random.rand(num_a)
    return ind


def create_greedy_energy_individual(ctx):
    im = ctx.index_mapping
    ind = Individual(im.num_sensor_candidates, im.num_ap_candidates)
    P_s = ctx.P_grid[im.Ls_local_to_global]
    rng = P_s.max() - P_s.min() + 1e-12
    ind.rho_s = (P_s - P_s.min()) / rng
    P_a = ctx.P_grid[im.La_local_to_global]
    rng2 = P_a.max() - P_a.min() + 1e-12
    ind.rho_a = (P_a - P_a.min()) / rng2
    ind.rho_s = np.clip(ind.rho_s + np.random.normal(0, 0.05, len(ind.rho_s)), 0, 1)
    ind.rho_a = np.clip(ind.rho_a + np.random.normal(0, 0.05, len(ind.rho_a)), 0, 1)
    return ind
