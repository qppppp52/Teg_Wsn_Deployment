"""种群管理"""
import numpy as np
from src.model.individual import Individual, create_random_individual, create_greedy_energy_individual


class Population:
    def __init__(self, size, num_s, num_a):
        self.size = size
        self.individuals = []
        self.solutions = []
        self.num_s = num_s
        self.num_a = num_a

    def initialize(self, ctx, strategy="mixed"):
        self.individuals = []
        n_random = self.size
        n_greedy = 0
        if strategy == "mixed":
            n_greedy = max(1, self.size // 4)
            n_random = self.size - n_greedy
        for _ in range(n_random):
            self.individuals.append(create_random_individual(self.num_s, self.num_a))
        for _ in range(n_greedy):
            self.individuals.append(create_greedy_energy_individual(ctx))

    def get_statistics(self):
        if not self.solutions:
            return {"FR": 0, "CV_avg": float("inf"),
                    "Coverage_avg_feasible": float("nan"),
                    "Rsum_avg_feasible": float("nan"),
                    "Coverage_avg_all": 0.0, "Rsum_avg_all": 0.0,
                    "feasible_count": 0}
        feasible = [s for s in self.solutions if s.feasible]
        FR = len(feasible) / len(self.solutions) if self.solutions else 0
        CVs = [s.cv for s in self.solutions]
        cov_all = [s.coverage for s in self.solutions]
        rsum_all = [s.throughput for s in self.solutions]
        if feasible:
            cov_feas = float(np.mean([s.coverage for s in feasible]))
            rsum_feas = float(np.mean([s.throughput for s in feasible]))
        else:
            cov_feas = float("nan")
            rsum_feas = float("nan")
        return {"FR": FR, "CV_avg": float(np.mean(CVs)),
                "Coverage_avg_feasible": cov_feas,
                "Rsum_avg_feasible": rsum_feas,
                "Coverage_avg_all": float(np.mean(cov_all)),
                "Rsum_avg_all": float(np.mean(rsum_all)),
                "feasible_count": len(feasible)}
