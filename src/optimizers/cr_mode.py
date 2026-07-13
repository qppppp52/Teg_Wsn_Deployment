import numpy as np
import os

from src.optimizers.base_optimizer import BaseOptimizer
from src.model.population import Population
from src.model.pareto_archive import ParetoArchive
from src.operators.strategy_controller import StrategyController
from src.evaluator.individual_evaluator import evaluate_individual
from src.utils.logger import get_logger
from src.evaluation.generation_diagnostics import make_convergence_history, record_generation
from src.optimizers.generation_executor import GenerationExecutor, copy_solution_metrics

from src.io.population_snapshot import load_population_snapshot, save_population_snapshot
logger = get_logger("CR-MODE")


class CRMode(BaseOptimizer):
    def __init__(self, ctx, config):
        super().__init__(ctx, config)
        mc = config.get("mode", {})
        self.NP = mc.get("population_size", 60)
        self.Tmax = mc.get("max_generations", 100)
        self.strategy = StrategyController(config)
        self.archive = ParetoArchive(mc.get("archive_max_size", 200))
        self.population = None
        self.convergence_history = make_convergence_history()
        self.executor = GenerationExecutor(ctx, self.NP)
        self.evaluation_count = 0

    def initialize_population(self):
        im = self.ctx.index_mapping
        snapshot = self.config.get("runtime", {}).get("initial_population_snapshot")
        if snapshot and os.path.isfile(snapshot):
            population, _ = load_population_snapshot(
                snapshot, self.NP, im.num_sensor_candidates, im.num_ap_candidates
            )
            return population
        population = Population(self.NP, im.num_sensor_candidates, im.num_ap_candidates)
        population.initialize(self.ctx, strategy=self.config.get("mode", {}).get("initialization", "mixed"))
        if snapshot:
            seed = self.config.get("experiment", {}).get("seeds", [0])[0]
            save_population_snapshot(population, snapshot, seed)
        return population

    def evaluate_population(self):
        solutions = []
        for idx, ind in enumerate(self.population.individuals):
            sol, rep_ind = evaluate_individual(ind, self.ctx)
            if rep_ind is not None:
                ind = rep_ind
                self.population.individuals[idx] = ind
            copy_solution_metrics(ind, sol)
            solutions.append(sol)
            self.evaluation_count += 1
        self.population.solutions = solutions

    def run(self):
        self.population = self.initialize_population()
        self.evaluate_population()
        self.archive.update(self.population.solutions)
        self._record_convergence(0)
        self._log_generation(0)

        action = {
            "F": self.strategy.F,
            "CR": self.strategy.crossover_rate,
            "mutation_strategy": self.strategy.mutation_strategy,
            "repair_strategy": None,
        }
        for generation in range(1, self.Tmax + 1):
            result = self.executor.execute(self.population, action)
            self.population = result.population
            self.evaluation_count += result.evaluations
            self.archive.update(result.trial_solutions)
            self._record_convergence(generation)
            if generation % 10 == 0 or generation == self.Tmax:
                self._log_generation(generation)
        return self.archive

    def _record_convergence(self, gen):
        record_generation(self.convergence_history, self.population, self.archive, self.config)

    def _log_generation(self, generation):
        stats = self.population.get_statistics()
        cov = stats["Coverage_avg_feasible"]
        rsum = stats["Rsum_avg_feasible"]
        cov_str = f"{cov:.3f}" if not np.isnan(cov) else "NaN"
        rsum_str = f"{rsum:.1f}" if not np.isnan(rsum) else "NaN"
        logger.info(
            f"Gen {generation:4d} | FR={stats['FR']:.3f} "
            f"CV={stats['CV_avg']:.4f} Cov={cov_str} Rsum={rsum_str} "
            f"Archive={len(self.archive)} Evals={self.evaluation_count}"
        )
