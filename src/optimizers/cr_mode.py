import os
import time
import numpy as np

from src.evaluation.generation_diagnostics import make_convergence_history, record_generation
from src.evaluator.individual_evaluator import evaluate_individual
from src.io.population_snapshot import load_population_snapshot, save_population_snapshot
from src.model.pareto_archive import ParetoArchive
from src.model.population import Population
from src.operators.strategy_controller import StrategyController
from src.optimizers.base_optimizer import BaseOptimizer
from src.optimizers.generation_executor import GenerationExecutor, copy_solution_metrics
from src.utils.logger import get_logger


class CRMode(BaseOptimizer):
    def __init__(self, ctx, config):
        super().__init__(ctx, config)
        mode_config = config.get("mode", {})
        self.NP = mode_config.get("population_size", 60)
        self.Tmax = mode_config.get("max_generations", 100)
        self.strategy = StrategyController(config)
        self.archive = ParetoArchive(mode_config.get("archive_max_size", 200))
        self.population = None
        self.convergence_history = make_convergence_history()
        self.executor = GenerationExecutor(ctx, self.NP)
        self.evaluation_count = 0
        self.logger = get_logger("CR-MODE")
        self.gen0_evaluation_seconds = 0.0
        self.cr_mode_search_seconds = 0.0
        self.online_optimization_seconds = 0.0
        self.total_end_to_end_seconds = 0.0

    def initialize_population(self):
        mapping = self.ctx.index_mapping
        snapshot = self.config.get("runtime", {}).get("initial_population_snapshot")
        if snapshot and os.path.isfile(snapshot):
            population, _ = load_population_snapshot(
                snapshot,
                self.NP,
                mapping.num_sensor_candidates,
                mapping.num_ap_candidates,
            )
            return population
        population = Population(
            self.NP,
            mapping.num_sensor_candidates,
            mapping.num_ap_candidates,
        )
        population.initialize(
            self.ctx,
            strategy=self.config.get("mode", {}).get("initialization", "mixed"),
        )
        if snapshot:
            seed = self.config.get("experiment", {}).get("seeds", [0])[0]
            save_population_snapshot(population, snapshot, seed)
        return population

    def evaluate_population(self):
        solutions = []
        for index, individual in enumerate(self.population.individuals):
            solution, repaired = evaluate_individual(individual, self.ctx)
            if repaired is not None:
                individual = repaired
                self.population.individuals[index] = individual
            copy_solution_metrics(individual, solution)
            solutions.append(solution)
            self.evaluation_count += 1
        self.population.solutions = solutions

    def run(self):
        total_start = time.time()
        self.population = self.initialize_population()

        gen0_start = time.time()
        self.evaluate_population()
        self.gen0_evaluation_seconds = time.time() - gen0_start
        self._on_gen0_evaluated(self.NP)
        self.archive.update(self.population.solutions)
        self._record_convergence(0)
        self._log_generation(0)

        action = {
            "F": self.strategy.F,
            "CR": self.strategy.crossover_rate,
            "mutation_strategy": self.strategy.mutation_strategy,
            "repair_strategy": None,
        }
        search_start = time.time()
        for generation in range(1, self.Tmax + 1):
            result = self.executor.execute(self.population, action)
            self.population = result.population
            self.evaluation_count += result.evaluations
            self._on_trials_evaluated(result.evaluations)
            self.archive.update(result.trial_solutions)
            self._record_convergence(generation)
            if generation % 10 == 0 or generation == self.Tmax:
                self._log_generation(generation)
        self.cr_mode_search_seconds = time.time() - search_start
        self.online_optimization_seconds = (
            self.gen0_evaluation_seconds + self.cr_mode_search_seconds
        )
        self.total_end_to_end_seconds = time.time() - total_start
        return self.archive

    def _on_gen0_evaluated(self, count):
        return None

    def _on_trials_evaluated(self, count):
        return None

    def _record_convergence(self, generation):
        record_generation(
            self.convergence_history,
            self.population,
            self.archive,
            self.config,
        )

    def _log_generation(self, generation):
        stats = self.population.get_statistics()
        coverage = stats["Coverage_avg_feasible"]
        rsum = stats["Rsum_avg_feasible"]
        coverage_text = f"{coverage:.3f}" if not np.isnan(coverage) else "NaN"
        rsum_text = f"{rsum:.1f}" if not np.isnan(rsum) else "NaN"
        self.logger.info(
            f"Gen {generation:4d} | FR={stats['FR']:.3f} "
            f"CV={stats['CV_avg']:.4f} Cov={coverage_text} Rsum={rsum_text} "
            f"Archive={len(self.archive)} Evals={self.evaluation_count}"
        )
