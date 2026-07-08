import numpy as np
from src.optimizers.base_optimizer import BaseOptimizer
from src.model.population import Population
from src.model.pareto_archive import ParetoArchive
from src.operators.strategy_controller import StrategyController
from src.operators.mode_mutation import mutate
from src.operators.mode_crossover import crossover
from src.operators.mode_selection import select_better
from src.evaluator.individual_evaluator import evaluate_individual
from src.utils.logger import get_logger
from src.evaluation.generation_diagnostics import make_convergence_history, record_generation

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


    def run(self):
        im = self.ctx.index_mapping
        self.population = Population(self.NP, im.num_sensor_candidates, im.num_ap_candidates)
        self.population.initialize(self.ctx, strategy="mixed")
        solutions = []
        for ind in self.population.individuals:
            sol, rep_ind = evaluate_individual(ind, self.ctx)
            if rep_ind is not None:
                ind = rep_ind
                self.population.individuals[len(solutions)] = ind
            _copy_solution_metrics(ind, sol)
            solutions.append(sol)
        self.population.solutions = solutions
        self.archive.update(solutions)
        self._record_convergence(0)
        for gen in range(self.Tmax):
            new_inds, new_sols = [], []
            for i, ind in enumerate(self.population.individuals):
                mutant = mutate(ind, self.strategy.F,
                                self.strategy.mutation_strategy,
                                self.population.individuals, i, self.ctx)
                trial = crossover(ind, mutant, self.strategy.crossover_rate)
                trial_sol, rep_ind = evaluate_individual(trial, self.ctx)
                if rep_ind is not None:
                    trial = rep_ind
                    trial_sol, _ = evaluate_individual(trial, self.ctx)
                winner, winner_sol, _ = select_better(
                    trial, trial_sol, ind, self.population.solutions[i])
                _copy_solution_metrics(winner, winner_sol)
                new_inds.append(winner)
                new_sols.append(winner_sol)
            self.population.individuals = new_inds
            self.population.solutions = new_sols
            self.archive.update(new_sols)
            self._record_convergence(gen + 1)
            if gen % 10 == 0 or gen == self.Tmax-1:
                stats = self.population.get_statistics()
                cov_str = f"{stats['Coverage_avg_feasible']:.3f}" if not (isinstance(stats['Coverage_avg_feasible'], float) and np.isnan(stats['Coverage_avg_feasible'])) else "NaN"
                rsum_str = f"{stats['Rsum_avg_feasible']:.1f}" if not (isinstance(stats['Rsum_avg_feasible'], float) and np.isnan(stats['Rsum_avg_feasible'])) else "NaN"
                logger.info(
                    f"Gen {gen:4d} | FR={stats['FR']:.3f} "
                    f"CV={stats['CV_avg']:.4f} "
                    f"Cov={cov_str} "
                    f"Rsum={rsum_str} "
                    f"Archive={len(self.archive)}")
        return self.archive

    def _record_convergence(self, gen):
        metrics = record_generation(self.convergence_history, self.population, self.archive, self.config)
        if metrics.get("saturated_link_ratio", 0.0) > 0.9:
            logger.warning("Throughput actual is saturated; Pareto front may degenerate.")

def _copy_solution_metrics(individual, solution):
    individual.coverage = solution.coverage
    individual.throughput = solution.throughput
    individual.cv = solution.cv
    individual.feasible = solution.feasible
