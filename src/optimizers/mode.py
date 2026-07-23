"""Plain multi-objective differential evolution baseline.

MODE shares CR-MODE's encoding, decoding, objectives, and constrained
environmental selection. Its defining difference is that it evaluates each
decoded individual directly and never invokes the constraint repair loop.
"""
from __future__ import annotations

from src.evaluator.individual_evaluator import evaluate_individual_without_repair
from src.optimizers.cr_mode import CRMode
from src.optimizers.generation_executor import GenerationExecutor, copy_solution_metrics
from src.utils.logger import get_logger


class MODE(CRMode):
    """MODE baseline without CR-MODE's iterative constraint repair."""

    def __init__(self, ctx, config):
        super().__init__(ctx, config)
        self.executor = GenerationExecutor(
            ctx,
            self.NP,
            evaluator=evaluate_individual_without_repair,
        )
        self.max_repair_iter = 0
        self.logger = get_logger("MODE")

    def evaluate_population(self):
        solutions = []
        for individual in self.population.individuals:
            solution, _ = evaluate_individual_without_repair(individual, self.ctx)
            copy_solution_metrics(individual, solution)
            solutions.append(solution)
            self.evaluation_count += 1
        self.population.solutions = solutions
        self._record_boost_statistics(solutions)