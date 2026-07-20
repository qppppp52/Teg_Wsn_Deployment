"""Shared one-generation execution for CR-MODE variants."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.evaluator.individual_evaluator import evaluate_individual, configured_max_repair_iter
from src.model.population import Population
from src.operators.mode_crossover import crossover
from src.operators.mode_mutation import mutate, select_pareto_guide
from src.evaluation.constrained_dominance import constrained_dominates as report_constrained_dominates


@dataclass(frozen=True)
class GenerationResult:
    population: Population
    trial_solutions: list
    evaluations: int


class GenerationExecutor:
    """Generate NP trials and apply constrained Pareto environmental selection."""

    def __init__(self, ctx, population_size: int):
        self.ctx = ctx
        self.population_size = int(population_size)
        self.max_repair_iter = configured_max_repair_iter(ctx)

    def execute(self, population, action: dict) -> GenerationResult:
        parents = list(population.individuals)
        parent_solutions = list(population.solutions)
        guide = select_pareto_guide(parents)
        trial_individuals = []
        trial_solutions = []

        for index, parent in enumerate(parents):
            mutant = mutate(
                parent,
                float(action["F"]),
                action["mutation_strategy"],
                parents,
                index,
                self.ctx,
                guide=guide,
            )
            trial = crossover(parent, mutant, float(action["CR"]))
            solution, repaired = evaluate_individual(
                trial,
                self.ctx,
                max_repair_iter=self.max_repair_iter,
                repair_strategy=action.get("repair_strategy", action),
            )
            if repaired is not None:
                trial = repaired
            copy_solution_metrics(trial, solution)
            trial_individuals.append(trial)
            trial_solutions.append(solution)

        selected_individuals, selected_solutions = environmental_select(
            parents + trial_individuals,
            parent_solutions + trial_solutions,
            self.population_size,
        )
        next_population = Population(
            self.population_size,
            population.num_s,
            population.num_a,
        )
        next_population.individuals = selected_individuals
        next_population.solutions = selected_solutions
        return GenerationResult(next_population, trial_solutions, len(trial_solutions))


def environmental_select(individuals, solutions, size: int):
    """NSGA-II style selection using Deb's constrained dominance rule."""
    fronts = _fast_non_dominated_sort(solutions)
    selected = []
    for front in fronts:
        remaining = int(size) - len(selected)
        if remaining <= 0:
            break
        if len(front) <= remaining:
            selected.extend(front)
            continue
        distances = _crowding_distance([solutions[index] for index in front])
        order = np.argsort(distances, kind="stable")[::-1]
        selected.extend(front[index] for index in order[:remaining])
        break
    return [individuals[index] for index in selected], [solutions[index] for index in selected]


def constrained_dominates(a, b) -> bool:
    return report_constrained_dominates(a, b)


def _fast_non_dominated_sort(solutions):
    dominates = [[] for _ in solutions]
    dominated_count = [0 for _ in solutions]
    first = []
    for i, candidate in enumerate(solutions):
        for j, other in enumerate(solutions):
            if i == j:
                continue
            if constrained_dominates(candidate, other):
                dominates[i].append(j)
            elif constrained_dominates(other, candidate):
                dominated_count[i] += 1
        if dominated_count[i] == 0:
            first.append(i)

    fronts = [first] if first else []
    while fronts and fronts[-1]:
        following = []
        for i in fronts[-1]:
            for j in dominates[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    following.append(j)
        if following:
            fronts.append(following)
        else:
            break
    return fronts


def _crowding_distance(solutions):
    count = len(solutions)
    if count <= 2:
        return np.full(count, np.inf)
    if all(solution.feasible for solution in solutions):
        values = np.asarray(
            [[solution.coverage, solution.rsum_capacity] for solution in solutions],
            dtype=float,
        )
    else:
        values = np.asarray(
            [[-float(solution.cv), solution.coverage, solution.rsum_capacity] for solution in solutions],
            dtype=float,
        )
    distances = np.zeros(count, dtype=float)
    for column in range(values.shape[1]):
        order = np.argsort(values[:, column], kind="stable")
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf
        span = values[order[-1], column] - values[order[0], column]
        if abs(span) <= 1.0e-12:
            continue
        for pos in range(1, count - 1):
            distances[order[pos]] += (
                values[order[pos + 1], column] - values[order[pos - 1], column]
            ) / span
    return distances


def copy_solution_metrics(individual, solution):
    individual.coverage = solution.coverage
    individual.rsum_capacity = solution.rsum_capacity
    individual.cv = solution.cv
    individual.feasible = solution.feasible
