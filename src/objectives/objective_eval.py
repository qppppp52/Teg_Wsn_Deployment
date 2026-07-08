from src.objectives.coverage_objective import compute_coverage
from src.objectives.throughput_objective import compute_throughput

def evaluate_objectives(solution, ctx):
    solution.coverage = compute_coverage(solution, ctx)
    solution.throughput = compute_throughput(solution, ctx)
    solution.objectives[0] = solution.coverage
    solution.objectives[1] = solution.throughput
    return solution.objectives
