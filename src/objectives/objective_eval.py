from src.objectives.coverage_objective import compute_coverage
from src.objectives.rsum_capacity_objective import compute_rsum_capacity

def evaluate_objectives(solution, ctx):
    solution.coverage = compute_coverage(solution, ctx)
    solution.rsum_capacity = compute_rsum_capacity(solution, ctx)
    solution.objectives[0] = solution.coverage
    solution.objectives[1] = solution.rsum_capacity
    return solution.objectives


