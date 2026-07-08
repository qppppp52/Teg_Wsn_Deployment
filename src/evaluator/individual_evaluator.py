from src.evaluator.repair_loop import repair_loop

def evaluate_individual(individual, ctx, max_repair_iter=5):
    return repair_loop(individual, ctx, max_iter=max_repair_iter)
