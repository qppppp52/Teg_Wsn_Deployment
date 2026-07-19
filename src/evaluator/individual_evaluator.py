from src.evaluator.repair_loop import repair_loop


def configured_max_repair_iter(ctx):
    """Return the shared repair limit from the problem configuration."""
    config = getattr(ctx, "config", None) or {}
    constraints = config.get("constraints", {}) or {}
    max_repair_iter = int(constraints.get("max_repair_iter", 5))
    if max_repair_iter < 0:
        raise ValueError("constraints.max_repair_iter must be non-negative")
    return max_repair_iter


def evaluate_individual(individual, ctx, max_repair_iter=None, repair_strategy=None):
    if max_repair_iter is None:
        max_repair_iter = configured_max_repair_iter(ctx)
    return repair_loop(
        individual,
        ctx,
        max_iter=max_repair_iter,
        repair_strategy=repair_strategy,
    )