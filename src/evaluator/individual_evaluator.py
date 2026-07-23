from src.constraints.repair_config import resolve_repair_config
from src.evaluator.repair_loop import (
    evaluate_decoded_solution_without_repair,
    repair_loop,
)


def configured_max_repair_iter(ctx):
    """Compatibility wrapper for the outer repair-round budget."""
    return resolve_repair_config(getattr(ctx, "config", None)).max_outer_repair_rounds


def evaluate_individual(individual, ctx, max_repair_iter=None, repair_strategy=None):
    if max_repair_iter is None:
        max_repair_iter = configured_max_repair_iter(ctx)
    return repair_loop(
        individual,
        ctx,
        max_iter=max_repair_iter,
        repair_strategy=repair_strategy,
    )


def evaluate_individual_without_repair(individual, ctx, max_repair_iter=None, repair_strategy=None):
    """Evaluate MODE's common decoder output without constraint repair.

    The extra parameters intentionally mirror :func:`evaluate_individual`, so
    a generation executor can switch evaluators without changing its contract.
    """
    del max_repair_iter, repair_strategy
    return evaluate_decoded_solution_without_repair(individual, ctx)