"""Heuristic initializers for encoded individuals."""
from src.model.individual import create_random_individual, create_greedy_energy_individual


def create_heuristic_individual(ctx):
    return create_greedy_energy_individual(ctx)


def create_random_individual_for_ctx(ctx):
    im = ctx.index_mapping
    return create_random_individual(im.num_sensor_candidates, im.num_ap_candidates)
