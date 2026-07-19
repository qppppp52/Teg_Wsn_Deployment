"""Initial link-power decoding."""
from src.constraints.link_constraints import single_valid_connected_ap


INITIAL_POWER_SEMANTICS_VERSION = 1


def decode_power(solution, ctx):
    """Initialize every sole valid connection exactly at its minimum power."""
    solution.p_tx.fill(0.0)
    for sensor_id in range(ctx.num_candidates):
        ap_id = single_valid_connected_ap(solution, sensor_id, ctx)
        if ap_id is not None:
            solution.p_tx[sensor_id, ap_id] = ctx.ptx_min_matrix[sensor_id, ap_id]
