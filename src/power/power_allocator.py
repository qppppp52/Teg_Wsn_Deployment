from src.power.power_bounds import calculate_actual_ptx_upper_bound
from src.power.adaptive_power import adaptive_power

def allocate_power(sensor_idx, ap_idx, solution, ctx):
    ptx_min = ctx.ptx_min_matrix[sensor_idx, ap_idx]
    ptx_up = calculate_actual_ptx_upper_bound(solution, sensor_idx, ap_idx, ctx)
    strategy = ctx.config.get("mode", {}).get("power_strategy", "conservative")
    return adaptive_power(sensor_idx, ap_idx, ptx_min, ptx_up, strategy)
