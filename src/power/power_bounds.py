"""Energy-safe transmit-power bounds from effective heatsink ownership."""
from __future__ import annotations

from src.heatsink.sink_ownership import build_sink_ownership
from src.physics.numerical_tolerances import ENERGY_ABS_TOL


def max_energy_feasible_ptx(
    solution,
    sensor_id,
    ap_id,
    ctx,
    *,
    extra_effective_sinks=0,
    ownership=None,
):
    """Return the energy-safe upper bound clipped to the channel power range."""
    del ap_id
    sensor_id = int(sensor_id)
    ownership = ownership or build_sink_ownership(solution, ctx)
    effective_count = len(ownership.effective_sensor_positions[sensor_id])
    effective_count += max(0, int(extra_effective_sinks))
    harvest = effective_count * float(ctx.P_grid[sensor_id])
    sensor_cfg = ctx.config.get("sensor", {})
    fixed_consumption = float(sensor_cfg.get("P_sens", 0.01)) + float(
        sensor_cfg.get("P_proc", 0.005)
    )
    ptx_max = float(ctx.config.get("channel", {}).get("p_tx_max", 0.5))
    energy_upper = harvest - fixed_consumption - ENERGY_ABS_TOL
    return min(ptx_max, max(0.0, energy_upper))


def calculate_actual_ptx_upper_bound(solution, sensor_idx, ap_idx, ctx):
    """Compatibility name for the shared energy-safe bound."""
    return max_energy_feasible_ptx(solution, sensor_idx, ap_idx, ctx)
