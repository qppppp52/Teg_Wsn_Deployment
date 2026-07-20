from types import SimpleNamespace

import numpy as np

from src.constraints.constraint_eval import evaluate_all_constraints
from src.heatsink.harvest_power import refresh_harvest_diagnostics
from src.heatsink.sink_allocator import rebuild_minimum_sink_allocation
from src.heatsink.sink_ownership import build_sink_ownership
from src.heatsink.sink_requirement import compute_sink_requirements
from src.model.solution import Solution
from src.objectives.objective_eval import evaluate_objectives
from src.power.throughput_enhancer import enhance_rsum_capacity_greedily


def _ctx():
    count = 4
    config = {
        "deployment": {"max_sensors": 2, "max_aps": 2},
        "sensor": {"P_sens": 0.01, "P_proc": 0.005},
        "ap": {
            "P_idle": 0.01,
            "P_proc": 0.005,
            "P_rx": 0.003,
        },
        "channel": {
            "p_tx_max": 0.2,
            "noise_power_density": 1.0e-12,
            "bandwidth": 1.0e6,
            "path_loss_alpha": 2.0,
        },
        "heatsink": {
            "allow_sink_sink_overlap": False,
            "allow_sink_on_other_node": False,
            "allow_sink_outside_neighborhood": False,
            "allow_self_node_sink_overlap": True,
        },
        "constraints": {
            "feasible_tol": 1.0e-8,
            "max_repair_iter": 5,
            "cv_weights": {
                "deploy": 1.0,
                "link": 1.0,
                "power": 1.0,
                "energy": 1.0,
                "sink": 1.0,
                "service": 1.0,
            },
        },
        "throughput_enhancement": {"enabled": True, "max_boost_steps": 4},
    }
    return SimpleNamespace(
        num_candidates=count,
        P_grid=np.asarray([0.05, 0.03, 0.03, 0.03]),
        nmax=np.full(count, count, dtype=np.int32),
        neighbor_sets=[[0, 2, 3], [1], [2], [3]],
        ptx_min_matrix=np.full((count, count), 0.01),
        link_feasible_matrix=np.ones((count, count), dtype=np.int8),
        channel_gain_matrix=np.ones((count, count)),
        distance_matrix=np.ones((count, count)),
        coverage_matrix=np.asarray(
            [[1, 1], [0, 0], [0, 0], [0, 0]], dtype=np.int8
        ),
        config=config,
    )


def _feasible_solution(ctx):
    solution = Solution(ctx.num_candidates)
    solution.x[0] = 1
    solution.y[1] = 1
    solution.c[0, 1] = 1
    solution.p_tx[0, 1] = ctx.ptx_min_matrix[0, 1]
    compute_sink_requirements(solution, ctx)
    rebuild_minimum_sink_allocation(solution, ctx)
    refresh_harvest_diagnostics(solution, ctx)
    evaluate_all_constraints(solution, ctx)
    evaluate_objectives(solution, ctx)
    assert solution.feasible
    return solution


def test_capacity_enhancement_is_feasible_monotonic_and_idempotent():
    ctx = _ctx()
    before = _feasible_solution(ctx)
    before_ownership = build_sink_ownership(before, ctx)
    enhanced = enhance_rsum_capacity_greedily(before, ctx)

    assert enhanced.feasible
    np.testing.assert_array_equal(enhanced.x, before.x)
    np.testing.assert_array_equal(enhanced.y, before.y)
    np.testing.assert_array_equal(enhanced.c, before.c)
    assert enhanced.coverage == before.coverage
    assert enhanced.rsum_capacity > before.rsum_capacity
    assert enhanced.metadata["boost_sinks_added"] > 0
    assert (
        build_sink_ownership(enhanced, ctx).effective_ap_count
        == before_ownership.effective_ap_count
    )

    enhanced_again = enhance_rsum_capacity_greedily(enhanced, ctx)
    np.testing.assert_array_equal(enhanced_again.p_tx, enhanced.p_tx)
    assert enhanced_again.z_sink_sensor == enhanced.z_sink_sensor
    assert enhanced_again.rsum_capacity == enhanced.rsum_capacity


def test_infeasible_solution_is_not_enhanced():
    ctx = _ctx()
    solution = Solution(ctx.num_candidates)
    solution.coverage = 0.25
    solution.rsum_capacity = 10.0
    solution.feasible = False

    result = enhance_rsum_capacity_greedily(solution, ctx)

    np.testing.assert_array_equal(result.x, solution.x)
    np.testing.assert_array_equal(result.p_tx, solution.p_tx)
    assert result.rsum_capacity == 10.0
    assert result.metadata["boost_stop_reason"] == "infeasible"
