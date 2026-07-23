from types import SimpleNamespace

import numpy as np
import pytest

from src.constraints.energy_constraints import (
    check_energy_constraints,
    repair_energy_constraints,
)
from src.constraints.constraint_report import evaluate_constraints, sink_diagnostic_metrics
from src.constraints.heatsink_constraints import check_heatsink_constraints
from src.heatsink.harvest_power import (
    actual_sink_count,
    ap_harvest_power,
    sensor_harvest_power,
)
from src.heatsink.sink_allocator import allocate_all_sinks
from src.heatsink.sink_requirement import compute_sink_requirements
from src.model.solution import Solution
from src.power.power_repair import repair_power


def _ctx(p_grid=(0.02, 0.03, 0.04, 0.05)):
    count = len(p_grid)
    return SimpleNamespace(
        num_candidates=count,
        P_grid=np.asarray(p_grid, dtype=float),
        nmax=np.full(count, count, dtype=int),
        neighbor_sets=[list(range(count)) for _ in range(count)],
        ptx_min_matrix=np.full((count, count), 0.01, dtype=float),
        link_feasible_matrix=np.ones((count, count), dtype=np.int8),
        config={
            "sensor": {"P_sens": 0.01, "P_proc": 0.005},
            "ap": {
                "P_idle": 0.01,
                "P_proc": 0.005,
                "P_rx": 0.003,
            },
            "channel": {"p_tx_max": 0.5},
            "constraints": {"max_outer_repair_rounds": 5, "max_energy_stabilization_iters": 5},
            "heatsink": {
                "allow_sink_sink_overlap": False,
                "allow_sink_on_other_node": False,
                "allow_sink_outside_neighborhood": False,
                "allow_self_node_sink_overlap": True,
            },
        },
    )


def test_actual_harvest_uses_unique_legal_owner_positions():
    ctx = _ctx()
    solution = Solution(ctx.num_candidates)
    solution.x[0] = 1
    solution.y[2] = 1
    solution.z_sink_sensor[0] = [0, 1, 1]
    solution.z_sink_ap[2] = [2, 3, 3]

    assert actual_sink_count(solution.z_sink_sensor[0]) == 2
    assert sensor_harvest_power(solution, 0, ctx) == pytest.approx(0.04)
    assert ap_harvest_power(solution, 2, ctx) == pytest.approx(0.08)


def test_forbidden_conflict_harvests_for_neither_owner():
    ctx = _ctx()
    solution = Solution(ctx.num_candidates)
    solution.x[0] = 1
    solution.x[1] = 1
    solution.n_sink_sensor[:2] = 1
    solution.z_sink_sensor[0] = [2]
    solution.z_sink_sensor[1] = [2]
    solution.sensor_power_consumption[:2] = 0.01

    assert sensor_harvest_power(solution, 0, ctx) == 0.0
    assert sensor_harvest_power(solution, 1, ctx) == 0.0
    assert check_heatsink_constraints(solution, ctx) == pytest.approx(1.0 / 12.0)


def test_matching_requested_and_actual_sinks_have_zero_energy_and_sink_cv():
    ctx = _ctx(p_grid=(0.02, 0.03, 0.04, 0.05, 0.06))
    solution = Solution(ctx.num_candidates)
    solution.x[0] = 1
    solution.n_sink_sensor[0] = 5
    solution.z_sink_sensor[0] = [0, 1, 2, 3, 4]
    solution.sensor_power_consumption[0] = 0.10

    assert check_energy_constraints(solution, ctx) == pytest.approx(0.0)
    assert check_heatsink_constraints(solution, ctx) == pytest.approx(0.0)


def test_constraint_checks_do_not_mutate_physical_state():
    ctx = _ctx()
    solution = Solution(ctx.num_candidates)
    solution.x[0] = 1
    solution.n_sink_sensor[0] = 2
    solution.z_sink_sensor[0] = [0]
    solution.sensor_power_consumption[0] = 0.04
    before = solution.copy()

    check_energy_constraints(solution, ctx)
    check_heatsink_constraints(solution, ctx)

    np.testing.assert_array_equal(solution.x, before.x)
    np.testing.assert_array_equal(solution.p_tx, before.p_tx)
    np.testing.assert_array_equal(solution.sensor_harvest_power, before.sensor_harvest_power)
    assert solution.z_sink_sensor == before.z_sink_sensor


def test_duplicate_sink_grid_is_not_extra_harvest_and_is_a_sink_violation():
    ctx = _ctx()
    solution = Solution(ctx.num_candidates)
    solution.x[0] = 1
    solution.n_sink_sensor[0] = 2
    solution.z_sink_sensor[0] = [0, 1, 1]
    solution.sensor_power_consumption[0] = 0.04

    assert sensor_harvest_power(solution, 0, ctx) == pytest.approx(0.04)
    assert check_heatsink_constraints(solution, ctx) == pytest.approx(1.0 / 18.0)
    metrics = sink_diagnostic_metrics(evaluate_constraints(solution, ctx))
    assert metrics["sink_duplicate_count"] == 1
    assert metrics["sink_duplicate_cv"] == pytest.approx(1.0 / 3.0)
    assert metrics["sink_dominant_violation"] == "duplicate"


def test_actual_shortfall_affects_energy_and_heatsink_constraints():
    ctx = _ctx()
    solution = Solution(ctx.num_candidates)
    solution.x[0] = 1
    solution.y[3] = 1
    solution.c[0, 3] = 1
    solution.p_tx[0, 3] = 0.2
    solution.n_sink_sensor[0] = 5
    solution.z_sink_sensor[0] = [0, 1, 2]

    report = evaluate_constraints(solution, ctx)
    assert report.energy_cv.sensor > 0.0
    assert report.physics.energy.sensor_harvest[0] == pytest.approx(0.06)
    assert report.raw.sink_diagnostics.shortage_sensor > 0


def test_sink_requirement_is_not_truncated_by_static_nmax():
    ctx = _ctx(p_grid=(0.01, 0.03, 0.04, 0.05))
    ctx.nmax[0] = 1
    solution = Solution(ctx.num_candidates)
    solution.x[0] = 1

    compute_sink_requirements(solution, ctx)

    assert solution.n_sink_sensor[0] == 2
    assert solution.n_sink_sensor.dtype == np.int32


def test_power_repair_retains_supported_current_power_without_active_increase():
    ctx = _ctx()
    solution = Solution(ctx.num_candidates)
    solution.x[0] = 1
    solution.y[3] = 1
    solution.c[0, 3] = 1
    solution.p_tx[0, 3] = 0.04
    solution.n_sink_sensor[0] = 3
    solution.z_sink_sensor[0] = [0, 1, 2]

    repair_power(solution, ctx)

    assert solution.sensor_harvest_power[0] == pytest.approx(0.06)
    assert solution.p_tx[0, 3] == pytest.approx(0.04)


def test_power_repair_uses_minimum_when_actual_harvest_only_supports_it():
    ctx = _ctx(p_grid=(0.025 / 3.0, 0.03, 0.04, 0.05))
    solution = Solution(ctx.num_candidates)
    solution.x[0] = 1
    solution.y[3] = 1
    solution.c[0, 3] = 1
    solution.p_tx[0, 3] = 0.04
    solution.n_sink_sensor[0] = 5
    solution.z_sink_sensor[0] = [0, 1, 2]

    repair_power(solution, ctx)

    assert solution.p_tx[0, 3] == pytest.approx(ctx.ptx_min_matrix[0, 3])


def test_energy_repair_keeps_sensor_when_sinks_cannot_support_minimum_power():
    ctx = _ctx(p_grid=(0.01, 0.20, 0.04, 0.05))
    ctx.nmax[0] = 1
    ctx.neighbor_sets[0] = [0]
    ctx.neighbor_sets[1] = [1]
    solution = Solution(ctx.num_candidates)
    solution.x[0] = 1
    solution.y[1] = 1
    solution.c[0, 1] = 1
    solution.p_tx[0, 1] = 0.01

    repair_energy_constraints(solution, ctx)

    assert solution.x[0] == 1
    assert solution.metadata["energy_repair_variant"] == "baseline_no_energy_node_deletion"


def test_energy_repair_deletion_requires_explicit_ablation_flag():
    ctx = _ctx(p_grid=(0.01, 0.20, 0.04, 0.05))
    ctx.config["constraints"]["enable_energy_node_deletion_fallback"] = True
    ctx.nmax[0] = 1
    ctx.neighbor_sets[0] = [0]
    ctx.neighbor_sets[1] = [1]
    solution = Solution(ctx.num_candidates)
    solution.x[0] = 1
    solution.y[1] = 1
    solution.c[0, 1] = 1
    solution.p_tx[0, 1] = 0.01

    repair_energy_constraints(solution, ctx)

    assert solution.x[0] == 0
    assert not np.any(solution.c[0])
    assert solution.z_sink_sensor[0] == []
    assert solution.metadata["energy_repair_variant"] == "aggressive_energy_node_deletion"


def test_sensor_and_ap_competing_for_grid_use_actual_allocations():
    ctx = _ctx()
    ctx.neighbor_sets[0] = [2]
    ctx.neighbor_sets[1] = [2]
    solution = Solution(ctx.num_candidates)
    solution.x[0] = 1
    solution.y[1] = 1
    solution.n_sink_sensor[0] = 1
    solution.n_sink_ap[1] = 1
    solution.sensor_power_consumption[0] = 0.02
    solution.ap_power_consumption[1] = 0.04

    allocate_all_sinks(solution, ctx)

    assert solution.z_sink_sensor[0] == [2]
    assert solution.z_sink_ap[1] == []
    assert sensor_harvest_power(solution, 0, ctx) == pytest.approx(0.02)
    assert ap_harvest_power(solution, 1, ctx) == 0.0
    assert check_energy_constraints(solution, ctx) > 0.0


def test_allocator_clears_stale_occupancy_and_is_deterministic():
    ctx = _ctx()
    ctx.neighbor_sets[0] = [0, 0, 1, 1]
    solution = Solution(ctx.num_candidates)
    solution.x[0] = 1
    solution.n_sink_sensor[0] = 2
    solution.z_sink_sensor[1] = [3]

    allocate_all_sinks(solution, ctx)
    first = [list(positions) for positions in solution.z_sink_sensor]
    allocate_all_sinks(solution, ctx)

    assert solution.z_sink_sensor[1] == []
    assert solution.z_sink_sensor[0] == [0, 1]
    assert solution.z_sink_sensor == first
