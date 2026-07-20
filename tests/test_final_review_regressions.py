from types import SimpleNamespace

import numpy as np
import pytest

from src.constraints.constraint_report import evaluate_constraints
from src.decoder.connection_decoder import assign_connections
from src.heatsink.sink_allocator import rebuild_minimum_sink_allocation
from src.heatsink.sink_ownership import build_sink_ownership
from src.io.result_io import save_pareto
from src.model.pareto_archive import ParetoArchive
from src.model.solution import Solution
from src.rl_init.checkpoint import stable_config_hash


def _heatsink_ctx(count=3):
    return SimpleNamespace(
        num_candidates=count,
        P_grid=np.full(count, 0.02, dtype=float),
        nmax=np.full(count, count, dtype=np.int32),
        neighbor_sets=[list(range(count)) for _ in range(count)],
        config={
            "heatsink": {
                "allow_sink_sink_overlap": False,
                "allow_sink_on_other_node": False,
                "allow_sink_outside_neighborhood": False,
                "allow_self_node_sink_overlap": True,
                "count_self_grid_as_sink": True,
            }
        },
    )


def test_initial_connection_rejects_power_above_channel_limit_and_ties_by_id():
    ctx = SimpleNamespace(
        num_candidates=3,
        ptx_min_matrix=np.asarray(
            [[0.0, 0.6, 0.1], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        ),
        link_feasible_matrix=np.ones((3, 3), dtype=np.int8),
        config={"channel": {"p_tx_max": 0.5}},
    )
    x = np.asarray([1, 0, 0], dtype=np.int8)
    y = np.asarray([0, 1, 1], dtype=np.int8)

    connections = assign_connections(x, y, ctx)

    assert connections[0, 1] == 0
    assert connections[0, 2] == 1

    ctx.ptx_min_matrix[0, 1] = 0.1
    connections = assign_connections(x, y, ctx)
    assert connections[0, 1] == 1
    assert connections[0, 2] == 0


def test_sensor_and_ap_with_same_global_id_are_distinct_sink_owners():
    ctx = _heatsink_ctx(2)
    ctx.neighbor_sets = [[1], [0]]
    solution = Solution(2)
    solution.x[0] = 1
    solution.y[0] = 1
    solution.n_sink_sensor[0] = 1
    solution.n_sink_ap[0] = 1

    rebuild_minimum_sink_allocation(solution, ctx)

    claims = solution.z_sink_sensor[0] + solution.z_sink_ap[0]
    assert claims == [1]
    ownership = build_sink_ownership(solution, ctx)
    assert ownership.forbidden_conflict_count == 0


def test_count_self_grid_as_sink_is_honored_by_allocation_and_ownership():
    ctx = _heatsink_ctx(2)
    ctx.config["heatsink"]["count_self_grid_as_sink"] = False
    solution = Solution(2)
    solution.x[0] = 1
    solution.n_sink_sensor[0] = 1

    rebuild_minimum_sink_allocation(solution, ctx)

    assert solution.z_sink_sensor[0] == [1]
    solution.z_sink_sensor[0] = [0]
    ownership = build_sink_ownership(solution, ctx)
    assert ownership.effective_sensor_count[0] == 0
    assert ownership.invalid_sensor_count[0] == 1


def test_feasible_solution_replaces_same_objective_infeasible_archive_entry():
    archive = ParetoArchive()
    infeasible = Solution(1)
    infeasible.coverage = 0.5
    infeasible.rsum_capacity = 100.0
    infeasible.cv = 1.0
    feasible = infeasible.copy()
    feasible.feasible = True
    feasible.cv = 0.0

    archive.update([infeasible, feasible])

    assert len(archive.solutions) == 1
    assert archive.solutions[0].feasible


def test_result_export_does_not_mutate_harvest_cache(tmp_path):
    ctx = _heatsink_ctx(2)
    solution = Solution(2)
    solution.x[0] = 1
    solution.z_sink_sensor[0] = [0, 1]
    solution.sensor_harvest_power[:] = -7.0
    solution.feasible = True
    before = solution.sensor_harvest_power.copy()
    path = tmp_path / "pareto.npz"

    save_pareto([solution], str(path), ctx)

    np.testing.assert_array_equal(solution.sensor_harvest_power, before)
    with np.load(path) as data:
        assert data["sensor_harvest_power"][0, 0] == pytest.approx(0.04)


def test_energy_check_rejects_nonpositive_harvest_power():
    ctx = _heatsink_ctx(1)
    ctx.P_grid[0] = 0.0
    solution = Solution(1)
    solution.x[0] = 1

    report = evaluate_constraints(solution, ctx)
    assert report.energy_cv.sensor > 0.0
    assert report.physics.sink_requirement.unachievable_sensor == (True,)


def test_ppo_checkpoint_hash_covers_physical_channel_semantics():
    base = {
        "experiment": {"name": "contract_test"},
        "channel": {"p_tx_max": 0.2},
        "drl_init": {
            "role_actions": ["sensor", "ap", "skip", "stop"],
            "reward": {},
            "normalization": {},
            "network": {},
        },
    }
    changed = {
        **base,
        "channel": {"p_tx_max": 0.5},
    }

    assert stable_config_hash(base) != stable_config_hash(changed)
