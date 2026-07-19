from types import SimpleNamespace

import numpy as np

from src.io.result_io import RESULT_SCHEMA_VERSION, save_pareto
from src.model.solution import Solution


def test_pareto_npz_preserves_owner_grid_tensor_and_int32_demand(tmp_path):
    ctx = SimpleNamespace(
        num_candidates=3,
        P_grid=np.asarray([0.02, 0.03, 0.04]),
        nmax=np.asarray([2, 2, 2], dtype=np.int32),
        neighbor_sets=[[0, 2], [1], [2]],
        config={
            "heatsink": {
                "allow_sink_sink_overlap": False,
                "allow_sink_on_other_node": False,
                "allow_sink_outside_neighborhood": False,
                "allow_self_node_sink_overlap": True,
            }
        },
    )
    solution = Solution(3)
    solution.x[0] = 1
    solution.y[1] = 1
    solution.n_sink_sensor[0] = 2
    solution.n_sink_ap[1] = 1
    solution.z_sink_sensor[0] = [0, 2]
    solution.z_sink_ap[1] = [1]
    solution.feasible = True
    solution.coverage = 0.5
    solution.rsum_capacity = 123.0
    solution.cv = 0.0
    output = tmp_path / "pareto.npz"

    save_pareto([solution], str(output), ctx)

    with np.load(output) as data:
        assert int(data["result_schema_version"]) == RESULT_SCHEMA_VERSION
        assert data["n_sink_sensor"].dtype == np.int32
        assert data["z_sink_sensor_by_owner"].shape == (1, 3, 3)
        assert data["z_sink_ap_by_owner"].shape == (1, 3, 3)
        assert data["z_sink_sensor_by_owner"][0, 0, 2]
        assert data["z_sink_ap_by_owner"][0, 1, 1]
        np.testing.assert_array_equal(data["n_effective_sensor"][0], [2, 0, 0])
