from types import SimpleNamespace

import numpy as np

from src.constraints.constraint_report import _connection_degree_mismatch


def _case():
    solution = SimpleNamespace(
        x=np.array([1, 0, 0], dtype=np.int8),
        y=np.array([0, 1, 1], dtype=np.int8),
        c=np.zeros((3, 3), dtype=np.int8),
    )
    ctx = SimpleNamespace(
        ptx_min_matrix=np.full((3, 3), 0.01, dtype=float),
        link_feasible_matrix=np.ones((3, 3), dtype=np.int8),
    )
    spec = SimpleNamespace(power_abs_tol=1.0e-10, ptx_max=0.1)
    return solution, ctx, spec


def test_link_degree_penalizes_sensor_without_declared_connection():
    solution, ctx, spec = _case()

    assert _connection_degree_mismatch(solution, ctx, spec) == 1


def test_link_degree_does_not_double_penalize_invalid_declared_edge():
    solution, ctx, spec = _case()
    solution.c[0, 1] = 1
    solution.c[0, 2] = 1
    ctx.link_feasible_matrix[0, 2] = 0

    assert _connection_degree_mismatch(solution, ctx, spec) == 0


def test_link_degree_counts_only_excess_physical_connections():
    solution, ctx, spec = _case()
    solution.c[0, 1] = 1
    solution.c[0, 2] = 1

    assert _connection_degree_mismatch(solution, ctx, spec) == 1
