from types import SimpleNamespace

import numpy as np
import pytest

from src.constraints.constraint_report import _power_cv, global_ptx_max
from src.model.problem_context import ProblemContext


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_problem_context_rejects_invalid_global_ptx_max(value):
    with pytest.raises(ValueError, match="finite and strictly positive"):
        ProblemContext({"channel": {"p_tx_max": value}})


def test_problem_context_requires_global_ptx_max():
    with pytest.raises(ValueError, match="channel.p_tx_max is required"):
        ProblemContext({"channel": {}})


def test_problem_context_accepts_positive_global_ptx_max():
    ctx = ProblemContext({"channel": {"p_tx_max": 0.5}})

    assert ctx.p_tx_max == pytest.approx(0.5)
    assert global_ptx_max(ctx) == pytest.approx(0.5)


def _power_case(pmin, ptx, *, global_max=0.5, per_edge_value=0.25):
    solution = SimpleNamespace(
        x=np.array([1, 0], dtype=np.int8),
        y=np.array([0, 1], dtype=np.int8),
        c=np.array([[0, 1], [0, 0]], dtype=np.int8),
        p_tx=np.array([[0.0, ptx], [0.0, 0.0]], dtype=float),
    )
    ctx = SimpleNamespace(
        num_candidates=2,
        p_tx_max=global_max,
        ptx_min_matrix=np.array([[0.0, pmin], [0.0, 0.0]], dtype=float),
        link_feasible_matrix=np.ones((2, 2), dtype=np.int8),
        optimistic_ptx_up_matrix=np.full((2, 2), per_edge_value, dtype=float),
    )
    spec = SimpleNamespace(
        ptx_max=global_max,
        power_abs_tol=0.0,
        power_rel_tol=0.0,
    )
    raw = SimpleNamespace(power_state_invalid_sensor_count=0)
    return solution, ctx, spec, raw


def test_cvpower_uses_one_global_upper_bound_and_denominator():
    solution, ctx, spec, raw = _power_case(0.1, 0.6)

    # p_bound=(0.6-0.5)/0.5=0.2, then CVpower=(p_bound+p_state)/2.
    assert _power_cv(solution, ctx, spec, raw, physical_edges=None) == pytest.approx(0.1)


def test_physically_unreachable_edge_does_not_enter_cvpower_bound():
    solution, ctx, spec, raw = _power_case(0.6, 0.6)

    assert _power_cv(solution, ctx, spec, raw, physical_edges=None) == 0.0
