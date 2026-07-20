import math
from types import SimpleNamespace

from src.rl_init.init_evaluator import quality_score, get_selected_rsum_capacity


class FakeSolution:
    def __init__(self, capacity):
        self.coverage = 0.5
        self.rsum_capacity = capacity
        self.cv = 0.0
        self.feasible = True
        self.repair_iter = 0
        self.metadata = {"rsum_capacity": capacity}


def _cfg():
    return {
        "drl_init": {"normalization": {"rsum_ref_min": 0.0, "rsum_ref_max": 100.0, "cv_ref": 10.0, "repair_iter_ref": 5}},
    }


def test_quality_score_uses_rsum_capacity_metric():
    sol = FakeSolution(capacity=80.0)
    assert get_selected_rsum_capacity(sol) == 80.0
    assert math.isclose(quality_score(sol, _cfg()), 0.30 * 0.5 + 0.30 * 0.8 + 0.25)


def test_quality_score_uses_capacity_metric_and_distinguishes_solutions():
    low = FakeSolution(capacity=30.0)
    high = FakeSolution(capacity=90.0)
    assert get_selected_rsum_capacity(high) == 90.0
    assert quality_score(high, _cfg()) > quality_score(low, _cfg())
