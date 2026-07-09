import math
from types import SimpleNamespace

from src.rl_init.init_evaluator import quality_score, get_selected_rsum


class FakeSolution:
    def __init__(self, actual, capacity):
        self.coverage = 0.5
        self.throughput = actual
        self.cv = 0.0
        self.feasible = True
        self.repair_iter = 0
        self.metadata = {"throughput_actual": actual, "throughput_capacity": capacity}


def _cfg(metric):
    return {
        "objectives": {"throughput_metric": metric},
        "drl_init": {"normalization": {"rsum_ref_min": 0.0, "rsum_ref_max": 100.0, "cv_ref": 10.0, "repair_iter_ref": 5}},
    }


def test_quality_score_uses_actual_metric():
    sol = FakeSolution(actual=20.0, capacity=80.0)
    assert get_selected_rsum(sol, _cfg("actual")) == 20.0
    assert math.isclose(quality_score(sol, _cfg("actual")), 0.30 * 0.5 + 0.30 * 0.2 + 0.25)


def test_quality_score_uses_capacity_metric_and_distinguishes_solutions():
    low = FakeSolution(actual=20.0, capacity=30.0)
    high = FakeSolution(actual=20.0, capacity=90.0)
    assert get_selected_rsum(high, _cfg("capacity")) == 90.0
    assert quality_score(high, _cfg("capacity")) > quality_score(low, _cfg("capacity"))
