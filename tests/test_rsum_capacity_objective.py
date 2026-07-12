from types import SimpleNamespace

import numpy as np

from src.model.solution import Solution
from src.objectives.objective_eval import evaluate_objectives
from src.rl_init.reward import compute_terminal_reward
from src.rl_init.init_evaluator import quality_score


def _ctx():
    return SimpleNamespace(
        num_candidates=2,
        coverage_matrix=np.array([[1], [0]], dtype=int),
        channel_gain_matrix=np.ones((2, 2)),
        distance_matrix=np.ones((2, 2)),
        config={"channel": {}},
    )


def test_rsum_capacity_is_primary_objective(monkeypatch):
    import src.objectives.rsum_capacity_objective as throughput_module

    monkeypatch.setattr(throughput_module, "compute_snr", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(throughput_module, "compute_rate", lambda snr, config: 100.0)
    sol = Solution(2)
    sol.x[0] = 1
    sol.y[1] = 1
    sol.c[0, 1] = 1
    evaluate_objectives(sol, _ctx())
    assert sol.rsum_capacity == 100.0
    evaluate_objectives(sol, _ctx())
    assert sol.rsum_capacity == 100.0


def test_drl_reward_and_quality_score_use_selected_metric(monkeypatch):
    import src.rl_init.reward as reward_module

    class FakeEnv:
        config = {
            "drl_init": {"normalization": {"rsum_ref_min": 0.0, "rsum_ref_max": 100.0, "cv_ref": 10.0, "repair_iter_ref": 5}},
        }
        cfg = {"max_repair_iter": 1}
        ctx = SimpleNamespace(config={"constraints": {"max_repair_iter": 1}})
        step_count = 1
        invalid_action_count = 0
        selection_order_sensors = [0]
        selection_order_aps = [1]

        def build_current_individual(self):
            return object()

        def too_few_nodes(self):
            return False

    sol = Solution(2)
    sol.coverage = 0.5
    sol.rsum_capacity = 10.0
    sol.cv = 0.0
    sol.feasible = True
    sol.metadata = {"rsum_capacity": 90.0}
    monkeypatch.setattr(reward_module, "evaluate_individual", lambda *args, **kwargs: (sol, None))
    reward, metrics = compute_terminal_reward(FakeEnv(), {"rsum": 1.0, "coverage": 0.0, "feasible_bonus": 0.0}, FakeEnv.config["drl_init"]["normalization"])
    assert metrics["rsum"] == 90.0
    assert metrics["rsum_norm"] == 0.9
    assert reward > 0.8
    assert quality_score(sol, FakeEnv.config) > quality_score(SimpleNamespace(coverage=0.5, rsum_capacity=10.0, cv=0.0, feasible=True, repair_iter=0, metadata={"rsum_capacity": 20.0}), FakeEnv.config)

