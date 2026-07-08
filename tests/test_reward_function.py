from src.dqn.reward_function import compute_reward
from src.rl_init.reward import compute_step_reward


def test_dqn_reward_penalizes_repair_cost_when_objectives_stall():
    before = {
        "CV_mean": 0.0,
        "FR": 1.0,
        "HV": 0.5,
        "best_coverage": 0.8,
        "best_rsum_norm": 0.8,
        "diversity": 0.2,
        "mean_repair_iter": 0.0,
        "pressure": {"deploy": 0.0, "link": 0.0, "capacity": 0.0, "energy": 0.0, "sink": 0.0, "service": 0.0},
    }
    after = dict(before)
    after["mean_repair_iter"] = 2.0
    reward, parts = compute_reward(before, after, max_repair_iter=5)
    assert parts["R_cost"] > 0
    assert reward < 0


def test_rl_init_step_reward_is_monotonic_for_coverage_gain():
    reward = compute_step_reward(
        {"estimated_coverage": 0.1, "estimated_energy": 0.1, "estimated_link": 0.1, "estimated_sink_pressure": 0.1},
        {"estimated_coverage": 0.5, "estimated_energy": 0.1, "estimated_link": 0.1, "estimated_sink_pressure": 0.1},
        {"invalid_action": False},
        {"coverage_step": 1.0},
    )
    assert reward > 0
