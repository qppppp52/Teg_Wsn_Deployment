from src.rl_init.reward import compute_step_reward


def test_rl_init_step_reward_rewards_estimated_improvement():
    before = {"estimated_coverage": 0.1, "estimated_energy": 0.2, "estimated_link": 0.1, "estimated_sink_pressure": 0.1}
    after = {"estimated_coverage": 0.4, "estimated_energy": 0.3, "estimated_link": 0.4, "estimated_sink_pressure": 0.1}
    reward = compute_step_reward(before, after, {"invalid_action": False}, {"coverage_step": 1.0, "energy_step": 1.0, "link_step": 1.0})
    assert reward > 0


def test_rl_init_step_reward_penalizes_invalid_action():
    before = {"estimated_coverage": 0.1}
    after = {"estimated_coverage": 0.1}
    reward = compute_step_reward(before, after, {"invalid_action": True}, {"invalid_action_penalty": 0.2})
    assert reward < 0
