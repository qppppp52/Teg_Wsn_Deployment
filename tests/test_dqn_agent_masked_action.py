import numpy as np
import pytest

from src.dqn.action_space import ACTIONS

from tests.dqn_test_config import make_dqn_config

def test_masked_actions_are_not_selected_random_or_greedy():
    torch = pytest.importorskip("torch")
    from src.dqn.dqn_agent import DQNAgent

    cfg = make_dqn_config(hidden_dims=(8,), batch_size=2, epsilon=(1.0, 1.0))
    agent = DQNAgent(4, len(ACTIONS), cfg)
    state = np.zeros(4, dtype=np.float32)
    mask = np.zeros(len(ACTIONS), dtype=bool)
    mask[3] = True
    for _ in range(20):
        assert agent.select_action(state, gen=0, action_mask=mask) == 3

    agent.epsilon_start = 0.0
    agent.epsilon_end = 0.0
    with torch.no_grad():
        last = agent.q_net.net[-1]
        last.bias.fill_(0.0)
        last.bias[0] = 100.0
        last.bias[3] = 1.0
    assert agent.select_action(state, gen=0, action_mask=mask) == 3


def test_q_stats_respects_action_mask():
    torch = pytest.importorskip("torch")
    from src.dqn.dqn_agent import DQNAgent

    cfg = make_dqn_config(hidden_dims=(8,), epsilon=(0.0, 0.0))
    agent = DQNAgent(4, len(ACTIONS), cfg)
    state = np.zeros(4, dtype=np.float32)
    mask = np.zeros(len(ACTIONS), dtype=bool)
    mask[2] = True
    mask[4] = True
    q_mean, q_max = agent.q_stats(state, action_mask=mask)
    assert isinstance(q_mean, float)
    assert isinstance(q_max, float)
    assert q_max >= q_mean or np.isclose(q_max, q_mean)
