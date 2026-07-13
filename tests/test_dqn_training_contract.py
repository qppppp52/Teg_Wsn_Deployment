import numpy as np
import pytest

from src.dqn.config import DQNConfig
from src.dqn.replay_buffer import ReplayBuffer
from src.model.individual import Individual
from src.model.solution import Solution
from src.optimizers.generation_executor import environmental_select


def _solution(coverage, rsum):
    solution = Solution(1)
    solution.coverage = coverage
    solution.rsum_capacity = rsum
    solution.feasible = True
    solution.cv = 0.0
    return solution


def test_legacy_flat_dqn_config_is_rejected():
    with pytest.raises(ValueError, match="Legacy flat DQN keys"):
        DQNConfig.from_mapping({"dqn": {"hidden_dims": [8]}})


def test_replay_buffer_preserves_action_masks():
    replay = ReplayBuffer(4)
    mask = np.array([True, False, True])
    next_mask = np.array([False, True, False])
    replay.add(np.zeros(2), 0, 0.5, np.ones(2), False, mask, next_mask)
    sample = replay.sample(1)
    np.testing.assert_array_equal(sample[5][0], mask)
    np.testing.assert_array_equal(sample[6][0], next_mask)


def test_environmental_selection_keeps_nondominated_tradeoff():
    individuals = [Individual(1, 1) for _ in range(3)]
    solutions = [
        _solution(0.8, 100.0),
        _solution(0.7, 120.0),
        _solution(0.6, 90.0),
    ]
    _, selected = environmental_select(individuals, solutions, 2)
    objectives = {(solution.coverage, solution.rsum_capacity) for solution in selected}
    assert objectives == {(0.8, 100.0), (0.7, 120.0)}