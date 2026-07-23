from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.dqn.config import DQNConfig
from src.dqn.dqn_trainer import DQNTrainer
from src.dqn.replay_buffer import ReplayBuffer
from src.model.individual import Individual
from src.model.solution import Solution
from src.optimizers.generation_executor import environmental_select
from tests.dqn_test_config import make_dqn_config


def _solution(coverage, rsum):
    solution = Solution(1)
    solution.coverage = coverage
    solution.rsum_capacity = rsum
    solution.feasible = True
    solution.cv = 0.0
    solution.constraint_report = SimpleNamespace(feasible=True, state_revision=0)
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

def test_invalid_nested_values_are_rejected_before_training():
    config = make_dqn_config()
    config["dqn"]["training"]["validation_interval"] = 0
    with pytest.raises(ValueError, match="validation interval"):
        DQNConfig.from_mapping(config)


def test_unknown_action_mask_threshold_is_rejected():
    config = make_dqn_config()
    config["dqn"]["action_mask"]["thresholds"] = {"unknown_threshold": 1.0}
    with pytest.raises(ValueError, match="Unknown keys"):
        DQNConfig.from_mapping(config)

def test_short_training_overwrites_stale_best_checkpoint(tmp_path, monkeypatch):
    config = make_dqn_config()
    config["project_root"] = str(tmp_path)
    config["dqn"]["checkpoint_path"] = "experiments/checkpoints/test_best.pt"
    config["dqn"]["training"]["episodes"] = 1
    config["dqn"]["training"]["validation_interval"] = 2
    config["experiment"] = {
        "seeds": [42],
        "dqn_training_seeds": [1000],
        "dqn_validation_seeds": [142],
    }
    trainer = DQNTrainer(config)
    stale = Path(trainer.best_path)
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"stale")

    class FakeAgent:
        interaction_steps = 1
        gradient_steps = 1
        last_loss = 0.1

        def epsilon(self):
            return 0.5

        def save(self, path):
            output = Path(path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"current-run")

    fake_agent = FakeAgent()
    fake_optimizer = SimpleNamespace(
        agent=fake_agent,
        convergence_history={"HV": [0.25]},
    )
    monkeypatch.setattr(trainer, "_run_episode", lambda *args, **kwargs: fake_optimizer)
    trainer.train()
    assert stale.read_bytes() == b"current-run"
    assert trainer.best_checkpoint_written is True
