import os
import uuid
from pathlib import Path
import numpy as np
import pytest
from src.dqn.action_space import ACTIONS
from src.dqn.replay_buffer import ReplayBuffer
from src.dqn.state_builder import build_state


def _writable_test_dir():
    candidates = [
        os.environ.get("TEG_WSN_TEST_TMP"),
        "C:/tmp/teg_wsn_tests",
        "C:/Users/qpppp/Documents/Codex/2026-06-30/new-chat/test_outputs",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            path = Path(candidate)
            path.mkdir(parents=True, exist_ok=True)
            probe = path / f"probe_{uuid.uuid4().hex}.txt"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return path
        except OSError:
            continue
    raise RuntimeError("No writable test directory found")


class Pop:
    solutions = []


class Ctx:
    config = {}


def test_dqn_action_space_and_replay_buffer():
    assert len(ACTIONS) >= 10
    rb = ReplayBuffer(10)
    rb.add(np.zeros(3), 1, 0.5, np.ones(3), False)
    assert len(rb) == 1
    batch = rb.sample(1)
    assert batch[0].shape == (1, 3)


def test_dqn_state_builder_empty_population_dimension():
    state = build_state(Pop(), None, Ctx())
    assert state.shape[0] == 19


def test_q_network_forward_and_agent_checkpoint():
    torch = pytest.importorskip("torch")
    from src.dqn.q_network import QNetwork
    from src.dqn.dqn_agent import DQNAgent

    net = QNetwork(state_dim=19, action_dim=len(ACTIONS), hidden_dims=[16, 16])
    x = torch.zeros((2, 19), dtype=torch.float32)
    y = net(x)
    assert tuple(y.shape) == (2, len(ACTIONS))

    cfg = {"dqn": {"hidden_dims": [16, 16], "batch_size": 2, "min_replay_size": 2}}
    agent = DQNAgent(19, len(ACTIONS), cfg)
    ckpt = _writable_test_dir() / f"dqn_{uuid.uuid4().hex}.pt"
    agent.save(str(ckpt))
    loaded = DQNAgent.from_checkpoint(str(ckpt), cfg)
    assert loaded.state_dim == 19
    assert loaded.action_dim == len(ACTIONS)
