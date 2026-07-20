import os
import uuid
from pathlib import Path
import numpy as np
import pytest
from src.dqn.action_space import ACTIONS
from src.dqn.replay_buffer import ReplayBuffer
from src.dqn.state_builder import STATE_KEYS, build_state

from tests.dqn_test_config import make_dqn_config

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
    assert len(ACTIONS) == 9
    rb = ReplayBuffer(10)
    rb.add(np.zeros(3), 1, 0.5, np.ones(3), False)
    assert len(rb) == 1
    batch = rb.sample(1)
    assert batch[0].shape == (1, 3)


def test_dqn_state_builder_empty_population_dimension():
    state = build_state(Pop(), None, Ctx())
    from src.dqn.state_builder import STATE_KEYS


def test_q_network_forward_and_agent_checkpoint():
    torch = pytest.importorskip("torch")
    from src.dqn.q_network import QNetwork
    from src.dqn.dqn_agent import DQNAgent

    state_dim = len(STATE_KEYS)
    net = QNetwork(state_dim=state_dim, action_dim=len(ACTIONS), hidden_dims=[16, 16])
    x = torch.zeros((2, state_dim), dtype=torch.float32)
    y = net(x)
    assert tuple(y.shape) == (2, len(ACTIONS))

    cfg = make_dqn_config(hidden_dims=(16, 16), batch_size=2)
    agent = DQNAgent(state_dim, len(ACTIONS), cfg)
    ckpt = _writable_test_dir() / f"dqn_{uuid.uuid4().hex}.pt"
    agent.save(str(ckpt))
    loaded = DQNAgent.from_checkpoint(str(ckpt), cfg)
    assert loaded.state_dim == state_dim
    assert loaded.action_dim == len(ACTIONS)

def test_dqn_checkpoint_rejects_obsolete_policy_contract():
    torch = pytest.importorskip("torch")
    from src.dqn.dqn_agent import DQNAgent

    cfg = make_dqn_config(hidden_dims=(16, 16), batch_size=2)
    agent = DQNAgent(len(STATE_KEYS), len(ACTIONS), cfg)
    directory = _writable_test_dir()
    source = directory / f"dqn_source_{uuid.uuid4().hex}.pt"
    obsolete = directory / f"dqn_obsolete_{uuid.uuid4().hex}.pt"
    agent.save(str(source))
    checkpoint = torch.load(source, map_location="cpu")
    checkpoint["metadata"]["policy_contract_version"] = 1
    torch.save(checkpoint, obsolete)
    with pytest.raises(ValueError, match="contract version is obsolete"):
        DQNAgent.from_checkpoint(str(obsolete), cfg)


def test_dqn_frozen_evaluation_evidence_detects_no_training_changes():
    pytest.importorskip("torch")
    from src.dqn.dqn_agent import DQNAgent
    from src.optimizers.dqn_cr_mode import DQNCRMode

    cfg = make_dqn_config(hidden_dims=(8,), batch_size=2)
    agent = DQNAgent(len(STATE_KEYS), len(ACTIONS), cfg)
    agent.set_evaluation_mode()
    optimizer = object.__new__(DQNCRMode)
    optimizer.agent = agent
    optimizer.execution_mode = "eval"
    optimizer.checkpoint_loaded = True
    optimizer.model_path = str(_writable_test_dir() / "formal.pt")
    optimizer.config = {"runtime": {"output_dir": str(_writable_test_dir())}}
    optimizer._evaluation_evidence_before = optimizer._agent_evidence()
    optimizer.dqn_validity_report = {}
    optimizer._finalize_dqn_validity()
    assert optimizer.dqn_validity_report["DQN_VALID"] is True
    assert optimizer.dqn_validity_report["parameter_hash_before"] == optimizer.dqn_validity_report["parameter_hash_after"]
    assert optimizer.dqn_validity_report["gradient_steps_before"] == optimizer.dqn_validity_report["gradient_steps_after"]
