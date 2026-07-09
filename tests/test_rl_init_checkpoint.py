from pathlib import Path

import pytest

from src.rl_init.checkpoint import resolve_checkpoint_path, save_checkpoint, load_checkpoint


torch = pytest.importorskip("torch")


class TinyAgent:
    def __init__(self):
        self.device = torch.device("cpu")
        self.policy = torch.nn.Linear(3, 2)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1.0e-3)


def test_checkpoint_path_uses_per_seed_template():
    cfg = {
        "experiment": {"name": "small_center_heat_compare"},
        "objectives": {"throughput_metric": "actual"},
        "drl_init": {
            "checkpoint_mode": "per_seed",
            "checkpoint_dir": "experiments/checkpoints",
            "checkpoint_name_template": "drl_init_{scene_name}_{throughput_metric}_seed_{seed}.pt",
            "seed": 43,
        },
    }
    path = resolve_checkpoint_path(cfg, seed=43)
    assert "seed_43" in path
    assert path.endswith(".pt")


def test_save_and_load_checkpoint_with_meta():
    path = Path("test_outputs") / "checkpoint_meta" / "policy.pt"
    meta = {
        "scene_name": "small_center_heat_compare",
        "seed": 42,
        "throughput_metric": "actual",
        "num_candidates": 5,
        "candidate_feature_dim": 3,
        "global_feature_dim": 4,
        "hidden_dim": 16,
        "action_roles": ["sensor", "ap", "skip", "stop"],
        "max_selected_sensors": 3,
        "max_selected_aps": 1,
        "config_hash": "abc123",
    }
    agent = TinyAgent()
    save_checkpoint(agent, str(path), meta=meta)
    loaded_agent = TinyAgent()
    result = load_checkpoint(loaded_agent, str(path), expected_meta=meta)
    assert result["loaded"] is True
    assert result["compatible"] is True
    assert result["meta"]["scene_name"] == "small_center_heat_compare"
    assert result["meta"]["config_hash"] == "abc123"
