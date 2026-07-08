import numpy as np

from src.dqn.action_mask import build_dqn_action_mask
from src.dqn.action_space import ACTIONS


def test_dqn_action_mask_keeps_at_least_one_action():
    metrics = {
        "FR": 0.0,
        "CV_mean": 10.0,
        "pressure": {"energy": 1.0, "link": 1.0, "capacity": 1.0, "sink": 1.0},
    }
    mask = build_dqn_action_mask(metrics, ACTIONS, {"dqn": {"action_mask_enabled": True}})
    assert mask.dtype == np.bool_
    assert mask.shape == (len(ACTIONS),)
    assert np.any(mask)


def test_dqn_action_mask_blocks_throughput_when_feasibility_is_low():
    metrics = {"FR": 0.05, "CV_mean": 5.0, "pressure": {"energy": 0.7}}
    mask = build_dqn_action_mask(metrics, ACTIONS, {"dqn": {"action_mask_enabled": True}})
    throughput_idx = next(i for i, action in enumerate(ACTIONS) if action["name"] == "throughput_priority")
    assert not bool(mask[throughput_idx])
