import numpy as np

from src.dqn.action_mask import build_dqn_action_mask
from src.dqn.action_space import ACTIONS


def test_dqn_action_mask_keeps_at_least_one_action():
    metrics = {
        "FR": 0.0,
        "CV_mean": 10.0,
        "pressure": {"energy": 1.0, "link": 1.0, "capacity": 1.0, "sink": 1.0},
    }
    mask = build_dqn_action_mask(metrics, ACTIONS, {"dqn": {"action_mask": {"enabled": True, "thresholds": {}}}})
    assert mask.dtype == np.bool_
    assert mask.shape == (len(ACTIONS),)
    assert np.any(mask)


def test_dqn_action_mask_blocks_throughput_when_feasibility_is_low():
    metrics = {"FR": 0.05, "CV_mean": 5.0, "pressure": {"energy": 0.7}}
    mask = build_dqn_action_mask(metrics, ACTIONS, {"dqn": {"action_mask": {"enabled": True, "thresholds": {}}}})
    throughput_idx = next(i for i, action in enumerate(ACTIONS) if action["name"] == "rsum_search_priority")
    assert not bool(mask[throughput_idx])


def test_dqn_action_mask_energy_pressure_whitelist():
    metrics = {"FR": 0.5, "CV_mean": 2.0, "pressure": {"energy": 0.8, "link": 0.2, "power": 0.1, "sink": 0.1}}
    mask = build_dqn_action_mask(metrics, ACTIONS, {"dqn": {"action_mask": {"enabled": True, "thresholds": {}}}})
    names = [a["name"] for a in ACTIONS]
    assert bool(mask[names.index("energy_first")])
    assert not bool(mask[names.index("rsum_search_priority")])
    assert not bool(mask[names.index("exploration_high_F")])
    assert not bool(mask[names.index("diversity_boost")])


def test_dqn_action_mask_small_explicit_hv_change_opens_exploration():
    metrics = {
        "FR": 0.5,
        "CV_mean": 2.0,
        "delta_HV": 0.0,
        "pressure": {"energy": 0.8, "link": 0.2, "capacity": 0.1, "sink": 0.1},
    }
    mask = build_dqn_action_mask(metrics, ACTIONS, {"dqn": {"action_mask": {"enabled": True, "thresholds": {}}}})
    names = [a["name"] for a in ACTIONS]
    assert bool(mask[names.index("exploration_high_F")])
    assert bool(mask[names.index("diversity_boost")])


def test_dqn_action_mask_high_fr_low_cv_opens_throughput_and_exploration():
    metrics = {"FR": 0.95, "CV_mean": 0.0, "pressure": {"energy": 0.0, "link": 0.0, "power": 0.0, "sink": 0.0}}
    mask = build_dqn_action_mask(metrics, ACTIONS, {"dqn": {"action_mask": {"enabled": True, "thresholds": {}}}})
    names = [a["name"] for a in ACTIONS]
    assert bool(mask[names.index("rsum_search_priority")])
    assert bool(mask[names.index("exploration_high_F")])
    assert bool(mask[names.index("diversity_boost")])

def test_dqn_action_mask_low_fr_does_not_reopen_exploration_on_zero_hv_delta():
    metrics = {
        "FR": 0.05,
        "CV_mean": 5.0,
        "delta_HV": 0.0,
        "pressure": {"energy": 0.8},
    }
    mask = build_dqn_action_mask(metrics, ACTIONS, {"dqn": {"action_mask": {"enabled": True, "thresholds": {}}}})
    names = [action["name"] for action in ACTIONS]
    assert not bool(mask[names.index("exploration_high_F")])
    assert not bool(mask[names.index("diversity_boost")])
