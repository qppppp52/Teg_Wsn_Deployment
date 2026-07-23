from types import SimpleNamespace

import pytest

from src.constraints.cv_pressure import aggregate_population_pressure
from src.dqn.action_mask import action_mask_diagnostics, build_dqn_action_mask
from src.dqn.action_space import ACTIONS
from src.model.solution import Solution
from tests.dqn_test_config import make_dqn_config


def test_population_pressure_uses_fixed_refs_and_violation_rates():
    first = Solution(1)
    first.cv_deploy = 0.5
    first.cv_link = 2.0
    second = Solution(1)
    second.cv_deploy = 1.5
    second.cv_link = 0.0
    context = SimpleNamespace(config=make_dqn_config())
    summary = aggregate_population_pressure([first, second], context)

    assert summary.mean_pressure["deploy"] == 0.75
    assert summary.mean_pressure["link"] == 0.5
    assert summary.violation_rate["deploy"] == 1.0
    assert summary.violation_rate["link"] == 0.5
    assert summary.dominant_component == "deploy"


def test_ambiguous_pressure_does_not_force_one_repair_family():
    metrics = {
        "FR": 0.5,
        "CV_mean": 0.3,
        "pressure": {
            "deploy": 0.0,
            "link": 0.66,
            "power": 0.70,
            "service": 0.0,
            "sink": 0.0,
            "energy": 0.0,
        },
    }
    config = make_dqn_config()
    config["dqn"]["action_mask"]["thresholds"] = {
        "high_pressure": 0.6,
        "dominant_margin": 0.05,
    }
    mask = build_dqn_action_mask(metrics, ACTIONS, config)
    names = [action["name"] for action in ACTIONS]
    diagnostics = action_mask_diagnostics(mask, ACTIONS, metrics)

    assert bool(mask[names.index("rsum_search_priority")])
    assert diagnostics["dominant_pressure"] == "power"
    assert diagnostics["dominant_pressure_margin"] == pytest.approx(0.04)
