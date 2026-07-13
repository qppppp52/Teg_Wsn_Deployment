import numpy as np
import pytest

from src.model.individual import Individual
from src.model.solution import Solution
from src.rl_init.config_validation import validate_drl_init_config
from src.rl_init.diversity_filter import filter_population_by_diversity
from src.rl_init.evaluation_counter import DRLInitEvaluationCounter
from src.rl_init.ppo_trainer import rollout_bootstrap_value
from src.rl_init.rollout_buffer import RolloutBuffer
from src.rl_init.source_summary import (
    build_ppo_training_summary,
    build_source_wise_summary,
)


def test_ppo_partial_rollout_uses_bootstrap_value():
    class Agent:
        def value(self, state):
            assert state == "next"
            return 3.25

    assert rollout_bootstrap_value(Agent(), "next", False) == 3.25


def test_ppo_terminal_rollout_uses_zero_bootstrap():
    class Agent:
        def value(self, state):
            raise AssertionError("terminal rollout must not query the critic")

    assert rollout_bootstrap_value(Agent(), "terminal", True) == 0.0


def test_gae_does_not_cross_done_boundary():
    buffer = RolloutBuffer()
    buffer.add({}, 0, 0.0, 0.0, 1.0, True)
    buffer.add({}, 0, 0.0, 0.0, 10.0, False)
    advantages, _ = buffer.compute_returns_and_advantages(
        last_value=5.0,
        gamma=1.0,
        gae_lambda=1.0,
    )
    assert advantages[0] == pytest.approx(1.0)
    assert advantages[1] == pytest.approx(15.0)


def test_drl_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown keys"):
        validate_drl_init_config({"drl_init": {"unknown_option": True}})


def test_drl_config_rejects_deprecated_unused_keys():
    with pytest.raises(ValueError, match="Deprecated"):
        validate_drl_init_config({"drl_init": {"evaluate_during_training": True}})


def test_drl_config_converts_rsum_reference_to_float():
    config = {"drl_init": {"normalization": {"rsum_ref_max": "2.0e8"}}}
    cfg = validate_drl_init_config(config)
    assert cfg["normalization"]["rsum_ref_max"] == 2.0e8
    assert isinstance(cfg["normalization"]["rsum_ref_max"], float)


def test_drl_config_ratio_policy_is_unambiguous():
    config = {
        "drl_init": {
            "drl_ratio": 0.6,
            "heuristic_ratio": 0.3,
            "random_ratio": 0.2,
        }
    }
    with pytest.raises(ValueError, match="must equal 1.0"):
        validate_drl_init_config(config)


def test_source_wise_summary_and_differences():
    rows = [
        _metric("drl", 0.4, 150.0, True),
        _metric("drl", 0.2, 130.0, True),
        _metric("heuristic", 0.1, 100.0, False),
    ]
    source_wise, differences = build_source_wise_summary(rows)
    assert source_wise["drl"]["count"] == 2
    assert source_wise["drl"]["mean_coverage"] == pytest.approx(0.3)
    assert source_wise["random"]["mean_coverage"] is None
    assert differences["drl_minus_heuristic_mean_coverage"] == pytest.approx(0.2)
    assert differences["drl_minus_random_mean_coverage"] is None


def test_ppo_training_summary_counts_only_real_updates():
    rows = [
        {"episode": 0, "episode_reward": 1.0, "feasible": False, "updated_this_episode": False, "ppo_update_count": 0},
        {"episode": 1, "episode_reward": 2.0, "feasible": True, "updated_this_episode": True, "ppo_update_count": 1, "policy_loss": 0.1, "value_loss": 0.2, "entropy": 0.3, "approx_kl": 0.01},
        {"episode": 2, "episode_reward": 3.0, "feasible": True, "updated_this_episode": False, "ppo_update_count": 1},
    ]
    summary = build_ppo_training_summary(rows, window_size=1)
    assert summary["update_count"] == 1
    assert summary["first_update_episode"] == 1
    assert summary["first_window_mean_reward"] == 1.0
    assert summary["last_window_mean_reward"] == 3.0


def test_evaluation_counter_separates_online_and_end_to_end():
    counter = DRLInitEvaluationCounter(
        ppo_terminal_evaluations=10,
        drl_candidate_post_evaluations=2,
        gen0_re_evaluations=4,
        cr_mode_trial_evaluations=8,
    )
    result = counter.to_dict()
    assert result["online_evaluation_count"] == 12
    assert result["total_end_to_end_evaluations"] == 24


def test_diversity_rejects_same_deployment_with_priority_noise():
    first = _candidate([1, 0], [1], 0.3, 100.0, 0.0)
    duplicate = _candidate([1, 0], [1], 0.3, 100.0, 0.2)
    kept = filter_population_by_diversity(
        [first, duplicate],
        target_count=2,
        config=_diversity_config(),
    )
    assert len(kept) == 1
    assert duplicate["diversity_reject_reason"] == "insufficient_diversity"


def test_diversity_accepts_deployment_or_objective_difference():
    first = _candidate([1, 0], [1], 0.3, 100.0, 0.0)
    deployment = _candidate([0, 1], [1], 0.3, 100.0, 0.0)
    objective = _candidate([1, 0], [1], 0.7, 180.0, 0.0)
    kept = filter_population_by_diversity(
        [first, deployment, objective],
        target_count=3,
        config=_diversity_config(),
    )
    reasons = {item.get("diversity_accept_reason") for item in kept}
    assert "deployment_hamming" in reasons
    assert "objective_space" in reasons


def _metric(source, coverage, rsum, feasible):
    return {
        "source_type": source,
        "feasible_before_repair": feasible,
        "feasible": feasible,
        "cv_before_repair": 0.0 if feasible else 1.0,
        "cv": 0.0 if feasible else 1.0,
        "coverage": coverage,
        "rsum_capacity": rsum,
        "repair_iter": 0,
        "quality_score": coverage,
    }


def _candidate(x, y, coverage, rsum, priority_shift):
    individual = Individual(2, 1)
    individual.rho_s = np.array([0.2, 0.8]) + priority_shift
    individual.rho_a = np.array([0.5]) + priority_shift
    solution = Solution(2)
    solution.x = np.asarray(x, dtype=np.int8)
    solution.y = np.asarray(y + [0], dtype=np.int8)[:2]
    solution.coverage = coverage
    solution.rsum_capacity = rsum
    return {
        "individual": individual,
        "solution": solution,
        "quality_score": coverage + rsum / 1000.0,
        "source_type": "drl",
    }


def _diversity_config():
    return {
        "drl_init": {
            "normalization": {"rsum_ref_max": 200.0},
            "diversity": {
                "min_hamming_distance": 0.2,
                "weak_hamming_distance": 0.1,
                "min_priority_l2_distance": 0.05,
                "min_objective_distance": 0.1,
            },
        }
    }
