from types import SimpleNamespace

import numpy as np
import pytest

from src.model.individual import Individual
from src.model.solution import Solution
from src.rl_init.population_generator import DRLInitPopulationGenerator


def test_drl_init_validity_for_trained_policy_and_failure_reasons():
    generator = DRLInitPopulationGenerator.__new__(DRLInitPopulationGenerator)
    generator.cfg = {"min_accepted_drl_ratio": 0.5, "training_summary_window": 10}
    generator.torch_available = True
    generator.policy_source = "trained"
    generator.accepted_counts = {"drl": 2, "heuristic": 1, "random": 1}
    generator.training_log = [
        {
            "episode": 0,
            "episode_reward": 1.0,
            "feasible": True,
            "updated_this_episode": True,
            "ppo_update_count": 1,
        }
    ]
    generator.checkpoint_load_attempted = False
    generator.checkpoint_load_skip_reason = ""
    generator._evaluate_validity(4)
    assert generator.drl_init_valid
    assert generator.drl_init_invalid_reasons == []

    generator.policy_source = "fallback"
    generator.torch_available = False
    generator.accepted_counts["drl"] = 0
    generator._evaluate_validity(4)
    assert not generator.drl_init_valid
    assert "torch_unavailable" in generator.drl_init_invalid_reasons
    assert "policy_source_fallback" in generator.drl_init_invalid_reasons
    assert "accepted_drl_count_below_threshold" in generator.drl_init_invalid_reasons


def test_drl_generation_retries_until_target(monkeypatch):
    generator = _generator(monkeypatch, deployments=[[1, 0], [1, 0], [0, 1]])
    population = generator.generate(4)
    assert len(population.individuals) == 4
    assert generator.accepted_counts["drl"] == 2
    assert generator.drl_candidate_count == 3
    assert generator.drl_retry_count == 1
    assert generator.fallback_fill_count == 0


def test_drl_generation_records_fallback_after_retry_limit(monkeypatch):
    generator = _generator(monkeypatch, deployments=[[1, 0], [1, 0], [1, 0]])
    population = generator.generate(4)
    assert len(population.individuals) == 4
    assert generator.accepted_counts["drl"] == 1
    assert generator.fallback_fill_heuristic_count == 1
    assert generator.fallback_fill_count == 1
    assert generator.fallback_reason == "drl_diversity_retry_exhausted"


def _generator(monkeypatch, deployments):
    config = {
        "drl_init": {
            "drl_ratio": 0.5,
            "heuristic_ratio": 0.25,
            "random_ratio": 0.25,
            "min_accepted_drl_ratio": 0.0,
            "invalid_run_policy": "mark_and_continue",
            "normalization": {"rsum_ref_max": 200.0},
            "diversity": {
                "min_hamming_distance": 0.2,
                "weak_hamming_distance": 0.1,
                "min_priority_l2_distance": 0.1,
                "min_objective_distance": 0.1,
                "max_duplicate_retry": 1,
                "max_drl_candidate_evaluations": 3,
            },
        }
    }
    mapping = SimpleNamespace(num_sensor_candidates=2, num_ap_candidates=1)
    context = SimpleNamespace(index_mapping=mapping)
    generator = DRLInitPopulationGenerator(context, config)
    generator.trainer = object()
    generator.torch_available = True
    generator.policy_source = "trained"
    generator.training_log = [
        {
            "episode": 0,
            "episode_reward": 1.0,
            "feasible": True,
            "updated_this_episode": True,
            "ppo_update_count": 1,
        }
    ]
    sequence = iter(deployments)
    generator.generate_one = lambda deterministic=False: _individual()

    def fake_evaluate(individual, ctx, cfg, source_type):
        if source_type == "drl":
            deployment = next(sequence)
        else:
            deployment = [0, 1]
        solution = Solution(2)
        solution.x = np.asarray(deployment, dtype=np.int8)
        solution.y = np.asarray([1, 0], dtype=np.int8)
        solution.coverage = 0.3
        solution.rsum_capacity = 100.0
        solution.feasible = True
        solution.cv = 0.0
        return {
            "individual": individual,
            "solution": solution,
            "source_type": source_type,
            "quality_score": 1.0,
        }

    monkeypatch.setattr(
        "src.rl_init.population_generator.evaluate_init_individual",
        fake_evaluate,
    )
    monkeypatch.setattr(
        "src.rl_init.population_generator.create_composite_heuristic_individual",
        lambda ctx, cfg, rng: _individual(),
    )
    return generator


def _individual():
    individual = Individual(2, 1)
    individual.rho_s = np.asarray([0.2, 0.8])
    individual.rho_a = np.asarray([0.5])
    return individual
