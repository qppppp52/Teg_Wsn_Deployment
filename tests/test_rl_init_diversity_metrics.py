import csv
import os
import uuid
from pathlib import Path

import numpy as np

from src.model.individual import Individual
from src.model.solution import Solution
from src.rl_init.diversity_filter import priority_l2_distance, hamming_distance_deployment
from main import run_experiment
from src.io.config_reader import load_experiment_config


def test_priority_l2_distance_zero_for_identical_individuals():
    ind_a = Individual(3, 2)
    ind_a.rho_s = np.array([0.1, 0.2, 0.3])
    ind_a.rho_a = np.array([0.4, 0.5])
    ind_b = ind_a.copy()
    assert priority_l2_distance(ind_a, ind_b) == 0.0


def test_priority_l2_distance_positive_for_different_individuals():
    ind_a = Individual(2, 1)
    ind_b = Individual(2, 1)
    ind_a.rho_s[:] = [0.0, 0.0]
    ind_a.rho_a[:] = [0.0]
    ind_b.rho_s[:] = [1.0, 0.0]
    ind_b.rho_a[:] = [0.5]
    assert priority_l2_distance(ind_a, ind_b) > 0.0


def test_hamming_distance_deployment_positive_for_different_solutions():
    sol_a = Solution(3)
    sol_b = Solution(3)
    sol_a.x[:] = [1, 0, 0]
    sol_b.x[:] = [0, 1, 0]
    sol_a.y[:] = [0, 0, 1]
    sol_b.y[:] = [1, 0, 0]
    assert hamming_distance_deployment(sol_a, sol_b) > 0.0


def test_init_population_metrics_diversity_not_all_zero():
    cfg = load_experiment_config("configs/experiment_small_compare.yaml")
    cfg["mode"]["population_size"] = 4
    cfg["mode"]["max_generations"] = 0
    cfg["experiment"]["seeds"] = [42]
    cfg["drl_init"]["train_episodes"] = 1
    cfg["drl_init"]["rollout_steps"] = 8
    cfg["drl_init"]["minibatch_size"] = 4
    cfg["drl_init"]["update_epochs"] = 1
    cfg["drl_init"]["max_steps_per_episode"] = 6
    cfg["drl_init"]["save_checkpoint"] = False
    cfg["drl_init"]["load_checkpoint_if_exists"] = False
    out_dir = Path("test_outputs") / f"diversity_{uuid.uuid4().hex}"
    run_experiment(cfg, "drl_init_cr_mode", str(out_dir), 42)
    with open(out_dir / "data" / "init_population_metrics.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    values = [float(row["diversity_score"]) for row in rows[1:] if row["diversity_score"] and row["diversity_score"].lower() != "nan"]
    assert values
    assert any(value > 0.0 for value in values)
