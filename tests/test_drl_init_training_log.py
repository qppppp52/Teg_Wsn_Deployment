import csv
import uuid
from pathlib import Path

from main import run_experiment
from src.io.config_reader import load_experiment_config


REQUIRED_COLUMNS = {
    "episode", "episode_steps", "total_reward", "step_reward_sum", "terminal_reward",
    "feasible", "cv", "cv_deploy", "cv_link", "cv_power", "cv_energy", "cv_energy_sensor", "cv_energy_ap", "cv_sink",
    "cv_service", "coverage", "rsum_capacity", "repair_iter",
    "repair_success", "num_sensors", "num_aps", "invalid_action_count", "policy_loss",
    "value_loss", "entropy", "approx_kl", "loaded_checkpoint", "policy_source", "torch_available",
    "checkpoint_path", "checkpoint_compatible", "checkpoint_skip_reason", "checkpoint_error", "checkpoint_mode", "config_hash",
}


def test_drl_init_training_log_contains_terminal_columns():
    cfg = load_experiment_config("configs/experiment_small_compare.yaml")
    cfg["mode"]["population_size"] = 4
    cfg["mode"]["max_generations"] = 0
    cfg["experiment"]["seeds"] = [42]
    cfg["drl_init"]["train_episodes"] = 2
    cfg["drl_init"]["rollout_steps"] = 8
    cfg["drl_init"]["minibatch_size"] = 4
    cfg["drl_init"]["update_epochs"] = 1
    cfg["drl_init"]["max_steps_per_episode"] = 6
    cfg["drl_init"]["save_checkpoint"] = False
    cfg["drl_init"]["load_checkpoint_if_exists"] = False
    out_dir = Path("test_outputs") / f"training_log_{uuid.uuid4().hex}"
    run_experiment(cfg, "drl_init_cr_mode", str(out_dir), 42)
    with open(out_dir / "data" / "drl_init_training_log.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert REQUIRED_COLUMNS.issubset(set(reader.fieldnames or []))
    assert rows

