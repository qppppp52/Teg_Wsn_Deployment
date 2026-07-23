import os
import uuid
from pathlib import Path

from main import run_experiment
from src.io.config_reader import load_experiment_config


def _output_dir():
    root = Path(os.environ.get("TEG_WSN_TEST_TMP", "test_outputs"))
    root.mkdir(parents=True, exist_ok=True)
    return root / f"dqn_smoke_{uuid.uuid4().hex}"


def test_dqn_cr_mode_train_smoke_uses_shared_finalizer():
    config = load_experiment_config("configs/experiment_small_compare.yaml")
    config["mode"]["population_size"] = 4
    config["mode"]["max_generations"] = 1
    config["experiment"]["seeds"] = [42]
    config["dqn"]["mode"] = "train"
    config["dqn"]["checkpoint"]["save_model"] = False
    output_dir = _output_dir()

    archive, summary, _ = run_experiment(
        config, "dqn_cr_mode", str(output_dir), 42
    )

    assert summary["mode_individual_evaluations"] == 8
    assert summary["boost_invocations"] >= 0
    assert (output_dir / "data" / "dqn_training_action_log.csv").exists()
    assert (output_dir / "data" / "dqn_pressure_contract.json").exists()
    assert (output_dir / "data" / "pareto_solution_details.csv").exists()
    assert archive.solutions
