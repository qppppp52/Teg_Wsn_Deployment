import os
import uuid
from pathlib import Path
from main import run_experiment
from src.io.config_reader import load_experiment_config


def _writable_test_dir():
    candidates = [os.environ.get("TEG_WSN_TEST_TMP"), "test_outputs", "C:/tmp/teg_wsn_tests"]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / f"probe_{uuid.uuid4().hex}.txt"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return path
        except OSError:
            continue
    raise RuntimeError("No writable test directory found")


def test_drl_init_cr_mode_smoke():
    cfg = load_experiment_config("configs/experiment_small_compare.yaml")
    cfg["mode"]["population_size"] = 4
    cfg["mode"]["max_generations"] = 1
    cfg["experiment"]["seeds"] = [42]
    cfg["drl_init"]["train_episodes"] = 1
    cfg["drl_init"]["rollout_steps"] = 8
    cfg["drl_init"]["minibatch_size"] = 4
    cfg["drl_init"]["update_epochs"] = 1
    cfg["drl_init"]["max_steps_per_episode"] = 6
    cfg["drl_init"]["max_selected_sensors"] = 4
    cfg["drl_init"]["max_selected_aps"] = 1
    cfg["drl_init"]["min_selected_sensors"] = 1
    cfg["drl_init"]["min_selected_aps"] = 1
    out_dir = _writable_test_dir() / f"drl_init_smoke_{uuid.uuid4().hex}"
    run_experiment(cfg, "drl_init_cr_mode", str(out_dir), 42)
    assert (out_dir / "data" / "convergence.csv").exists()
    assert (out_dir / "data" / "init_population_metrics.csv").exists()

