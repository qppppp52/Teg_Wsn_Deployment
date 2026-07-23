import csv
import os
import uuid
from pathlib import Path
from main import run_experiment
from src.io.config_reader import load_experiment_config


def _writable_test_dir():
    candidates = [
        os.environ.get("TEG_WSN_TEST_TMP"),
        "test_outputs",
        "C:/tmp/teg_wsn_tests",
        "C:/Users/qpppp/Documents/Codex/2026-06-30/new-chat/test_outputs",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            path = Path(candidate)
            path.mkdir(parents=True, exist_ok=True)
            probe = path / f"probe_{uuid.uuid4().hex}.txt"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return path
        except OSError:
            continue
    raise RuntimeError("No writable test directory found")


def test_small_center_heat_experiment_smoke():
    cfg = load_experiment_config("configs/experiment_small_compare.yaml")
    cfg["mode"]["population_size"] = 6
    cfg["mode"]["max_generations"] = 1
    cfg["experiment"]["seeds"] = [42]
    cfg["dqn"]["min_replay_size"] = 9999
    out_dir = _writable_test_dir() / f"small_smoke_{uuid.uuid4().hex}"
    run_experiment(cfg, "cr_mode", str(out_dir), 42)
    assert (out_dir / "data" / "convergence.csv").exists()
    assert (out_dir / "data" / "pareto_solutions.csv").exists()
    assert (out_dir / "figures" / "temperature_faces.png").exists()
    assert (out_dir / "data" / "candidate_temperature.csv").exists()
    with open(out_dir / "data" / "convergence.csv", newline="", encoding="utf-8") as file:
        convergence_fields = next(csv.reader(file))
    assert "sink_duplicate_count_mean" in convergence_fields
    assert "sink_dominant_violation" in convergence_fields
    with open(out_dir / "data" / "pareto_solution_details.csv", newline="", encoding="utf-8") as file:
        detail_fields = next(csv.reader(file))
    assert "sink_duplicate_count" in detail_fields
    assert "sink_shortage_sensor" in detail_fields
