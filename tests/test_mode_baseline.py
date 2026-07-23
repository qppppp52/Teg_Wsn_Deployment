import csv
import os
import uuid
from pathlib import Path

from main import run_experiment
from src.io.config_reader import load_experiment_config


def _writable_test_dir():
    for candidate in (os.environ.get("TEG_WSN_TEST_TMP"), "test_outputs", "C:/tmp/teg_wsn_tests"):
        if not candidate:
            continue
        try:
            path = Path(candidate)
            path.mkdir(parents=True, exist_ok=True)
            return path
        except OSError:
            continue
    raise RuntimeError("No writable test directory found")


def test_mode_smoke_skips_constraint_repair():
    cfg = load_experiment_config("configs/experiment_small_compare.yaml")
    cfg["mode"]["population_size"] = 6
    cfg["mode"]["max_generations"] = 1
    cfg["experiment"]["seeds"] = [42]
    out_dir = _writable_test_dir() / f"mode_smoke_{uuid.uuid4().hex}"

    run_experiment(cfg, "mode", str(out_dir), 42)

    with open(out_dir / "data" / "convergence.csv", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows
    assert all(float(row["mean_repair_iter"]) == 0.0 for row in rows)
    assert all(row["FR_before_repair"] == row["FR_after_repair"] for row in rows)