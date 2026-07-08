#!/usr/bin/env python
"""WSN/TEG deployment experiment entry point."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.io.config_reader import load_config, load_experiment_config
from src.scene.scenario_builder import Scenario
from src.preprocessing.preprocessor import run_preprocessing
from src.optimizers.cr_mode import CRMode
from src.optimizers.dqn_cr_mode import DQNCRMode
from src.optimizers.drl_init_cr_mode import DRLInitCRMode
from src.visualization.plot_convergence import plot_convergence, plot_dual_convergence
from src.visualization.plot_deployment import plot_deployment
from src.visualization.temperature_plot import export_candidate_temperature, plot_temperature_faces, plot_pgrid_distribution
from src.io.result_io import save_log
from src.physics.analytic_temperature import sample_heat_source_ids
from src.utils.seed import set_seed
from src.utils.logger import get_logger

logger = get_logger("main")


BASE_SUMMARY_FIELDS = [
    "algorithm", "seed", "final_FR_after_repair", "final_CV_after_repair",
    "final_HV", "archive_size", "pareto_count", "best_coverage",
    "best_rsum_actual", "best_rsum_capacity", "recommended_coverage",
    "recommended_rsum_actual", "recommended_rsum_capacity",
    "first_feasible_generation", "mean_repair_iter", "repair_success_rate",
    "saturated_link_ratio", "runtime_seconds",
    "final_FR", "final_CV_mean", "best_rsum", "recommended_rsum",
    "throughput_capacity_best", "throughput_actual_best",
]


FACE_REPRESENTATIVE_ROLES = [
    "coverage_best", "rsum_best", "recommended_compromise",
]


def _save_context(ctx, path):
    """Save scenario context for standalone visualization scripts."""
    np.savez_compressed(
        path,
        candidate_points=ctx.candidate_points_full,
        target_coords=ctx.target_coords,
        T_r=ctx.T_r,
        P_grid=ctx.P_grid,
        neighbor_sets=_pack_neighbors(ctx.neighbor_sets, ctx.num_candidates),
        nmax=ctx.nmax,
        config_dict=str(ctx.config),
        num_candidates=ctx.num_candidates,
        num_targets=ctx.num_targets,
        Rs=ctx.Rs,
    )


def _pack_neighbors(neighbor_sets, num_candidates):
    max_len = max(len(ns) for ns in neighbor_sets) if neighbor_sets else 0
    mat = np.full((num_candidates, max_len), -1, dtype=np.int32)
    for idx, neighbors in enumerate(neighbor_sets):
        mat[idx, :len(neighbors)] = neighbors
    return mat


def _save_heat_sources(ctx, seed, path):
    """Persist heat-source metadata for standalone visualization scripts."""
    if ctx.config.get("temperature", {}).get("model") == "center_air_convection_synthetic":
        np.savez_compressed(
            path,
            heat_source_coords=np.asarray([ctx.config["temperature"].get("source_position", [0, 0, 0])], dtype=float),
            heat_source_T=np.asarray([ctx.config["temperature"].get("source_temperature", 343.15)], dtype=float),
            heat_source_model="center_air_convection_synthetic",
        )
        return
    source_ids = sample_heat_source_ids(ctx.candidate_points_full, seed=seed, config=ctx.config)
    source_ids = np.asarray(source_ids, dtype=np.int32)
    candidate_points = ctx.candidate_points_full
    np.savez_compressed(
        path,
        heat_source_ids=source_ids,
        heat_source_coords=candidate_points[source_ids, :3].astype(float),
        heat_source_faces=candidate_points[source_ids, 3].astype(np.int32),
        heat_source_T=ctx.T_r[source_ids].astype(float),
    )


def _make_optimizer(algorithm, ctx, config):
    if algorithm == "cr_mode":
        return CRMode(ctx, config)
    if algorithm == "dqn_cr_mode":
        return DQNCRMode(ctx, config)
    if algorithm == "drl_init_cr_mode":
        return DRLInitCRMode(ctx, config)
    raise ValueError(f"Unknown algorithm: {algorithm}")


def run_experiment(config, algorithm="cr_mode", output_dir="results", seed=None):
    """Run one algorithm once and write standardized outputs."""
    config = copy.deepcopy(config)
    seed = int(seed if seed is not None else config.get("experiment", {}).get("seeds", [42])[0])
    config.setdefault("experiment", {})["seeds"] = [seed]
    config.setdefault("runtime", {})["output_dir"] = output_dir
    set_seed(seed)
    logger.info(f"Seed={seed}, algorithm={algorithm}")

    data_dir = os.path.join(output_dir, "data")
    fig_dir = os.path.join(output_dir, "figures")
    pareto_dir = os.path.join(output_dir, "pareto")
    log_dir = os.path.join(output_dir, "logs")
    for directory in (data_dir, fig_dir, pareto_dir, log_dir):
        os.makedirs(directory, exist_ok=True)

    scenario = Scenario(config).build()
    ctx = run_preprocessing(scenario, config, seed)
    ctx.config = config
    logger.info(f"Scenario: {scenario.num_candidates} candidates, {scenario.num_targets} targets")

    start = time.time()
    algo = _make_optimizer(algorithm, ctx, config)
    archive = algo.run()
    runtime_seconds = time.time() - start

    archive.save(os.path.join(pareto_dir, f"{algorithm}.npz"))
    _save_context(ctx, os.path.join(pareto_dir, f"{algorithm}_context.npz"))
    _save_heat_sources(ctx, seed, os.path.join(pareto_dir, f"{algorithm}_heat_sources.npz"))
    export_candidate_temperature(ctx, os.path.join(data_dir, "candidate_temperature.csv"))
    plot_temperature_faces(ctx, os.path.join(fig_dir, "temperature_faces.png"))
    plot_pgrid_distribution(ctx, os.path.join(fig_dir, "pgrid_distribution.png"))

    history = getattr(algo, "convergence_history", {})
    representatives = _representative_solutions(archive.solutions, config)
    _save_convergence_csv(history, os.path.join(data_dir, "convergence.csv"))
    _save_pareto_csv(archive.solutions, representatives, os.path.join(data_dir, "pareto_solutions.csv"))
    _save_recommended_solution_csv(representatives, os.path.join(data_dir, "recommended_solutions.csv"))
    _plot_standard_convergence(history, fig_dir, algorithm, config)

    feasible_objectives = archive.get_feasible_objectives()
    if len(feasible_objectives) > 0:
        _plot_pareto_with_representatives(archive.solutions, representatives, os.path.join(fig_dir, "pareto_front.png"), algorithm)
        rec_solution = representatives.get("recommended_compromise")
        if rec_solution is not None:
            try:
                plot_deployment(rec_solution, ctx, os.path.join(fig_dir, "deployment_recommended.png"))
            except Exception as exc:
                logger.warning(f"Deployment plot failed: {exc}")
    else:
        logger.warning(f"No feasible solutions for {algorithm}")

    summary = _build_summary(algorithm, seed, archive, history, runtime_seconds, config, algo, representatives)
    with open(os.path.join(data_dir, "final_summary.json"), "w", encoding="utf-8") as f:
        json.dump(_json_ready(summary), f, indent=2)
    save_log(json.dumps(_json_ready(summary), indent=2), os.path.join(log_dir, f"{algorithm}.log"))
    logger.info(f"{_display_algorithm_name(algorithm)} finished")
    return archive, summary, history


def run_compare_experiment(config):
    """Run all configured algorithms/seeds and write cross-algorithm outputs."""
    exp = config.get("experiment", {})
    name = exp.get("name", "small_center_heat_compare")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join("experiments", name, timestamp)
    algorithms = exp.get("algorithms", ["cr_mode"])
    seeds = exp.get("seeds", [42])[: int(exp.get("num_runs", len(exp.get("seeds", [42]))))]
    summaries = []
    histories = {}
    pareto_by_algorithm = {}

    for algorithm in algorithms:
        for seed in seeds:
            out_dir = os.path.join(root, algorithm, f"seed_{seed}")
            archive, summary, history = run_experiment(config, algorithm, out_dir, seed)
            summaries.append(summary)
            histories[(algorithm, seed)] = history
            pareto_by_algorithm.setdefault(algorithm, []).append(archive.get_feasible_objectives())

    os.makedirs(os.path.join(root, "figures"), exist_ok=True)
    _save_summary_csv(summaries, os.path.join(root, "summary_all_algorithms.csv"))
    _save_recommended_all_csv(summaries, os.path.join(root, "recommended_solutions_all_algorithms.csv"))
    _plot_pareto_compare(pareto_by_algorithm, os.path.join(root, "figures", "pareto_compare_all.png"))
    _plot_history_compare(histories, "HV", os.path.join(root, "figures", "hv_compare_all.png"), "Hypervolume")
    _plot_history_compare(histories, "FR_current", os.path.join(root, "figures", "fr_compare_all.png"), "Feasible Ratio")
    logger.info("summary_all_algorithms.csv generated")
    logger.info("pareto_compare_all.png generated")


def _save_convergence_csv(history, path):
    if not history:
        return
    keys = list(history.keys())
    rows = max(len(v) for v in history.values())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["generation"] + keys)
        writer.writeheader()
        for i in range(rows):
            row = {"generation": i}
            for key in keys:
                row[key] = history[key][i] if i < len(history[key]) else ""
            writer.writerow(row)


def _save_pareto_csv(solutions, representatives, path):
    feasible = [sol for sol in solutions if sol.feasible]
    rec = representatives.get("recommended_compromise")
    cov_best = representatives.get("coverage_best")
    rsum_best = representatives.get("rsum_best")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "solution_id", "coverage", "rsum_actual", "rsum_capacity", "cv",
            "active_sensors", "active_aps", "recommended_flag",
            "coverage_best_flag", "rsum_best_flag",
        ])
        for idx, solution in enumerate(feasible):
            writer.writerow([
                idx,
                solution.coverage,
                solution.metadata.get("throughput_actual", solution.throughput),
                solution.metadata.get("throughput_capacity", solution.throughput),
                solution.cv,
                int(np.sum(solution.x)),
                int(np.sum(solution.y)),
                int(solution is rec),
                int(solution is cov_best),
                int(solution is rsum_best),
            ])

def _save_recommended_solution_csv(representatives, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["role", "coverage", "rsum", "throughput_capacity", "throughput_actual", "cv", "num_sensors", "num_aps", "repair_iter"])
        for role in FACE_REPRESENTATIVE_ROLES:
            solution = representatives.get(role)
            if solution is None:
                continue
            writer.writerow([
                role,
                solution.coverage,
                solution.throughput,
                solution.metadata.get("throughput_capacity", solution.throughput),
                solution.metadata.get("throughput_actual", solution.throughput),
                solution.cv,
                int(np.sum(solution.x)),
                int(np.sum(solution.y)),
                getattr(solution, "repair_iter", 0),
            ])


def _plot_standard_convergence(history, fig_dir, algorithm, config):
    if not history:
        return
    tol = config.get("constraints", {}).get("feasible_tol", 1.0e-8)
    plot_convergence(history.get("FR_current", []), "Feasible Ratio", os.path.join(fig_dir, "convergence_fr.png"), title=f"{algorithm}: FR")
    _plot_two_series(history.get("FR_before_repair", []), "Before repair", history.get("FR_after_repair", []), "After repair", "Feasible Ratio", os.path.join(fig_dir, "convergence_fr_before_after.png"), f"{algorithm}: FR before/after repair")
    plot_convergence(history.get("HV", []), "HV", os.path.join(fig_dir, "convergence_hv.png"), title=f"{algorithm}: HV")
    plot_convergence(history.get("CV_mean", []), "Mean CV", os.path.join(fig_dir, "convergence_cv.png"), title=f"{algorithm}: CV", hline=tol, hline_label=f"CV tol={tol:g}")
    _plot_two_series(history.get("CV_before_repair_mean", []), "Before repair", history.get("CV_after_repair_mean", []), "After repair", "Mean CV", os.path.join(fig_dir, "convergence_cv_before_after.png"), f"{algorithm}: CV before/after repair")
    plot_dual_convergence(history.get("Coverage_feasible", []), "Avg Coverage", history.get("archive_best_coverage", []), "Best Coverage", "Coverage", os.path.join(fig_dir, "convergence_coverage.png"), title=f"{algorithm}: Coverage")
    plot_dual_convergence(history.get("Rsum_feasible_mbps", []), "Avg Rsum Mbps", history.get("archive_best_rsum_mbps", []), "Best Rsum Mbps", "Mbps", os.path.join(fig_dir, "convergence_rsum.png"), title=f"{algorithm}: Rsum", fmt="%.1f")
    plot_convergence([v / 1e6 for v in history.get("rsum_actual_best", [])], "Best actual Rsum (Mbps)", os.path.join(fig_dir, "convergence_rsum_actual.png"), title=f"{algorithm}: Actual Rsum")
    plot_convergence([v / 1e6 for v in history.get("rsum_capacity_best", [])], "Best capacity Rsum (Mbps)", os.path.join(fig_dir, "convergence_rsum_capacity.png"), title=f"{algorithm}: Capacity Rsum")


def _plot_two_series(y1, label1, y2, label2, ylabel, out_path, title):
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    if y1:
        ax.plot(range(len(y1)), y1, label=label1, linewidth=1.8)
    if y2:
        ax.plot(range(len(y2)), y2, label=label2, linewidth=1.8)
    ax.set_xlabel("Generation")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _representative_solutions(solutions, config):
    feasible = [s for s in solutions if s.feasible]
    if not feasible:
        return {}
    objs = np.array([[s.coverage, s.throughput] for s in feasible], dtype=float)
    rmax = max(float(config.get("evaluation", {}).get("rsum_ref_max", np.max(objs[:, 1]))), 1.0)
    norm = np.column_stack([objs[:, 0], np.clip(objs[:, 1] / rmax, 0.0, 1.0)])
    rec_idx = int(np.argmin(np.linalg.norm(1.0 - norm, axis=1)))
    cov_idx = int(np.argmax(objs[:, 0]))
    rsum_idx = int(np.argmax(objs[:, 1]))
    return {
        "coverage_best": feasible[cov_idx],
        "rsum_best": feasible[rsum_idx],
        "recommended_compromise": feasible[rec_idx],
    }


def _plot_pareto_with_representatives(solutions, representatives, out_path, algorithm):
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    feasible = [s for s in solutions if s.feasible]
    if not feasible:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    objs = np.array([[s.coverage, s.throughput / 1e6] for s in feasible], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(objs[:, 0], objs[:, 1], c="#7FB3D5", s=70, edgecolors="#1B4F72", linewidth=1.0, label="Feasible Pareto")

    markers = {
        "coverage_best": ("^", "#2ECC71", "Coverage-best"),
        "rsum_best": ("s", "#E67E22", "Rsum-best"),
        "recommended_compromise": ("*", "#F1C40F", "Recommended"),
    }
    for role, (marker, color, label) in markers.items():
        solution = representatives.get(role)
        if solution is None:
            continue
        ax.scatter([solution.coverage], [solution.throughput / 1e6], marker=marker, c=color, s=180, edgecolors="black", linewidth=1.2, label=label, zorder=5)
        ax.annotate(label, (solution.coverage, solution.throughput / 1e6), textcoords="offset points", xytext=(8, 8), fontsize=8)

    ax.set_xlabel("Coverage")
    ax.set_ylabel("Rsum (Mbps)")
    ax.set_title(f"{algorithm}: Pareto Front")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _build_summary(algorithm, seed, archive, history, runtime_seconds, config, algo, representatives):
    feasible = [s for s in archive.solutions if s.feasible]
    objs = archive.get_feasible_objectives()
    rec = representatives.get("recommended_compromise")
    best_actual = _best_metadata_value(feasible, "throughput_actual")
    best_capacity = _best_metadata_value(feasible, "throughput_capacity")
    rec_actual = float(rec.metadata.get("throughput_actual", rec.throughput)) if rec is not None else 0.0
    rec_capacity = float(rec.metadata.get("throughput_capacity", rec.throughput)) if rec is not None else 0.0
    summary = {
        "algorithm": algorithm,
        "seed": seed,
        "final_FR_after_repair": history.get("FR_after_repair", history.get("FR_current", [0.0]))[-1] if history else 0.0,
        "final_CV_after_repair": history.get("CV_after_repair_mean", history.get("CV_mean", [float("nan")]))[-1] if history else float("nan"),
        "final_HV": history.get("HV", [0.0])[-1] if history else 0.0,
        "archive_size": len(archive),
        "pareto_count": len(feasible),
        "best_coverage": float(np.max(objs[:, 0])) if len(objs) else 0.0,
        "best_rsum_actual": best_actual,
        "best_rsum_capacity": best_capacity,
        "recommended_coverage": rec.coverage if rec is not None else 0.0,
        "recommended_rsum_actual": rec_actual,
        "recommended_rsum_capacity": rec_capacity,
        "first_feasible_generation": _first_feasible_generation(history),
        "mean_repair_iter": history.get("mean_repair_iter", [0.0])[-1] if history else 0.0,
        "repair_success_rate": history.get("repair_success_rate", [0.0])[-1] if history else 0.0,
        "saturated_link_ratio": history.get("saturated_link_ratio", [0.0])[-1] if history else 0.0,
        "runtime_seconds": runtime_seconds,
        "final_FR": history.get("FR_current", [0.0])[-1] if history else 0.0,
        "final_CV_mean": history.get("CV_mean", [float("nan")])[-1] if history else float("nan"),
        "best_rsum": float(np.max(objs[:, 1])) if len(objs) else 0.0,
        "recommended_rsum": rec.throughput if rec is not None else 0.0,
        "throughput_capacity_best": best_capacity,
        "throughput_actual_best": best_actual,
    }
    if algorithm == "dqn_cr_mode":
        training_log = getattr(algo, "training_log", [])
        rewards = [float(row.get("reward", 0.0)) for row in training_log]
        actions = [row.get("action_name", "") for row in training_log]
        summary.update({
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "last_epsilon": float(training_log[-1].get("epsilon", float("nan"))) if training_log else float("nan"),
            "most_used_action": max(set(actions), key=actions.count) if actions else "",
            "dqn_model_path": getattr(algo, "model_path", None) or "",
        })
    if algorithm == "drl_init_cr_mode":
        init_metrics = getattr(algo, "init_metrics", [])
        init_cvs = [float(row["cv"]) for row in init_metrics]
        init_feasible = [bool(row["feasible"]) for row in init_metrics]
        accepted = getattr(algo, "accepted_counts", {}) or {}
        drl_training_time = float(getattr(algo, "init_train_seconds", 0.0))
        online_optimization_time = max(float(runtime_seconds) - drl_training_time, 0.0)
        summary.update({
            "drl_training_time": drl_training_time,
            "online_optimization_time": online_optimization_time,
            "total_time": float(runtime_seconds),
            "init_FR_before_repair": float(np.mean([bool(row.get("feasible_before_repair", False)) for row in init_metrics])) if init_metrics else 0.0,
            "init_CV_before_repair": float(np.mean([float(row.get("cv_before_repair", row["cv"])) for row in init_metrics])) if init_metrics else float("nan"),
            "init_FR_after_repair": float(np.mean(init_feasible)) if init_feasible else 0.0,
            "init_CV_after_repair": float(np.mean(init_cvs)) if init_cvs else float("nan"),
            "init_FR": float(np.mean(init_feasible)) if init_feasible else 0.0,
            "init_CV_mean": float(np.mean(init_cvs)) if init_cvs else float("nan"),
            "init_diversity": _init_source_diversity(init_metrics),
            "accepted_drl_count": int(accepted.get("drl", 0)),
            "accepted_heuristic_count": int(accepted.get("heuristic", 0)),
            "accepted_random_count": int(accepted.get("random", 0)),
            "drl_train_episodes": int(config.get("drl_init", {}).get("train_episodes", 0)),
            "drl_init_train_seconds": drl_training_time,
            "init_policy_path": getattr(algo, "init_policy_path", None) or "",
        })
    return summary


def _best_metadata_value(solutions, key):
    if not solutions:
        return 0.0
    values = [float(s.metadata.get(key, s.throughput)) for s in solutions]
    return float(max(values)) if values else 0.0


def _init_source_diversity(init_metrics):
    if not init_metrics:
        return 0.0
    counts = {}
    for row in init_metrics:
        counts[row["source_type"]] = counts.get(row["source_type"], 0) + 1
    probs = np.array(list(counts.values()), dtype=float) / max(len(init_metrics), 1)
    return float(1.0 - np.sum(probs ** 2))


def _first_feasible_generation(history):
    for idx, count in enumerate(history.get("feasible_count", [])):
        if count > 0:
            return idx
    return -1


def _save_summary_csv(summaries, path):
    if not summaries:
        return
    extra_fields = []
    for summary in summaries:
        for key in summary.keys():
            if key not in BASE_SUMMARY_FIELDS and key not in extra_fields:
                extra_fields.append(key)
    fieldnames = BASE_SUMMARY_FIELDS + extra_fields
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for summary in summaries:
            writer.writerow(_json_ready(summary))


def _save_recommended_all_csv(summaries, path):
    fields = ["algorithm", "seed", "recommended_coverage", "recommended_rsum", "best_coverage", "best_rsum", "final_HV", "final_FR"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: _json_ready(summary).get(key, "") for key in fields})


def _plot_pareto_compare(pareto_by_algorithm, out_path):
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for algorithm, arrays in pareto_by_algorithm.items():
        pts = np.vstack([a for a in arrays if len(a)]) if any(len(a) for a in arrays) else np.zeros((0, 2))
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1] / 1e6, label=algorithm, s=55)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Rsum (Mbps)")
    ax.set_title("Pareto Comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_history_compare(histories, key, out_path, ylabel):
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for (algorithm, seed), history in histories.items():
        y = history.get(key, [])
        if y:
            ax.plot(range(len(y)), y, label=f"{algorithm}-s{seed}", alpha=0.85)
    ax.set_xlabel("Generation")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} Comparison")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and np.isnan(value):
        return "nan"
    return value


def _display_algorithm_name(algorithm):
    return algorithm.replace("_", "-").upper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default=None, help="Path to experiment YAML")
    parser.add_argument("--algorithm", default=None, help="Single algorithm override")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    if args.experiment:
        config = load_experiment_config(args.experiment)
        if args.algorithm:
            config.setdefault("experiment", {})["algorithms"] = [args.algorithm]
        run_compare_experiment(config)
    else:
        config = load_config(os.path.join(project_root, "configs"))
        run_experiment(config, args.algorithm or "cr_mode", "results")


if __name__ == "__main__":
    main()
