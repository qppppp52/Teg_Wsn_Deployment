#!/usr/bin/env python
"""CR-MODE WSN Deployment Optimization - Main Entry Point"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.io.config_reader import load_config
from src.scene.scenario_builder import Scenario
from src.preprocessing.preprocessor import run_preprocessing
from src.optimizers.cr_mode import CRMode
from src.visualization.plot_pareto import plot_single_pareto, plot_pareto_front
from src.visualization.plot_convergence import plot_convergence, plot_dual_convergence
from src.visualization.plot_deployment import plot_deployment
from src.io.result_io import save_pareto, save_log
from src.physics.analytic_temperature import sample_heat_source_ids
from src.utils.seed import set_seed
from src.utils.logger import get_logger
from tools.plot_selected_pareto_solution import (
    load_data as load_selected_solution_data,
    plot_3d_deployment,
    select_pareto_solution,
)

logger = get_logger("main")


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
    """Pack variable-length neighbor lists into a fixed matrix."""
    max_len = max(len(ns) for ns in neighbor_sets)
    mat = np.full((num_candidates, max_len), -1, dtype=np.int32)
    for idx, neighbors in enumerate(neighbor_sets):
        mat[idx, :len(neighbors)] = neighbors
    return mat


def _save_heat_sources(ctx, seed, path):
    """Persist the current run's heat-source metadata for standalone plots."""
    source_ids = sample_heat_source_ids(
        ctx.candidate_points_full,
        seed=seed,
        config=ctx.config,
    )
    candidate_points = ctx.candidate_points_full
    source_ids = np.asarray(source_ids, dtype=np.int32)

    np.savez_compressed(
        path,
        heat_source_ids=source_ids,
        heat_source_coords=candidate_points[source_ids, :3].astype(float),
        heat_source_faces=candidate_points[source_ids, 3].astype(np.int32),
        heat_source_T=ctx.T_r[source_ids].astype(float),
    )


def _save_selected_solution_figures():
    """Refresh the three standalone selected-solution 3D figures."""
    npz_path = "results/pareto/cr_mode.npz"
    ctx_path = "results/pareto/cr_mode_context.npz"
    pareto_data, ctx_data = load_selected_solution_data(npz_path, ctx_path)
    for mode in ("knee", "highest_coverage", "highest_rsum"):
        sol_idx, mode_label = select_pareto_solution(pareto_data, mode)
        out_name = (
            "results/figures/selected_solution_"
            f"{mode_label.replace('=', '_').replace(' ', '_')}_3d.png"
        )
        plot_3d_deployment(pareto_data, sol_idx, ctx_data, mode_label, out_name)


def run_experiment(config, algorithm="cr_mode"):
    seed = config.get("experiment", {}).get("seeds", [42])[0]
    set_seed(seed)
    logger.info(f"Seed={seed}")

    scenario = Scenario(config)
    scenario.build()
    logger.info(
        f"Scenario: {scenario.num_candidates} candidates, {scenario.num_targets} targets"
    )

    ctx = run_preprocessing(scenario, config, seed)
    index_mapping = ctx.index_mapping
    logger.info(
        f"Ls={index_mapping.num_sensor_candidates}, La={index_mapping.num_ap_candidates}"
    )

    if algorithm == "cr_mode":
        algo = CRMode(ctx, config)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    archive = algo.run()
    feasible_count = len([solution for solution in archive.solutions if solution.feasible])
    logger.info(f"Archive size={len(archive)}, feasible={feasible_count}")

    os.makedirs("results/pareto", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)

    archive.save("results/pareto/cr_mode.npz")
    _save_context(ctx, "results/pareto/cr_mode_context.npz")
    _save_heat_sources(ctx, seed, "results/pareto/cr_mode_heat_sources.npz")

    feasible_objectives = archive.get_feasible_objectives()
    if len(feasible_objectives) > 0:
        plot_single_pareto(
            feasible_objectives,
            "CR-MODE",
            "results/figures/pareto_cr_mode.png",
        )
        logger.info(
            f"Best Cov={feasible_objectives[:, 0].max():.4f}, "
            f"Best Rsum={feasible_objectives[:, 1].max() / 1e6:.1f} Mbps"
        )

        _, uniq_idx = np.unique(
            np.round(feasible_objectives, 6), axis=0, return_index=True
        )
        unique_objs = feasible_objectives[np.sort(uniq_idx)]
        non_dominated = np.ones(len(unique_objs), dtype=bool)
        for i in range(len(unique_objs)):
            for j in range(len(unique_objs)):
                dominates = (
                    i != j
                    and unique_objs[j, 0] >= unique_objs[i, 0]
                    and unique_objs[j, 1] >= unique_objs[i, 1]
                    and (
                        unique_objs[j, 0] > unique_objs[i, 0]
                        or unique_objs[j, 1] > unique_objs[i, 1]
                    )
                )
                if dominates:
                    non_dominated[i] = False
                    break
        pareto = unique_objs[non_dominated]
        pareto = pareto[np.argsort(pareto[:, 0])]
        np.savez("results/pareto/cr_mode_pareto.npz", objectives=pareto)

        import matplotlib as mpl
        mpl.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(
            pareto[:, 0],
            pareto[:, 1] / 1e6,
            c="#2196F3",
            s=100,
            edgecolors="#0D47A1",
            linewidth=1.5,
            zorder=5,
        )
        ax.plot(pareto[:, 0], pareto[:, 1] / 1e6, "--", color="#0D47A1", alpha=0.3)
        for point in pareto:
            ax.annotate(
                f"({point[0]:.3f}, {point[1] / 1e6:.0f})",
                (point[0], point[1] / 1e6),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=7,
                ha="center",
            )
        ax.set_xlabel("Coverage", fontsize=12)
        ax.set_ylabel("Throughput (Mbps)", fontsize=12)
        ax.set_title("CR-MODE Pareto Front", fontsize=13)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("results/figures/pareto_cr_mode_front.png", dpi=200)
        plt.close()
        logger.info(f"Clean Pareto front: {len(pareto)} non-dominated points")

        median_solution = None
        if hasattr(algo, "population") and algo.population and algo.population.solutions:
            solutions = algo.population.solutions
            feasible_solutions = [solution for solution in solutions if solution.feasible]
            if feasible_solutions:
                obj_array = np.array(
                    [[solution.coverage, solution.throughput] for solution in feasible_solutions]
                )
                cov_mid = (obj_array[:, 0].max() + obj_array[:, 0].min()) / 2
                best_idx = np.argmin(np.abs(obj_array[:, 0] - cov_mid))
                if best_idx < len(feasible_solutions):
                    median_solution = feasible_solutions[best_idx]
        if median_solution is not None:
            try:
                plot_deployment(median_solution, ctx, "results/figures/deployment.png")
                logger.info("Deployment figure saved")
            except Exception as exc:
                logger.warning(f"Deployment plot failed: {exc}")
    else:
        logger.warning("No feasible solutions!")

    convergence_history = getattr(algo, "convergence_history", {})
    if len(convergence_history) > 0:
        plot_convergence(
            convergence_history.get("FR_current", []),
            "Feasible Ratio (current gen)",
            "results/figures/convergence_fr.png",
            title="CR-MODE: Feasible Solution Ratio per Generation",
        )
        plot_dual_convergence(
            convergence_history.get("Coverage_feasible", []),
            "Avg Coverage (feasible solutions)",
            convergence_history.get("archive_best_coverage", []),
            "Best Coverage (Pareto Archive)",
            "Coverage",
            "results/figures/convergence_coverage.png",
            title="CR-MODE: Coverage Convergence",
        )
        plot_dual_convergence(
            convergence_history.get("Rsum_feasible_mbps", []),
            "Avg Rsum (feasible solutions, Mbps)",
            convergence_history.get("archive_best_rsum_mbps", []),
            "Best Rsum (Pareto Archive, Mbps)",
            "Throughput (Mbps)",
            "results/figures/convergence_rsum.png",
            title="CR-MODE: Throughput Convergence",
            fmt="%.1f",
        )
        plot_convergence(
            convergence_history.get("CV_mean", []),
            "Mean CV (all solutions)",
            "results/figures/convergence_cv.png",
            title="CR-MODE: Constraint Violation over Generations",
            hline=0.05,
            hline_label="CV=0.05 (feasibility threshold)",
        )
        logger.info("Convergence curves saved")

    try:
        _save_selected_solution_figures()
        logger.info("Selected solution figures saved")
    except Exception as exc:
        logger.warning(f"Selected solution plots failed: {exc}")

    lines = [
        f"Seed: {seed}",
        f"Candidates: {ctx.num_candidates}",
        f"Ls: {index_mapping.num_sensor_candidates}, La={index_mapping.num_ap_candidates}",
        f"Archive size: {len(archive)}",
        f"Feasible: {feasible_count}",
    ]
    save_log(chr(10).join(lines), "results/logs/cr_mode.log")
    logger.info("Experiment complete.")
    return archive


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    config = load_config(os.path.join(project_root, "configs"))
    run_experiment(config, "cr_mode")
