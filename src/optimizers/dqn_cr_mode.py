"""DQN-controlled CR-MODE optimizer."""
from __future__ import annotations

import csv
import os
import numpy as np

from src.optimizers.base_optimizer import BaseOptimizer
from src.model.population import Population
from src.model.pareto_archive import ParetoArchive
from src.evaluator.individual_evaluator import evaluate_individual
from src.dqn.action_space import ACTIONS, get_action, action_dim
from src.dqn.action_mask import build_dqn_action_mask, action_mask_diagnostics
from src.dqn.config import DQNConfig
from src.dqn.state_builder import build_state
from src.dqn.reward_function import compute_reward
from src.evaluation.diversity import objective_space_diversity
from src.evaluation.generation_diagnostics import make_convergence_history, record_generation
from src.constraints.cv_pressure import normalize_cv_components
from src.optimizers.generation_executor import GenerationExecutor, copy_solution_metrics
from src.io.population_snapshot import load_population_snapshot, save_population_snapshot
from src.utils.logger import get_logger

logger = get_logger("DQN-CR-MODE")


class DQNCRMode(BaseOptimizer):
    """CR-MODE where DQN selects one generation-level action."""

    def __init__(self, ctx, config, agent=None, phase=None):
        super().__init__(ctx, config)
        mode_config = config.get("mode", {})
        self.NP = int(mode_config.get("population_size", 60))
        self.Tmax = int(mode_config.get("max_generations", 100))
        self.archive = ParetoArchive(mode_config.get("archive_max_size", 200))
        self.population = None
        self.convergence_history = make_convergence_history()
        self.training_log = []
        self.model_path = None
        self.evaluation_count = 0
        self.executor = GenerationExecutor(ctx, self.NP)
        self.dqn_config = DQNConfig.from_mapping(config)
        self.execution_mode = str(phase or self.dqn_config.mode).lower()
        if self.execution_mode not in {"train", "eval"}:
            raise ValueError("DQN execution phase must be 'train' or 'eval'")

        from src.dqn.dqn_agent import DQNAgent

        state_dim = len(build_state(_EmptyPopulation(), self.archive, ctx))
        if agent is not None:
            self.agent = agent
            if agent.state_dim != state_dim or agent.action_dim != action_dim():
                raise ValueError("Provided DQN agent dimensions are incompatible")
        elif self.execution_mode == "eval":
            checkpoint = self.dqn_config.checkpoint_path
            if not checkpoint or not os.path.isfile(checkpoint):
                raise FileNotFoundError(
                    "DQN evaluation requires a trained checkpoint. "
                    f"Not found: {checkpoint or '<empty path>'}"
                )
            self.agent = DQNAgent.from_checkpoint(checkpoint, config)
            self.agent.set_evaluation_mode()
            self.model_path = checkpoint
        else:
            self.agent = DQNAgent(state_dim, action_dim(), config)

    def initialize_population(self):
        mapping = self.ctx.index_mapping
        snapshot = self.config.get("runtime", {}).get("initial_population_snapshot")
        if snapshot and os.path.isfile(snapshot):
            population, _ = load_population_snapshot(
                snapshot, self.NP, mapping.num_sensor_candidates, mapping.num_ap_candidates
            )
            return population
        population = Population(
            self.NP,
            mapping.num_sensor_candidates,
            mapping.num_ap_candidates,
        )
        population.initialize(
            self.ctx,
            strategy=self.config.get("mode", {}).get("initialization", "mixed"),
        )
        if snapshot:
            seed = self.config.get("experiment", {}).get("seeds", [0])[0]
            save_population_snapshot(population, snapshot, seed)
        return population

    def evaluate_population(self):
        self.population.solutions = []
        for index, individual in enumerate(self.population.individuals):
            solution, repaired = evaluate_individual(individual, self.ctx)
            if repaired is not None:
                individual = repaired
                self.population.individuals[index] = individual
            copy_solution_metrics(individual, solution)
            self.population.solutions.append(solution)
            self.evaluation_count += 1

    def run(self):
        self.population = self.initialize_population()
        self.evaluate_population()
        self.archive.update(self.population.solutions)
        self._record_convergence()
        current_metrics = self._metrics()
        state = build_state(
            self.population,
            self.archive,
            self.ctx,
            {"HV": current_metrics["HV"], "current_HV": current_metrics["HV"]},
            0,
            self.Tmax,
        )
        hv_stall = 0
        mask = self._build_mask(current_metrics, hv_stall)
        self._log_generation(0, None, None, current_metrics)

        for generation in range(1, self.Tmax + 1):
            deterministic = self.execution_mode == "eval"
            action_id = self.agent.select_action(
                state,
                action_mask=mask,
                deterministic=deterministic,
            )
            q_mean, q_max = self.agent.q_stats(state, action_mask=mask)
            action = get_action(action_id)

            result = self.executor.execute(self.population, {
                **action,
                "repair_strategy": action,
            })
            self.population = result.population
            self.evaluation_count += result.evaluations
            self.archive.update(result.trial_solutions)
            self._record_convergence()
            next_metrics = self._metrics()
            next_state = build_state(
                self.population,
                self.archive,
                self.ctx,
                {
                    "HV": current_metrics["HV"],
                    "current_HV": next_metrics["HV"],
                },
                generation,
                self.Tmax,
            )
            improved = next_metrics["HV"] > current_metrics["HV"] + 1.0e-12
            hv_stall = 0 if improved else hv_stall + 1
            next_mask = self._build_mask(next_metrics, hv_stall)
            reward, parts = compute_reward(
                current_metrics,
                next_metrics,
                self.config.get("constraints", {}).get("max_repair_iter", 5),
            )
            loss = None
            if self.execution_mode == "train":
                self.agent.observe(
                    state,
                    action_id,
                    reward,
                    next_state,
                    generation == self.Tmax,
                    mask,
                    next_mask,
                )
                for _ in range(self.dqn_config.updates_per_step):
                    step_loss = self.agent.train_step()
                    if step_loss is not None:
                        loss = step_loss

            mask_diag = action_mask_diagnostics(mask, ACTIONS, current_metrics)
            row = {
                "generation": generation,
                "action_id": action_id,
                "action_name": action["name"],
                "reward": reward,
                **parts,
                "epsilon": 0.0 if deterministic else self.agent.epsilon(),
                "loss": loss if loss is not None else np.nan,
                "q_mean": q_mean,
                "q_max": q_max,
                "target_updated": bool(self.agent.target_updated),
                "action_mask_used": self.dqn_config.action_mask_enabled,
                "num_allowed_actions": mask_diag["num_allowed_actions"],
                "allowed_action_names": mask_diag["allowed_action_names"],
                "dominant_pressure": mask_diag["dominant_pressure"],
                "FR_after_repair": next_metrics["FR"],
                "CV_after_repair": next_metrics["CV_mean"],
                "HV": next_metrics["HV"],
                "coverage_best": next_metrics["best_coverage"],
                "rsum_capacity_best": self.convergence_history["rsum_capacity_best"][-1],
                "archive_size": len(self.archive),
                "pareto_count": self.convergence_history["pareto_count"][-1],
                "evaluation_count": self.evaluation_count,
            }
            self.training_log.append(row)

            state = next_state
            mask = next_mask
            current_metrics = next_metrics
            if generation % 10 == 0 or generation == self.Tmax:
                self._log_generation(generation, action, reward, current_metrics)

        self._save_training_log()
        self._save_model()
        return self.archive

    def _record_convergence(self):
        record_generation(
            self.convergence_history,
            self.population,
            self.archive,
            self.config,
        )

    def _build_mask(self, metrics, hv_stall):
        mask_metrics = dict(metrics)
        mask_metrics["hv_stall_generations"] = hv_stall
        return build_dqn_action_mask(mask_metrics, ACTIONS, self.config)

    def _metrics(self):
        solutions = list(self.population.solutions or [])
        feasible = [solution for solution in solutions if solution.feasible]
        pressures = [normalize_cv_components(solution, self.ctx) for solution in solutions]
        if pressures:
            pressure = {
                key: float(np.mean([item[key] for item in pressures]))
                for key in pressures[0]
            }
        else:
            pressure = {
                key: 0.0
                for key in ["deploy", "link", "capacity", "energy", "sink", "service", "total"]
            }
        rsum_max = float(self.config.get("evaluation", {}).get("rsum_ref_max", 2.0e8))
        return {
            "CV_mean": float(np.mean([solution.cv for solution in solutions])) if solutions else 0.0,
            "FR": len(feasible) / len(solutions) if solutions else 0.0,
            "HV": self.convergence_history["HV"][-1] if self.convergence_history["HV"] else 0.0,
            "best_coverage": max([solution.coverage for solution in feasible], default=0.0),
            "best_rsum_norm": max(
                [float(solution.rsum_capacity) for solution in feasible],
                default=0.0,
            ) / max(rsum_max, 1.0),
            "diversity": objective_space_diversity(solutions),
            "pressure": pressure,
            "mean_repair_iter": float(np.mean([
                getattr(solution, "repair_iter", 0) for solution in solutions
            ])) if solutions else 0.0,
        }

    def _log_generation(self, generation, action, reward, metrics):
        action_name = action["name"] if action else "initial"
        reward_text = f"{reward:.3f}" if reward is not None else "n/a"
        logger.info(
            f"Gen {generation:4d} | action={action_name} reward={reward_text} "
            f"FR={metrics['FR']:.3f} CV={metrics['CV_mean']:.4f} "
            f"Archive={len(self.archive)} Evals={self.evaluation_count}"
        )

    def _save_training_log(self):
        if not self.training_log:
            return
        base = self.config.get("runtime", {}).get("output_dir", "results")
        os.makedirs(os.path.join(base, "data"), exist_ok=True)
        path = os.path.join(base, "data", "dqn_training_log.csv")
        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(self.training_log[0].keys()))
            writer.writeheader()
            writer.writerows(self.training_log)

    def _save_model(self):
        if self.execution_mode != "train" or not self.dqn_config.save_model:
            return
        base = self.config.get("runtime", {}).get("output_dir", "results")
        model_dir = os.path.join(base, "models")
        os.makedirs(model_dir, exist_ok=True)
        self.model_path = os.path.join(model_dir, "dqn_q_network.pt")
        self.agent.save(self.model_path)


class _EmptyPopulation:
    solutions = []
