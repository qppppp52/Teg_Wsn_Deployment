"""DQN-controlled CR-MODE optimizer."""
from __future__ import annotations

import csv
import json
import os
import numpy as np

from src.constraints.cv_pressure import aggregate_population_pressure
from src.constraints.cv_schema import CV_COMPONENT_KEYS
from src.dqn.action_mask import build_dqn_action_mask_details
from src.dqn.action_space import ACTIONS, action_dim, audit_action_space, get_action
from src.dqn.config import DQNConfig
from src.dqn.reward_function import compute_reward
from src.dqn.state_builder import build_state
from src.evaluation.diversity import objective_space_diversity
from src.evaluation.generation_diagnostics import make_convergence_history, record_generation
from src.evaluator.individual_evaluator import evaluate_individual, configured_max_repair_iter
from src.io.population_snapshot import load_population_snapshot, save_population_snapshot
from src.model.pareto_archive import ParetoArchive
from src.model.population import Population
from src.optimizers.base_optimizer import BaseOptimizer
from src.optimizers.generation_executor import GenerationExecutor, copy_solution_metrics
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
        self.checkpoint_loaded = False
        self.dqn_validity_report = {}
        self._evaluation_evidence_before = None
        self.executor = GenerationExecutor(ctx, self.NP)
        self.max_repair_iter = configured_max_repair_iter(ctx)
        self.dqn_config = DQNConfig.from_mapping(config)
        self.action_audit = audit_action_space()
        for audit_row in self.action_audit:
            logger.info(
                "Action contract | id=%s name=%s signature=%s aliases=%s",
                audit_row["action_id"],
                audit_row["action_name"],
                audit_row["effective_signature"],
                audit_row["aliased_with"],
            )
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
            self.checkpoint_loaded = True
        else:
            self.agent = DQNAgent(state_dim, action_dim(), config)

        if self.execution_mode == "eval":
            self._evaluation_evidence_before = self._agent_evidence()

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
            solution, repaired = evaluate_individual(
                individual,
                self.ctx,
                max_repair_iter=self.max_repair_iter,
            )
            if repaired is not None:
                individual = repaired
                self.population.individuals[index] = individual
            copy_solution_metrics(individual, solution)
            self.population.solutions.append(solution)
            self.evaluation_count += 1
        self._record_boost_statistics(self.population.solutions)

    def run(self):
        self.population = self.initialize_population()
        self.evaluate_population()
        self.archive.update(self.population.solutions)
        self._save_pressure_contract()
        self._record_convergence()
        current_metrics = self._metrics()
        state = build_state(
            self.population,
            self.archive,
            self.ctx,
            {
                "HV": current_metrics["HV"],
                "current_HV": current_metrics["HV"],
                "population_pressure": current_metrics["population_pressure"],
            },
            0,
            self.Tmax,
        )
        hv_stall = 0
        mask, mask_diag = self._build_mask(current_metrics, hv_stall)
        mask_delta_hv = None
        self._log_generation(0, None, None, current_metrics)

        for generation in range(1, self.Tmax + 1):
            current_mask_hv_stall = hv_stall
            current_mask_delta_hv = mask_delta_hv
            deterministic = self.execution_mode == "eval"
            epsilon_used = 0.0 if deterministic else self.agent.epsilon()
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
            self._record_boost_statistics(result.trial_solutions)
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
                    "population_pressure": next_metrics["population_pressure"],
                },
                generation,
                self.Tmax,
            )
            delta_hv = next_metrics["HV"] - current_metrics["HV"]
            improved = delta_hv > 1.0e-12
            hv_stall = 0 if improved else hv_stall + 1
            next_mask, next_mask_diag = self._build_mask(next_metrics, hv_stall, delta_hv)
            reward, parts = compute_reward(
                current_metrics,
                next_metrics,
                self.max_repair_iter,
                self.dqn_config.reward_clip,
                self.dqn_config.pressure_regression_tolerance,
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

            row = {
                "generation": generation,
                "action_id": action_id,
                "action_name": action["name"],
                "reward": reward,
                **parts,
                "epsilon": epsilon_used,
                "loss": loss if loss is not None else np.nan,
                "q_mean": q_mean,
                "q_max": q_max,
                "target_updated": bool(self.agent.target_updated),
                "action_mask_used": self.dqn_config.action_mask_enabled,
                "num_allowed_actions": mask_diag["num_allowed_actions"],
                "allowed_action_names": mask_diag["allowed_action_names"],
                "dominant_pressure": mask_diag["dominant_pressure"],
                "dominant_pressure_value": mask_diag["dominant_pressure_value"],
                "second_pressure_value": mask_diag["second_pressure_value"],
                "dominant_pressure_margin": mask_diag["dominant_pressure_margin"],
                "dominant_pressure_clear": mask_diag["dominant_pressure_clear"],
                "mask_fallback_used": mask_diag["mask_fallback_used"],
                "mask_reason": mask_diag["mask_reason"],
                "delta_HV_for_mask": current_mask_delta_hv,
                "hv_stall_generations_for_mask": current_mask_hv_stall,
                "action_was_allowed": bool(mask[action_id]),
                "FR_after_repair": next_metrics["FR"],
                "CV_after_repair": next_metrics["CV_mean"],
                "CV_min_after_repair": next_metrics["CV_min"],
                "HV": next_metrics["HV"],
                "coverage_best": next_metrics["best_coverage"],
                "rsum_capacity_best": self.convergence_history["rsum_capacity_best"][-1],
                "archive_size": len(self.archive),
                "pareto_count": self.convergence_history["pareto_count"][-1],
                "evaluation_count": self.evaluation_count,
                "pressure_schema_version": self.dqn_config.pressure_schema_version,
                "pressure_reference_source": self.dqn_config.pressure_reference_source,
                **{
                    f"cv_{key}_mean": next_metrics["raw_cv"][key]
                    for key in CV_COMPONENT_KEYS
                },
                **{
                    f"p_{key}_mean": next_metrics["pressure"][key]
                    for key in CV_COMPONENT_KEYS
                },
                **{
                    f"b_{key}_rate": next_metrics["pressure_violation_rate"][key]
                    for key in CV_COMPONENT_KEYS
                },
                **{
                    f"p_{key}_saturation_rate": next_metrics["pressure_saturation_rate"][key]
                    for key in CV_COMPONENT_KEYS
                },
                "next_dominant_pressure": next_mask_diag["dominant_pressure"],
                "next_dominant_pressure_clear": next_mask_diag["dominant_pressure_clear"],
            }
            self.training_log.append(row)

            state = next_state
            mask = next_mask
            mask_diag = next_mask_diag
            mask_delta_hv = delta_hv
            current_metrics = next_metrics
            if generation % 10 == 0 or generation == self.Tmax:
                self._log_generation(generation, action, reward, current_metrics)

        self._save_training_log()
        self._save_model()
        self._finalize_dqn_validity()
        return self.archive

    def _agent_evidence(self):
        return {
            "parameter_hash": self.agent.parameter_hash(),
            "gradient_steps": int(self.agent.gradient_steps),
            "interaction_steps": int(self.agent.interaction_steps),
            "replay_size": len(self.agent.replay),
            "state_normalizer_count": int(self.agent.state_normalizer.count),
            "reward_normalizer_count": int(self.agent.reward_normalizer.count),
        }

    def _finalize_dqn_validity(self):
        if self.execution_mode != "eval":
            return
        before = dict(self._evaluation_evidence_before or self._agent_evidence())
        after = self._agent_evidence()
        unchanged = all(before[key] == after[key] for key in before)
        self.dqn_validity_report = {
            "DQN_VALID": bool(
                self.checkpoint_loaded
                and unchanged
                and self.agent.state_normalizer.frozen
                and self.agent.reward_normalizer.frozen
            ),
            "checkpoint_loaded": bool(self.checkpoint_loaded),
            "checkpoint_path": os.path.abspath(self.model_path) if self.model_path else "",
            "execution_mode": self.execution_mode,
            "epsilon": 0.0,
            "collect_experience": False,
            "parameter_hash_before": before["parameter_hash"],
            "parameter_hash_after": after["parameter_hash"],
            "gradient_steps_before": before["gradient_steps"],
            "gradient_steps_after": after["gradient_steps"],
            "interaction_steps_before": before["interaction_steps"],
            "interaction_steps_after": after["interaction_steps"],
            "replay_size_before": before["replay_size"],
            "replay_size_after": after["replay_size"],
            "state_normalizer_count_before": before["state_normalizer_count"],
            "state_normalizer_count_after": after["state_normalizer_count"],
            "reward_normalizer_count_before": before["reward_normalizer_count"],
            "reward_normalizer_count_after": after["reward_normalizer_count"],
            "state_normalizer_frozen": bool(self.agent.state_normalizer.frozen),
            "reward_normalizer_frozen": bool(self.agent.reward_normalizer.frozen),
        }
        dqn_config = getattr(self, "dqn_config", None)
        if dqn_config is not None:
            self.dqn_validity_report.update({
                "pressure_schema_version": dqn_config.pressure_schema_version,
                "pressure_refs": dict(dqn_config.pressure_refs),
                "pressure_reference_source": dqn_config.pressure_reference_source,
            })
        if self.checkpoint_loaded:
            base = self.config.get("runtime", {}).get("output_dir", "results")
            path = os.path.join(base, "data", "dqn_validity_report.json")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as file:
                json.dump(self.dqn_validity_report, file, indent=2)

    def _record_convergence(self):
        record_generation(
            self.convergence_history,
            self.population,
            self.archive,
            self.config,
        )

    def _build_mask(self, metrics, hv_stall, delta_hv=None):
        mask_metrics = dict(metrics)
        mask_metrics["hv_stall_generations"] = hv_stall
        if delta_hv is not None:
            mask_metrics["delta_HV"] = float(delta_hv)
        return build_dqn_action_mask_details(mask_metrics, ACTIONS, self.config)

    def _metrics(self):
        solutions = list(self.population.solutions or [])
        feasible = [solution for solution in solutions if solution.feasible]
        summary = aggregate_population_pressure(solutions, self.ctx)
        rsum_max = float(self.config.get("evaluation", {}).get("rsum_ref_max", 2.0e8))
        best_rsum_capacity = max(
            [float(solution.rsum_capacity) for solution in feasible],
            default=0.0,
        )
        cv_values = [float(solution.cv) for solution in solutions]
        return {
            "CV_mean": float(np.mean(cv_values)) if cv_values else 0.0,
            "CV_min": float(np.min(cv_values)) if cv_values else 0.0,
            "FR": len(feasible) / len(solutions) if solutions else 0.0,
            "HV": self.convergence_history["HV"][-1] if self.convergence_history["HV"] else 0.0,
            "best_coverage": max([solution.coverage for solution in feasible], default=0.0),
            "best_rsum_capacity": best_rsum_capacity,
            "best_rsum_norm": best_rsum_capacity / max(rsum_max, 1.0),
            "energy_cv_sensor": float(np.mean([solution.cv_energy_sensor for solution in solutions])) if solutions else 0.0,
            "energy_cv_ap": float(np.mean([solution.cv_energy_ap for solution in solutions])) if solutions else 0.0,
            "diversity": objective_space_diversity(solutions),
            "raw_cv": dict(summary.mean_raw_cv),
            "pressure": dict(summary.mean_pressure),
            "pressure_violation_rate": dict(summary.violation_rate),
            "pressure_saturation_rate": dict(summary.saturation_rate),
            "population_pressure": summary,
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
            f"Cov={metrics['best_coverage']:.3f} "
            f"Rsum={metrics['best_rsum_capacity']:.1f} "
            f"Archive={len(self.archive)} Evals={self.evaluation_count}"
        )

    def _save_pressure_contract(self):
        """Persist the fixed DQN-only scale used by this run."""
        runtime = self.config.get("runtime", {})
        base = runtime.get("output_dir", "results")
        path = os.path.join(base, "data", "dqn_pressure_contract.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "cv_component_keys": list(CV_COMPONENT_KEYS),
            "pressure_schema_version": self.dqn_config.pressure_schema_version,
            "pressure_refs": dict(self.dqn_config.pressure_refs),
            "cv_zero_tol": self.dqn_config.pressure_cv_zero_tol,
            "reference_source": self.dqn_config.pressure_reference_source,
            "state_cv_total_ref": self.dqn_config.state_cv_total_ref,
            "action_mask_thresholds": dict(self.dqn_config.mask_thresholds),
        }
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True)

    def _save_training_log(self):
        if not self.training_log:
            return
        runtime = self.config.get("runtime", {})
        base = runtime.get("output_dir", "results")
        default_name = (
            "dqn_evaluation_action_log.csv"
            if self.execution_mode == "eval"
            else "dqn_training_action_log.csv"
        )
        path = runtime.get("dqn_action_log_path") or os.path.join(base, "data", default_name)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
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
