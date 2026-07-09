"""DQN-controlled CR-MODE optimizer."""
from __future__ import annotations

import csv
import os
import numpy as np
from src.optimizers.base_optimizer import BaseOptimizer
from src.model.population import Population
from src.model.pareto_archive import ParetoArchive
from src.operators.mode_mutation import mutate
from src.operators.mode_crossover import crossover
from src.operators.mode_selection import select_better
from src.evaluator.individual_evaluator import evaluate_individual
from src.dqn.action_space import ACTIONS, get_action, action_dim
from src.dqn.action_mask import build_dqn_action_mask, action_mask_diagnostics
from src.dqn.state_builder import build_state
from src.dqn.reward_function import compute_reward
from src.evaluation.diversity import objective_space_diversity
from src.evaluation.generation_diagnostics import make_convergence_history, record_generation
from src.constraints.cv_pressure import normalize_cv_components
from src.utils.logger import get_logger

logger = get_logger("DQN-CR-MODE")


class DQNCRMode(BaseOptimizer):
    """CR-MODE where DQN chooses generation-level search/repair actions."""

    def __init__(self, ctx, config):
        super().__init__(ctx, config)
        mc = config.get("mode", {})
        self.NP = mc.get("population_size", 60)
        self.Tmax = mc.get("max_generations", 100)
        self.archive = ParetoArchive(mc.get("archive_max_size", 200))
        self.population = None
        self.convergence_history = make_convergence_history()

        self.training_log = []
        self.model_path = None
        self.agent = None
        try:
            from src.dqn.dqn_agent import DQNAgent
            state_dim = len(build_state(_EmptyPopulation(), self.archive, ctx))
            self.agent = DQNAgent(state_dim, action_dim(), config)
        except Exception as exc:
            logger.warning(f"DQNAgent unavailable; using masked stochastic action policy: {exc}")

    def run(self):
        im = self.ctx.index_mapping
        self.population = Population(self.NP, im.num_sensor_candidates, im.num_ap_candidates)
        self.population.initialize(self.ctx, strategy=self.config.get("mode", {}).get("initialization", "mixed"))
        self.population.solutions = []
        for idx, ind in enumerate(self.population.individuals):
            sol, rep_ind = evaluate_individual(ind, self.ctx)
            if rep_ind is not None:
                ind = rep_ind
                self.population.individuals[idx] = ind
            _copy_solution_metrics(ind, sol)
            self.population.solutions.append(sol)
        self.archive.update(self.population.solutions)
        self._record_convergence(0)

        prev_metrics = self._metrics()
        hv_stall = 0
        prev_hv = prev_metrics["HV"]
        for gen in range(self.Tmax):
            state = build_state(
                self.population, self.archive, self.ctx,
                {"HV": self.convergence_history["HV"][-1], "current_HV": self.convergence_history["HV"][-1]},
                gen, self.Tmax,
            )
            mask_metrics = dict(prev_metrics)
            mask_metrics["hv_stall_generations"] = hv_stall
            mask = build_dqn_action_mask(mask_metrics, ACTIONS, self.config)
            mask_diag = action_mask_diagnostics(mask, ACTIONS, mask_metrics)
            if self.agent is not None:
                action_id = self.agent.select_action(state, gen, action_mask=mask)
                q_mean, q_max = self.agent.q_stats(state, action_mask=mask)
            else:
                choices = np.flatnonzero(mask)
                action_id = int(np.random.choice(choices if len(choices) else np.arange(action_dim())))
                q_mean, q_max = np.nan, np.nan
            action = get_action(action_id)

            new_inds, new_sols = [], []
            for i, ind in enumerate(self.population.individuals):
                mutant = mutate(ind, action["F"], action["mutation_strategy"], self.population.individuals, i, self.ctx)
                trial = crossover(ind, mutant, action["CR"])
                trial_sol, rep_ind = evaluate_individual(trial, self.ctx, repair_strategy=action)
                if rep_ind is not None:
                    trial = rep_ind
                winner, winner_sol, _ = select_better(trial, trial_sol, ind, self.population.solutions[i])
                _copy_solution_metrics(winner, winner_sol)
                new_inds.append(winner)
                new_sols.append(winner_sol)

            self.population.individuals = new_inds
            self.population.solutions = new_sols
            self.archive.update(new_sols)
            self._record_convergence(gen + 1)
            metrics = self._metrics()
            next_state = build_state(
                self.population, self.archive, self.ctx,
                {"HV": prev_metrics["HV"], "current_HV": metrics["HV"]},
                gen + 1, self.Tmax,
            )
            reward, parts = compute_reward(prev_metrics, metrics, self.config.get("constraints", {}).get("max_repair_iter", 5))
            loss = None
            if self.agent is not None:
                self.agent.replay.add(state, action_id, reward, next_state, gen == self.Tmax - 1)
                loss = self.agent.train_step()

            row = {
                "generation": gen + 1,
                "action_id": action_id,
                "action_name": action["name"],
                "reward": reward,
            }
            row.update(parts)
            row.update({
                "epsilon": self.agent.epsilon(gen) if self.agent is not None else np.nan,
                "loss": loss if loss is not None else np.nan,
                "q_mean": q_mean,
                "q_max": q_max,
                "target_updated": bool(getattr(self.agent, "target_updated", False)) if self.agent is not None else False,
                "action_mask_used": bool(self.config.get("dqn", {}).get("action_mask_enabled", True)),
                "num_allowed_actions": mask_diag["num_allowed_actions"],
                "allowed_action_names": mask_diag["allowed_action_names"],
                "dominant_pressure": mask_diag["dominant_pressure"],
                "FR_after_repair": metrics["FR"],
                "CV_after_repair": metrics["CV_mean"],
                "HV": metrics["HV"],
                "coverage_best": metrics["best_coverage"],
                "rsum_actual_best": self.convergence_history["rsum_actual_best"][-1] if self.convergence_history["rsum_actual_best"] else 0.0,
                "rsum_capacity_best": self.convergence_history["rsum_capacity_best"][-1] if self.convergence_history["rsum_capacity_best"] else 0.0,
                "archive_size": len(self.archive),
                "pareto_count": self.convergence_history["pareto_count"][-1] if self.convergence_history["pareto_count"] else 0,
            })
            self.training_log.append(row)

            hv_stall = hv_stall + 1 if metrics["HV"] <= prev_hv + 1.0e-12 else 0
            prev_hv = metrics["HV"]
            prev_metrics = metrics
            if gen % 10 == 0 or gen == self.Tmax - 1:
                logger.info(
                    f"Gen {gen:4d} | action={action['name']} | reward={reward:.3f} "
                    f"| FR={metrics['FR']:.3f} | CV={metrics['CV_mean']:.4f} | Archive={len(self.archive)}"
                )
        self._save_training_log()
        self._save_model()
        return self.archive

    def _record_convergence(self, gen):
        metrics = record_generation(self.convergence_history, self.population, self.archive, self.config)
        if metrics.get("saturated_link_ratio", 0.0) > 0.9:
            logger.warning("Throughput actual is saturated; Pareto front may degenerate.")

    def _metrics(self):
        sols = self.population.solutions
        feasible = [s for s in sols if s.feasible]
        cv_mean = float(np.mean([s.cv for s in sols])) if sols else 0.0
        pressures = [normalize_cv_components(s, self.ctx) for s in sols] or [
            {k: 0.0 for k in ["deploy", "link", "capacity", "energy", "sink", "service", "total"]}
        ]
        pressure = {k: float(np.mean([p[k] for p in pressures])) for k in pressures[0]}
        rmax = float(self.config.get("evaluation", {}).get("rsum_ref_max", 1.0e9))
        return {
            "CV_mean": cv_mean,
            "FR": len(feasible) / len(sols) if sols else 0.0,
            "HV": self.convergence_history["HV"][-1] if self.convergence_history["HV"] else 0.0,
            "best_coverage": max([s.coverage for s in feasible], default=0.0),
            "best_rsum_norm": max([s.throughput for s in feasible], default=0.0) / (rmax + 1e-12),
            "diversity": objective_space_diversity(sols),
            "pressure": pressure,
            "mean_repair_iter": float(np.mean([getattr(s, "repair_iter", 0) for s in sols])) if sols else 0.0,
        }

    def _save_training_log(self):
        if not self.training_log:
            return
        base = self.config.get("runtime", {}).get("output_dir", "results")
        os.makedirs(os.path.join(base, "data"), exist_ok=True)
        path = os.path.join(base, "data", "dqn_training_log.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.training_log[0].keys()))
            writer.writeheader()
            writer.writerows(self.training_log)

    def _save_model(self):
        """Persist the DQN model checkpoint for reproducibility."""
        if self.agent is None or not self.config.get("dqn", {}).get("save_model", True):
            return
        base = self.config.get("runtime", {}).get("output_dir", "results")
        model_dir = os.path.join(base, "models")
        os.makedirs(model_dir, exist_ok=True)
        self.model_path = os.path.join(model_dir, "dqn_q_network.pt")
        self.agent.save(self.model_path)


class _EmptyPopulation:
    solutions = []


def _copy_solution_metrics(individual, solution):
    individual.coverage = solution.coverage
    individual.throughput = solution.throughput
    individual.cv = solution.cv
    individual.feasible = solution.feasible
