"""Population generator for PPO-initialized CR-MODE."""
from __future__ import annotations

import json
import os
import time
import numpy as np
from src.model.individual import create_random_individual
from src.model.population import Population
from src.rl_init.env import InitDeploymentEnv
from src.rl_init.heuristic_initializer import create_composite_heuristic_individual
from src.rl_init.init_evaluator import evaluate_init_individual, get_selected_rsum
from src.rl_init.diversity_filter import filter_population_by_diversity, priority_l2_distance, hamming_distance_deployment
from src.rl_init.logger import save_rows
from src.rl_init.checkpoint import save_checkpoint, load_checkpoint
from src.utils.logger import get_logger

logger = get_logger("DRLInitPopulationGenerator")


TRAINING_LOG_COLUMNS = [
    "episode", "step", "episode_steps", "episode_reward", "total_reward", "step_reward_sum",
    "terminal_reward", "feasible", "cv", "cv_deploy", "cv_link", "cv_capacity", "cv_energy",
    "cv_sink", "cv_service", "coverage", "rsum", "rsum_actual", "rsum_capacity", "repair_iter",
    "repair_success", "num_sensors", "num_aps", "invalid_action_count", "loss", "policy_loss",
    "value_loss", "entropy", "approx_kl", "loaded_checkpoint", "fallback_reason", "policy_source",
]


def _training_log_placeholder(loaded_checkpoint=False, fallback_reason="", policy_source="trained"):
    row = {key: np.nan for key in TRAINING_LOG_COLUMNS}
    row.update({
        "episode": -1,
        "step": 0,
        "episode_steps": 0,
        "episode_reward": 0.0,
        "total_reward": 0.0,
        "step_reward_sum": 0.0,
        "loaded_checkpoint": loaded_checkpoint,
        "fallback_reason": fallback_reason,
        "policy_source": policy_source,
    })
    return row


class DRLInitPopulationGenerator:
    def __init__(self, ctx, config):
        self.ctx = ctx
        self.config = config
        self.cfg = config.get("drl_init", config)
        self.rng = np.random.default_rng(int(self.cfg.get("seed", 42)))
        self.trainer = None
        self.training_log = []
        self.train_seconds = 0.0
        self.init_metrics = []
        self.accepted_counts = {"drl": 0, "heuristic": 0, "random": 0}
        self.policy_path = ""
        self.loaded_checkpoint = False
        self.policy_source = "trained"
        self.torch_available = False
        self.summary = {}
        self.generation_seconds = 0.0

    def train(self):
        if self.trainer is not None:
            return self.training_log
        try:
            from src.rl_init.ppo_trainer import PPOTrainer
            self.torch_available = True
            self.trainer = PPOTrainer(self.ctx, self.config)
            checkpoint_path = self.cfg.get("checkpoint_path", "experiments/checkpoints/drl_init_policy.pt")
            if (not self.cfg.get("force_retrain", False)) and self.cfg.get("load_checkpoint_if_exists", True) and load_checkpoint(self.trainer.agent, checkpoint_path):
                self.loaded_checkpoint = True
                self.policy_source = "loaded_checkpoint"
                self.policy_path = checkpoint_path
                self.training_log = [_training_log_placeholder(True, "", "loaded_checkpoint")]
                self.train_seconds = 0.0
                return self.training_log
            self.training_log = self.trainer.train()
            self.loaded_checkpoint = False
            self.policy_source = "trained"
            self.train_seconds = self.trainer.train_seconds
            for row in self.training_log:
                row.setdefault("loaded_checkpoint", False)
                row.setdefault("fallback_reason", "")
                row.setdefault("policy_source", self.policy_source)
            if self.cfg.get("save_checkpoint", True):
                self.policy_path = save_checkpoint(self.trainer.agent, checkpoint_path)
        except ImportError as exc:
            logger.warning(f"PyTorch unavailable; DRL initializer will use heuristic/random fallback: {exc}")
            self.training_log = [_training_log_placeholder(False, "torch_unavailable", "fallback")]
            self.policy_source = "fallback"
            self.torch_available = False
            self.train_seconds = 0.0
        return self.training_log

    def generate(self, population_size):
        start = time.time()
        population_size = int(population_size)
        self.train()
        drl_target = int(round(population_size * float(self.cfg.get("drl_ratio", self.cfg.get("generated_ratio", 0.6)))))
        heuristic_target = int(round(population_size * float(self.cfg.get("heuristic_ratio", 0.2))))
        random_target = max(0, population_size - drl_target - heuristic_target)
        if self.trainer is None:
            heuristic_target += drl_target
            drl_target = 0

        candidates = []
        for _ in range(max(drl_target * 2, drl_target)):
            if len([c for c in candidates if c["source_type"] == "drl"]) >= drl_target:
                break
            ind = self.generate_one(deterministic=False)
            candidates.append(evaluate_init_individual(ind, self.ctx, self.config, "drl"))

        min_dist = float(self.cfg.get("diversity", {}).get("min_priority_l2_distance", 0.05))
        accepted = filter_population_by_diversity(candidates, min_dist, drl_target)

        individuals = [item["individual"] for item in accepted]
        sources = [item["source_type"] for item in accepted]
        for _ in range(heuristic_target):
            ind = create_composite_heuristic_individual(self.ctx, self.config, self.rng)
            item = evaluate_init_individual(ind, self.ctx, self.config, "heuristic")
            individuals.append(item["individual"])
            sources.append("heuristic")
            accepted.append(item)
        im = self.ctx.index_mapping
        for _ in range(random_target):
            ind = create_random_individual(im.num_sensor_candidates, im.num_ap_candidates)
            item = evaluate_init_individual(ind, self.ctx, self.config, "random")
            individuals.append(item["individual"])
            sources.append("random")
            accepted.append(item)

        while len(individuals) < population_size:
            ind = create_random_individual(im.num_sensor_candidates, im.num_ap_candidates)
            item = evaluate_init_individual(ind, self.ctx, self.config, "random")
            individuals.append(item["individual"])
            sources.append("random")
            accepted.append(item)

        population = Population(population_size, im.num_sensor_candidates, im.num_ap_candidates)
        population.individuals = individuals[:population_size]
        self.accepted_counts = {
            "drl": sources[:population_size].count("drl"),
            "heuristic": sources[:population_size].count("heuristic"),
            "random": sources[:population_size].count("random"),
        }
        self.init_metrics = []
        accepted_final = accepted[:population_size]
        for idx, item in enumerate(accepted_final):
            sol = item["solution"]
            prev_items = accepted_final[:idx]
            priority_distances = [priority_l2_distance(item["individual"], prev["individual"]) for prev in prev_items]
            hamming_distances = [hamming_distance_deployment(sol, prev["solution"]) for prev in prev_items if prev.get("solution") is not None]
            priority_dist = float(min(priority_distances)) if priority_distances else float("nan")
            hamming_dist = float(min(hamming_distances)) if hamming_distances else float("nan")
            diversity_score = float(0.5 * priority_dist + 0.5 * hamming_dist) if priority_distances and hamming_distances else float("nan")
            self.init_metrics.append({
                "individual_id": idx,
                "source_type": item["source_type"],
                "quality_score": item["quality_score"],
                "feasible_before_repair": getattr(sol, "feasible_before_repair", False),
                "cv_before_repair": getattr(sol, "cv_before_repair", sol.cv),
                "feasible": sol.feasible,
                "cv": sol.cv,
                "cv_deploy": getattr(sol, "cv_deploy", 0.0),
                "cv_link": getattr(sol, "cv_link", 0.0),
                "cv_capacity": getattr(sol, "cv_capacity", 0.0),
                "cv_energy": getattr(sol, "cv_energy", 0.0),
                "cv_sink": getattr(sol, "cv_sink", 0.0),
                "cv_service": getattr(sol, "cv_service", 0.0),
                "coverage": sol.coverage,
                "rsum": get_selected_rsum(sol, self.config),
                "rsum_actual": sol.metadata.get("throughput_actual", sol.throughput),
                "rsum_capacity": sol.metadata.get("throughput_capacity", sol.throughput),
                "repair_iter": getattr(sol, "repair_iter", 0),
                "repair_success": getattr(sol, "repair_success", bool(sol.feasible)),
                "active_sensors": int(np.sum(sol.x)),
                "active_aps": int(np.sum(sol.y)),
                "diversity_score": diversity_score,
                "priority_l2_distance_to_prev": priority_dist,
                "deployment_hamming_distance_to_prev": hamming_dist,
            })
        self.generation_seconds = time.time() - start
        return population

    def generate_one(self, deterministic=False):
        if self.trainer is None:
            return create_composite_heuristic_individual(self.ctx, self.config, self.rng)
        env = InitDeploymentEnv(self.ctx, self.config)
        state = env.reset(seed=int(self.rng.integers(0, 2**31 - 1)))
        done = False
        while not done:
            action, _, _, _ = self.trainer.agent.act(state, deterministic=deterministic)
            state, _, done, _ = env.step(action)
        return env.build_current_individual()

    def save_artifacts(self, output_dir):
        data_dir = os.path.join(output_dir, "data")
        save_rows(self.training_log, os.path.join(data_dir, "drl_init_training_log.csv"))
        save_rows(self.init_metrics, os.path.join(data_dir, "init_population_metrics.csv"))
        os.makedirs(data_dir, exist_ok=True)
        feasible_before = [bool(row.get("feasible_before_repair", False)) for row in self.init_metrics]
        feasible_after = [bool(row.get("feasible", False)) for row in self.init_metrics]
        cv_before = [float(row.get("cv_before_repair", row.get("cv", 0.0))) for row in self.init_metrics]
        cv_after = [float(row.get("cv", 0.0)) for row in self.init_metrics]
        finite_diversity = [float(row.get("diversity_score", 0.0)) for row in self.init_metrics if np.isfinite(float(row.get("diversity_score", 0.0)))]
        summary = {
            "algorithm": "drl_init_cr_mode",
            "policy_algorithm": "ppo",
            "loaded_checkpoint": self.loaded_checkpoint,
            "checkpoint_path": self.policy_path,
            "policy_source": self.policy_source,
            "torch_available": self.torch_available,
            "train_episodes": int(self.cfg.get("train_episodes", 0)),
            "drl_training_time": self.train_seconds,
            "generation_time": self.generation_seconds,
            "accepted_drl_count": self.accepted_counts.get("drl", 0),
            "accepted_heuristic_count": self.accepted_counts.get("heuristic", 0),
            "accepted_random_count": self.accepted_counts.get("random", 0),
            "init_FR_before_repair": float(np.mean(feasible_before)) if feasible_before else 0.0,
            "init_CV_before_repair": float(np.mean(cv_before)) if cv_before else 0.0,
            "init_FR_after_repair": float(np.mean(feasible_after)) if feasible_after else 0.0,
            "init_CV_after_repair": float(np.mean(cv_after)) if cv_after else 0.0,
            "init_diversity": float(np.mean(finite_diversity)) if finite_diversity else 0.0,
            "init_best_coverage": max([float(row.get("coverage", 0.0)) for row in self.init_metrics], default=0.0),
            "init_best_rsum": max([float(row.get("rsum", 0.0)) for row in self.init_metrics], default=0.0),
            "init_best_rsum_actual": max([float(row.get("rsum_actual", 0.0)) for row in self.init_metrics], default=0.0),
            "init_best_rsum_capacity": max([float(row.get("rsum_capacity", 0.0)) for row in self.init_metrics], default=0.0),
        }
        self.summary = summary
        with open(os.path.join(data_dir, "drl_init_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

