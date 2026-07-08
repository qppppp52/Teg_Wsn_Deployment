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
from src.rl_init.init_evaluator import evaluate_init_individual
from src.rl_init.diversity_filter import filter_population_by_diversity
from src.rl_init.logger import save_rows
from src.rl_init.checkpoint import save_checkpoint, load_checkpoint
from src.utils.logger import get_logger

logger = get_logger("DRLInitPopulationGenerator")


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
        self.generation_seconds = 0.0

    def train(self):
        if self.trainer is not None:
            return self.training_log
        try:
            from src.rl_init.ppo_trainer import PPOTrainer
            self.trainer = PPOTrainer(self.ctx, self.config)
            checkpoint_path = self.cfg.get("checkpoint_path", "experiments/checkpoints/drl_init_policy.pt")
            if self.cfg.get("load_checkpoint_if_exists", True) and load_checkpoint(self.trainer.agent, checkpoint_path):
                self.loaded_checkpoint = True
                self.policy_path = checkpoint_path
                self.training_log = [{
                    "episode": -1,
                    "step": 0,
                    "episode_reward": 0.0,
                    "loss": 0.0,
                    "policy_loss": 0.0,
                    "value_loss": 0.0,
                    "entropy": 0.0,
                    "loaded_checkpoint": True,
                }]
                self.train_seconds = 0.0
                return self.training_log
            self.training_log = self.trainer.train()
            self.loaded_checkpoint = False
            self.train_seconds = self.trainer.train_seconds
            for row in self.training_log:
                row.setdefault("loaded_checkpoint", False)
            if self.cfg.get("save_checkpoint", True):
                self.policy_path = save_checkpoint(self.trainer.agent, checkpoint_path)
        except ImportError as exc:
            logger.warning(f"PyTorch unavailable; DRL initializer will use heuristic/random fallback: {exc}")
            self.training_log = [{"loaded_checkpoint": False, "fallback_reason": "torch_unavailable"}]
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
        for idx, item in enumerate(accepted[:population_size]):
            sol = item["solution"]
            self.init_metrics.append({
                "individual_id": idx,
                "source_type": item["source_type"],
                "quality_score": item["quality_score"],
                "feasible_before_repair": getattr(sol, "feasible_before_repair", False),
                "cv_before_repair": getattr(sol, "cv_before_repair", sol.cv),
                "feasible": sol.feasible,
                "cv": sol.cv,
                "coverage": sol.coverage,
                "rsum_actual": sol.metadata.get("throughput_actual", sol.throughput),
                "rsum_capacity": sol.metadata.get("throughput_capacity", sol.throughput),
                "repair_iter": getattr(sol, "repair_iter", 0),
                "active_sensors": int(np.sum(sol.x)),
                "active_aps": int(np.sum(sol.y)),
                "diversity_score": 0.0,
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
        summary = {
            "drl_training_time": self.train_seconds,
            "generation_seconds": self.generation_seconds,
            "loaded_checkpoint": self.loaded_checkpoint,
            "checkpoint_path": self.policy_path,
            "accepted_drl_count": self.accepted_counts.get("drl", 0),
            "accepted_heuristic_count": self.accepted_counts.get("heuristic", 0),
            "accepted_random_count": self.accepted_counts.get("random", 0),
        }
        with open(os.path.join(data_dir, "drl_init_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
