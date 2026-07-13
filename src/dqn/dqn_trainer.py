"""Offline episode trainer for the generation-level DQN controller."""
from __future__ import annotations

import copy
import csv
import json
import os
import time
import numpy as np

from src.dqn.config import DQNConfig
from src.optimizers.dqn_cr_mode import DQNCRMode
from src.preprocessing.preprocessor import run_preprocessing
from src.scene.scenario_builder import Scenario
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger("DQNTrainer")


class DQNTrainer:
    """Train one persistent DQN agent across independently seeded MODE episodes."""

    def __init__(self, config: dict):
        self.config = copy.deepcopy(config)
        self.dqn_config = DQNConfig.from_mapping(self.config)
        if self.dqn_config.mode != "train":
            raise ValueError("DQNTrainer requires dqn.mode=train")
        project_root = self.config.get("project_root", os.getcwd())
        self.output_dir = os.path.join(project_root, "experiments", "dqn_training")
        self.checkpoint_dir = os.path.join(project_root, "experiments", "checkpoints")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_path = self.dqn_config.checkpoint_path
        if not os.path.isabs(self.best_path):
            self.best_path = os.path.join(project_root, self.best_path)
        self.latest_path = os.path.join(
            self.checkpoint_dir,
            "dqn_cr_mode_small_center_heat_latest.pt",
        )
        self.records = []
        self.agent = None

    def train(self):
        training_seeds = self.config.get("experiment", {}).get(
            "dqn_training_seeds",
            list(range(1000, 1000 + self.dqn_config.training_episodes)),
        )
        validation_seeds = self.config.get("experiment", {}).get(
            "dqn_validation_seeds",
            [142, 143, 144],
        )
        if len(training_seeds) < self.dqn_config.training_episodes:
            raise ValueError("Not enough dqn_training_seeds for configured episodes")

        best_score = -np.inf
        validations_without_improvement = 0
        started = time.time()
        for episode in range(1, self.dqn_config.training_episodes + 1):
            seed = int(training_seeds[episode - 1])
            optimizer = self._run_episode(seed, phase="train")
            self.agent = optimizer.agent
            final_hv = float(optimizer.convergence_history["HV"][-1])
            record = {
                "episode": episode,
                "seed": seed,
                "training_hv": final_hv,
                "validation_hv_median": np.nan,
                "interaction_steps": self.agent.interaction_steps,
                "gradient_steps": self.agent.gradient_steps,
                "epsilon": self.agent.epsilon(),
                "loss": self.agent.last_loss,
                "elapsed_seconds": time.time() - started,
            }

            if episode % self.dqn_config.validation_interval == 0:
                scores = [
                    float(self._run_episode(int(val_seed), phase="eval").convergence_history["HV"][-1])
                    for val_seed in validation_seeds
                ]
                score = float(np.median(scores))
                record["validation_hv_median"] = score
                if score > best_score + 1.0e-12:
                    best_score = score
                    validations_without_improvement = 0
                    self.agent.save(self.best_path)
                else:
                    validations_without_improvement += 1
                logger.info(
                    f"Episode {episode}: train_HV={final_hv:.6f} "
                    f"validation_HV={score:.6f} best={best_score:.6f}"
                )
            self.agent.save(self.latest_path)
            self.records.append(record)
            self._save_records()
            if validations_without_improvement >= self.dqn_config.validation_patience:
                logger.info(f"Early stopping at episode {episode}")
                break

        if not os.path.isfile(self.best_path):
            self.agent.save(self.best_path)
        self._save_manifest(started)
        return self.best_path

    def _run_episode(self, seed, phase):
        set_seed(seed, include_torch=True)
        episode_config = copy.deepcopy(self.config)
        episode_config.setdefault("runtime", {})["output_dir"] = self.output_dir
        scenario = Scenario(episode_config).build()
        context = run_preprocessing(scenario, episode_config, seed)
        context.config = episode_config
        optimizer = DQNCRMode(
            context,
            episode_config,
            agent=self.agent,
            phase=phase,
        )
        optimizer.run()
        return optimizer

    def _save_records(self):
        path = os.path.join(self.output_dir, "training_metrics.csv")
        if not self.records:
            return
        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(self.records[0]))
            writer.writeheader()
            writer.writerows(self.records)

    def _save_manifest(self, started):
        manifest = {
            "schema_version": 1,
            "training_episodes_completed": len(self.records),
            "best_checkpoint": os.path.abspath(self.best_path),
            "latest_checkpoint": os.path.abspath(self.latest_path),
            "population_size": self.config.get("mode", {}).get("population_size"),
            "max_generations": self.config.get("mode", {}).get("max_generations"),
            "rsum_ref_min": self.config.get("evaluation", {}).get("rsum_ref_min"),
            "rsum_ref_max": self.config.get("evaluation", {}).get("rsum_ref_max"),
            "validation_seeds": self.config.get("experiment", {}).get(
                "dqn_validation_seeds", [142, 143, 144]
            ),
            "elapsed_seconds": time.time() - started,
        }
        with open(
            os.path.join(self.output_dir, "training_manifest.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(manifest, file, indent=2)
