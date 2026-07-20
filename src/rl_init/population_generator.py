"""Population generator for PPO-initialized CR-MODE."""
from __future__ import annotations

import json
import math
import os
import time
import numpy as np

from src.model.individual import create_random_individual
from src.model.population import Population
from src.rl_init.checkpoint import (
    build_checkpoint_meta,
    load_checkpoint,
    resolve_checkpoint_path,
    save_checkpoint,
)
from src.rl_init.config_validation import validate_drl_init_config
from src.rl_init.diversity_filter import (
    filter_population_by_diversity,
    hamming_distance_deployment,
    objective_space_distance,
    priority_l2_distance,
)
from src.rl_init.env import InitDeploymentEnv
from src.rl_init.evaluation_counter import DRLInitEvaluationCounter
from src.rl_init.heuristic_initializer import create_composite_heuristic_individual
from src.rl_init.init_evaluator import evaluate_init_individual
from src.rl_init.logger import save_rows
from src.rl_init.source_summary import (
    build_ppo_training_summary,
    build_source_wise_summary,
)
from src.utils.logger import get_logger


logger = get_logger("DRL-INIT")


TRAINING_LOG_COLUMNS = [
    "episode", "step", "episode_steps", "episode_reward", "total_reward",
    "step_reward_sum", "terminal_reward", "feasible", "cv", "cv_deploy",
    "cv_link", "cv_power", "cv_energy", "cv_energy_sensor", "cv_energy_ap", "cv_sink", "cv_service",
    "coverage", "rsum_capacity", "rsum_norm", "rsum_ref_min", "rsum_ref_max",
    "repair_iter", "repair_success", "num_sensors", "num_aps",
    "invalid_action_count", "loss", "policy_loss", "value_loss", "entropy",
    "approx_kl", "updated_this_episode", "ppo_update_count",
    "loaded_checkpoint", "fallback_reason", "policy_source", "torch_available",
    "checkpoint_path", "checkpoint_compatible", "checkpoint_skip_reason",
    "checkpoint_error", "checkpoint_mode", "config_hash",
]


def _training_log_placeholder(
    loaded_checkpoint=False,
    fallback_reason="",
    policy_source="trained",
    torch_available=False,
    checkpoint_path="",
    checkpoint_compatible=False,
    checkpoint_skip_reason="",
    checkpoint_error="",
    checkpoint_mode="",
    config_hash="",
):
    row = {key: np.nan for key in TRAINING_LOG_COLUMNS}
    row.update({
        "episode": -1,
        "step": 0,
        "episode_steps": 0,
        "episode_reward": 0.0,
        "total_reward": 0.0,
        "step_reward_sum": 0.0,
        "updated_this_episode": False,
        "ppo_update_count": 0,
        "loaded_checkpoint": loaded_checkpoint,
        "fallback_reason": fallback_reason,
        "policy_source": policy_source,
        "torch_available": torch_available,
        "checkpoint_path": checkpoint_path,
        "checkpoint_compatible": checkpoint_compatible,
        "checkpoint_skip_reason": checkpoint_skip_reason,
        "checkpoint_error": checkpoint_error,
        "checkpoint_mode": checkpoint_mode,
        "config_hash": config_hash,
    })
    return row


class DRLInitPopulationGenerator:
    def __init__(self, ctx, config, evaluation_counter=None):
        self.ctx = ctx
        self.config = config
        self.cfg = validate_drl_init_config(config)
        self.rng = np.random.default_rng(int(self.cfg.get("seed", 42)))
        self.evaluation_counter = evaluation_counter or DRLInitEvaluationCounter()
        self.trainer = None
        self.training_log = []
        self.train_seconds = 0.0
        self.init_metrics = []
        self.accepted_counts = {"drl": 0, "heuristic": 0, "random": 0}
        self.target_counts = {"drl": 0, "heuristic": 0, "random": 0}
        self.policy_path = ""
        self.loaded_checkpoint = False
        self.policy_source = "trained"
        self.torch_available = False

        self.checkpoint_load_attempted = False
        self.loaded_checkpoint_compatible = False
        self.checkpoint_load_skip_reason = ""
        self.checkpoint_load_error = ""
        self.new_checkpoint_saved = False
        self.new_checkpoint_path = ""
        self.new_checkpoint_save_error = ""

        # Backward-readable aliases; compatibility now refers only to the loaded file.
        self.checkpoint_compatible = False
        self.checkpoint_skip_reason = ""
        self.checkpoint_error = ""
        self.checkpoint_mode = str(self.cfg.get("checkpoint_mode", "per_seed"))
        self.config_hash = ""
        self.checkpoint_load_seconds = 0.0
        self.pretrain_seconds = 0.0
        self.policy_setup_seconds = 0.0
        self.summary = {}
        self.population_generation_seconds = 0.0
        self.generation_seconds = 0.0
        self.cost_timing = {}

        self.drl_candidate_count = 0
        self.drl_rejected_by_diversity_count = 0
        self.drl_retry_count = 0
        self.fallback_fill_count = 0
        self.fallback_fill_heuristic_count = 0
        self.fallback_fill_random_count = 0
        self.fallback_reason = ""
        self.drl_init_valid = False
        self.drl_init_invalid_reasons = []

    def train(self):
        if self.trainer is not None:
            return self.training_log
        try:
            from src.rl_init.ppo_trainer import PPOTrainer

            self.torch_available = True
            setup_start = time.time()
            self.trainer = PPOTrainer(
                self.ctx,
                self.config,
                evaluation_counter=self.evaluation_counter,
            )
            self.policy_setup_seconds = time.time() - setup_start
            checkpoint_path = resolve_checkpoint_path(
                self.config,
                seed=self.cfg.get("seed", 42),
            )
            self.policy_path = checkpoint_path
            expected_meta = build_checkpoint_meta(
                self.trainer.agent,
                self.ctx,
                self.config,
                self.checkpoint_mode,
            )
            self.config_hash = expected_meta.get("config_hash", "")
            self.trainer.set_training_context({
                "policy_source": "trained",
                "torch_available": True,
                "checkpoint_path": checkpoint_path,
                "checkpoint_mode": self.checkpoint_mode,
                "config_hash": self.config_hash,
            })
            logger.info(f"checkpoint path={checkpoint_path}")

            load_result = {
                "loaded": False,
                "compatible": False,
                "skip_reason": "load_disabled",
                "error": "",
            }
            load_enabled = (
                not self.cfg.get("force_retrain", False)
                and self.cfg.get("load_checkpoint_if_exists", True)
            )
            self.checkpoint_load_attempted = bool(load_enabled)
            if load_enabled:
                load_start = time.time()
                load_result = load_checkpoint(
                    self.trainer.agent,
                    checkpoint_path,
                    expected_meta=expected_meta,
                )
                self.checkpoint_load_seconds = time.time() - load_start

            self.loaded_checkpoint_compatible = bool(load_result.get("compatible", False))
            self.checkpoint_load_skip_reason = str(load_result.get("skip_reason", ""))
            self.checkpoint_load_error = str(load_result.get("error", ""))
            self.checkpoint_compatible = self.loaded_checkpoint_compatible
            self.checkpoint_skip_reason = self.checkpoint_load_skip_reason
            self.checkpoint_error = self.checkpoint_load_error
            logger.info(
                "checkpoint load "
                f"loaded={bool(load_result.get('loaded', False))} "
                f"compatible={self.loaded_checkpoint_compatible} "
                f"reason={self.checkpoint_load_skip_reason or 'none'}"
            )

            if load_result.get("loaded", False):
                self.loaded_checkpoint = True
                self.policy_source = "loaded_checkpoint"
                self.training_log = [_training_log_placeholder(
                    True,
                    "",
                    "loaded_checkpoint",
                    True,
                    checkpoint_path,
                    True,
                    "",
                    "",
                    self.checkpoint_mode,
                    self.config_hash,
                )]
                self.train_seconds = 0.0
                return self.training_log

            if self.checkpoint_load_skip_reason not in {"", "missing_checkpoint", "load_disabled"}:
                logger.warning(
                    f"Skipping PPO checkpoint {checkpoint_path}: "
                    f"{self.checkpoint_load_skip_reason} {self.checkpoint_load_error}"
                )

            self.training_log = self.trainer.train()
            self.loaded_checkpoint = False
            self.policy_source = "trained"
            self.train_seconds = self.trainer.train_seconds
            for row in self.training_log:
                row.setdefault("loaded_checkpoint", False)
                row.setdefault("fallback_reason", "")
                row.setdefault("policy_source", self.policy_source)
                row.setdefault("torch_available", self.torch_available)
                row.setdefault("checkpoint_path", self.policy_path)
                row.setdefault("checkpoint_compatible", self.loaded_checkpoint_compatible)
                row.setdefault("checkpoint_skip_reason", self.checkpoint_load_skip_reason)
                row.setdefault("checkpoint_error", self.checkpoint_load_error)
                row.setdefault("checkpoint_mode", self.checkpoint_mode)
                row.setdefault("config_hash", self.config_hash)

            if self.cfg.get("save_checkpoint", True):
                try:
                    saved_path = save_checkpoint(
                        self.trainer.agent,
                        checkpoint_path,
                        meta=expected_meta,
                    )
                    self.new_checkpoint_saved = bool(saved_path)
                    self.new_checkpoint_path = saved_path or ""
                    if saved_path:
                        self.policy_path = saved_path
                except Exception as exc:
                    self.new_checkpoint_saved = False
                    self.new_checkpoint_save_error = str(exc)
                    if self.cfg.get("require_drl_policy", True):
                        raise
        except ImportError as exc:
            logger.warning(
                "PyTorch unavailable; DRL initializer will use heuristic/random "
                f"fallback: {exc}"
            )
            self.training_log = [_training_log_placeholder(
                False,
                "torch_unavailable",
                "fallback",
                False,
                self.policy_path,
                False,
                "torch_unavailable",
                str(exc),
                self.checkpoint_mode,
                self.config_hash,
            )]
            self.policy_source = "fallback"
            self.torch_available = False
            self.loaded_checkpoint_compatible = False
            self.checkpoint_compatible = False
            self.checkpoint_load_skip_reason = "torch_unavailable"
            self.checkpoint_skip_reason = "torch_unavailable"
            self.checkpoint_load_error = str(exc)
            self.checkpoint_error = str(exc)
            self.train_seconds = 0.0
        return self.training_log

    def generate(self, population_size):
        population_size = int(population_size)
        self.train()
        generation_start = time.time()
        self.target_counts = _target_mix(population_size, self.cfg)
        drl_target = self.target_counts["drl"]
        heuristic_target = self.target_counts["heuristic"]
        random_target = self.target_counts["random"]
        logger.info(
            f"target init mix drl={drl_target} heuristic={heuristic_target} "
            f"random={random_target}"
        )

        accepted_drl = []
        drl_candidates = []
        if self.trainer is not None:
            diversity_cfg = self.cfg.get("diversity", {})
            retry_limit = int(diversity_cfg.get("max_duplicate_retry", 30))
            max_candidates = int(
                diversity_cfg.get(
                    "max_drl_candidate_evaluations",
                    drl_target + retry_limit,
                )
            )
            max_candidates = max(
                drl_target,
                min(max_candidates, drl_target + retry_limit),
            )
            while len(accepted_drl) < drl_target and len(drl_candidates) < max_candidates:
                individual = self.generate_one(deterministic=False)
                item = evaluate_init_individual(
                    individual,
                    self.ctx,
                    self.config,
                    "drl",
                )
                self.evaluation_counter.increment("drl_candidate_post_evaluations")
                drl_candidates.append(item)
                accepted_drl = filter_population_by_diversity(
                    drl_candidates,
                    target_count=drl_target,
                    config=self.config,
                )
            self.drl_candidate_count = len(drl_candidates)
            self.drl_rejected_by_diversity_count = (
                self.drl_candidate_count - len(accepted_drl)
            )
            self.drl_retry_count = max(0, self.drl_candidate_count - drl_target)
        else:
            self.fallback_reason = "drl_policy_unavailable"

        individuals = [item["individual"] for item in accepted_drl]
        sources = ["drl"] * len(accepted_drl)
        accepted = list(accepted_drl)
        drl_deficit = max(0, drl_target - len(accepted_drl))
        if drl_deficit:
            self.fallback_reason = self.fallback_reason or "drl_diversity_retry_exhausted"

        heuristic_fill_target = heuristic_target + drl_deficit
        for index in range(heuristic_fill_target):
            try:
                individual = create_composite_heuristic_individual(
                    self.ctx,
                    self.config,
                    self.rng,
                )
                item = evaluate_init_individual(
                    individual,
                    self.ctx,
                    self.config,
                    "heuristic",
                )
                self.evaluation_counter.increment("heuristic_candidate_evaluations")
                individuals.append(item["individual"])
                sources.append("heuristic")
                accepted.append(item)
                if index >= heuristic_target:
                    self.fallback_fill_heuristic_count += 1
            except Exception:
                self.fallback_fill_random_count += 1

        random_fill_target = random_target + self.fallback_fill_random_count
        mapping = self.ctx.index_mapping
        for _ in range(random_fill_target):
            individual = create_random_individual(
                mapping.num_sensor_candidates,
                mapping.num_ap_candidates,
            )
            item = evaluate_init_individual(
                individual,
                self.ctx,
                self.config,
                "random",
            )
            self.evaluation_counter.increment("random_candidate_evaluations")
            individuals.append(item["individual"])
            sources.append("random")
            accepted.append(item)

        while len(individuals) < population_size:
            individual = create_random_individual(
                mapping.num_sensor_candidates,
                mapping.num_ap_candidates,
            )
            item = evaluate_init_individual(
                individual,
                self.ctx,
                self.config,
                "random",
            )
            self.evaluation_counter.increment("random_candidate_evaluations")
            individuals.append(item["individual"])
            sources.append("random")
            accepted.append(item)
            self.fallback_fill_random_count += 1
            self.fallback_reason = self.fallback_reason or "population_underfilled"

        self.fallback_fill_count = (
            self.fallback_fill_heuristic_count + self.fallback_fill_random_count
        )
        population = Population(
            population_size,
            mapping.num_sensor_candidates,
            mapping.num_ap_candidates,
        )
        population.individuals = individuals[:population_size]
        final_sources = sources[:population_size]
        self.accepted_counts = {
            source: final_sources.count(source)
            for source in ("drl", "heuristic", "random")
        }
        self.init_metrics = self._build_init_metrics(accepted[:population_size])
        self.population_generation_seconds = time.time() - generation_start
        self.generation_seconds = self.population_generation_seconds
        self._evaluate_validity(population_size)
        logger.info(
            f"accepted init mix drl={self.accepted_counts['drl']} "
            f"heuristic={self.accepted_counts['heuristic']} "
            f"random={self.accepted_counts['random']}"
        )
        logger.info(
            f"drl candidates={self.drl_candidate_count} "
            f"rejected_by_diversity={self.drl_rejected_by_diversity_count} "
            f"fallback_fill={self.fallback_fill_count}"
        )
        return population

    def generate_one(self, deterministic=False):
        if self.trainer is None:
            return create_composite_heuristic_individual(
                self.ctx,
                self.config,
                self.rng,
            )
        env = InitDeploymentEnv(
            self.ctx,
            self.config,
            evaluation_counter=self.evaluation_counter,
            terminal_counter_key="drl_policy_rollout_terminal_evaluations",
        )
        state = env.reset(seed=int(self.rng.integers(0, 2**31 - 1)))
        done = False
        while not done:
            action, _, _, _ = self.trainer.agent.act(
                state,
                deterministic=deterministic,
            )
            state, _, done, _ = env.step(action)
        return env.build_current_individual()

    def save_artifacts(self, output_dir):
        data_dir = os.path.join(output_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        if self.cfg.get("save_training_log", True):
            save_rows(
                self.training_log,
                os.path.join(data_dir, "drl_init_training_log.csv"),
            )
        if self.cfg.get("save_generated_population", True):
            save_rows(
                self.init_metrics,
                os.path.join(data_dir, "init_population_metrics.csv"),
            )
        self.summary = self._build_summary()
        with open(
            os.path.join(data_dir, "drl_init_summary.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(self.summary, file, indent=2, allow_nan=False)

    def _build_init_metrics(self, accepted):
        rows = []
        rsum_scale = float(
            self.cfg.get("normalization", {}).get("rsum_ref_max", 2.0e8)
        )
        for index, item in enumerate(accepted):
            solution = item["solution"]
            previous = accepted[:index]
            priority = [
                priority_l2_distance(item["individual"], prev["individual"])
                for prev in previous
            ]
            hamming = [
                hamming_distance_deployment(solution, prev["solution"])
                for prev in previous
            ]
            objective = [
                objective_space_distance(solution, prev["solution"], rsum_scale)
                for prev in previous
            ]
            rows.append({
                "individual_id": index,
                "source_type": item["source_type"],
                "quality_score": item["quality_score"],
                "feasible_before_repair": getattr(solution, "feasible_before_repair", False),
                "cv_before_repair": getattr(solution, "cv_before_repair", solution.cv),
                "feasible": solution.feasible,
                "cv": solution.cv,
                "cv_deploy": getattr(solution, "cv_deploy", 0.0),
                "cv_link": getattr(solution, "cv_link", 0.0),
                "cv_power": getattr(solution, "cv_power", 0.0),
                "cv_energy_sensor": getattr(solution, "cv_energy_sensor", 0.0),
                "cv_energy_ap": getattr(solution, "cv_energy_ap", 0.0),
                "cv_energy": getattr(solution, "cv_energy", 0.0),
                "cv_sink": getattr(solution, "cv_sink", 0.0),
                "cv_service": getattr(solution, "cv_service", 0.0),
                "coverage": solution.coverage,
                "rsum_capacity": solution.rsum_capacity,
                "repair_iter": getattr(solution, "repair_iter", 0),
                "repair_success": getattr(solution, "repair_success", bool(solution.feasible)),
                "active_sensors": int(np.sum(solution.x)),
                "active_aps": int(np.sum(solution.y)),
                "priority_l2_distance_to_prev": min(priority) if priority else None,
                "deployment_hamming_distance_to_prev": min(hamming) if hamming else None,
                "objective_space_distance_to_prev": min(objective) if objective else None,
                "diversity_score": (
                    0.5 * min(priority) + 0.5 * min(hamming)
                    if priority and hamming else None
                ),
                "diversity_accept_reason": item.get("diversity_accept_reason"),
                "diversity_reject_reason": item.get("diversity_reject_reason"),
            })
        return rows

    def _evaluate_validity(self, population_size):
        reasons = []
        if not self.torch_available:
            reasons.append("torch_unavailable")
        if self.policy_source not in {"trained", "loaded_checkpoint"}:
            reasons.append("policy_source_fallback")
        minimum = int(math.ceil(
            population_size * float(self.cfg.get("min_accepted_drl_ratio", 0.5))
        ))
        if self.accepted_counts.get("drl", 0) < minimum:
            reasons.append("accepted_drl_count_below_threshold")
        ppo_summary = build_ppo_training_summary(
            self.training_log,
            self.cfg.get("training_summary_window", 100),
        )
        if (
            self.policy_source == "trained"
            and ppo_summary.get("update_count", 0) <= 0
        ):
            reasons.append("ppo_no_valid_update")
        if (
            self.checkpoint_load_attempted
            and self.policy_source == "fallback"
            and self.checkpoint_load_skip_reason
        ):
            reasons.append("checkpoint_load_failed")
        self.drl_init_invalid_reasons = list(dict.fromkeys(reasons))
        self.drl_init_valid = not self.drl_init_invalid_reasons

    def _build_summary(self):
        feasible_before = [
            bool(row.get("feasible_before_repair", False))
            for row in self.init_metrics
        ]
        feasible_after = [bool(row.get("feasible", False)) for row in self.init_metrics]
        cv_before = [
            float(row.get("cv_before_repair", row.get("cv", 0.0)))
            for row in self.init_metrics
        ]
        cv_after = [float(row.get("cv", 0.0)) for row in self.init_metrics]
        source_wise, differences = build_source_wise_summary(self.init_metrics)
        ppo_training = build_ppo_training_summary(
            self.training_log,
            self.cfg.get("training_summary_window", 100),
        )
        timing = {
            "policy_setup_seconds": self.policy_setup_seconds,
            "checkpoint_load_seconds": self.checkpoint_load_seconds,
            "ppo_training_seconds": self.train_seconds,
            "population_generation_seconds": self.population_generation_seconds,
            **self.cost_timing,
        }
        total_seconds = float(timing.get("total_end_to_end_seconds", 0.0))
        accounted_keys = (
            "policy_setup_seconds",
            "checkpoint_load_seconds",
            "ppo_training_seconds",
            "population_generation_seconds",
            "gen0_evaluation_seconds",
            "cr_mode_search_seconds",
        )
        timing["unattributed_overhead_seconds"] = max(
            total_seconds - sum(
                float(timing.get(key, 0.0)) for key in accounted_keys
            ),
            0.0,
        )
        return {
            "algorithm": "drl_init_cr_mode",
            "policy_algorithm": "ppo",
            "DRL_INIT_VALID": self.drl_init_valid,
            "DRL_INIT_INVALID_REASONS": self.drl_init_invalid_reasons,
            "loaded_checkpoint": self.loaded_checkpoint,
            "checkpoint_path": self.policy_path,
            "policy_source": self.policy_source,
            "torch_available": self.torch_available,
            "train_episodes": int(self.cfg.get("train_episodes", 0)),
            "drl_training_time": self.train_seconds,
            "generation_time": self.population_generation_seconds,
            "init_generation_time": self.population_generation_seconds,
            "population_generation_seconds": self.population_generation_seconds,
            "pretrain_time_seconds": self.pretrain_seconds,
            "checkpoint_load_time_seconds": self.checkpoint_load_seconds,
            "policy_setup_seconds": self.policy_setup_seconds,
            "checkpoint_load_attempted": self.checkpoint_load_attempted,
            "checkpoint_loaded": self.loaded_checkpoint,
            "loaded_checkpoint_compatible": self.loaded_checkpoint_compatible,
            "checkpoint_load_skip_reason": self.checkpoint_load_skip_reason,
            "checkpoint_load_error": self.checkpoint_load_error,
            "new_checkpoint_saved": self.new_checkpoint_saved,
            "new_checkpoint_path": self.new_checkpoint_path,
            "new_checkpoint_save_error": self.new_checkpoint_save_error,
            "checkpoint_compatible": self.loaded_checkpoint_compatible,
            "checkpoint_skip_reason": self.checkpoint_load_skip_reason,
            "checkpoint_error": self.checkpoint_load_error,
            "checkpoint_mode": self.checkpoint_mode,
            "config_hash": self.config_hash,
            "accepted_drl_count": self.accepted_counts.get("drl", 0),
            "accepted_heuristic_count": self.accepted_counts.get("heuristic", 0),
            "accepted_random_count": self.accepted_counts.get("random", 0),
            "init_FR_before_repair": float(np.mean(feasible_before)) if feasible_before else 0.0,
            "init_CV_before_repair": float(np.mean(cv_before)) if cv_before else 0.0,
            "init_FR_after_repair": float(np.mean(feasible_after)) if feasible_after else 0.0,
            "init_CV_after_repair": float(np.mean(cv_after)) if cv_after else 0.0,
            "init_best_coverage": max(
                [float(row.get("coverage", 0.0)) for row in self.init_metrics],
                default=0.0,
            ),
            "init_best_rsum_capacity": max(
                [float(row.get("rsum_capacity", 0.0)) for row in self.init_metrics],
                default=0.0,
            ),
            "rsum_ref_min": float(
                self.cfg.get("normalization", {}).get("rsum_ref_min", 0.0)
            ),
            "rsum_ref_max": float(
                self.cfg.get("normalization", {}).get("rsum_ref_max", 1.0)
            ),
            "mix": {
                "target": self.target_counts,
                "accepted": self.accepted_counts,
                "drl_candidate_count": self.drl_candidate_count,
                "drl_rejected_by_diversity_count": self.drl_rejected_by_diversity_count,
                "drl_retry_count": self.drl_retry_count,
                "fallback_fill_count": self.fallback_fill_count,
                "fallback_fill_heuristic_count": self.fallback_fill_heuristic_count,
                "fallback_fill_random_count": self.fallback_fill_random_count,
                "fallback_reason": self.fallback_reason,
            },
            "source_wise": source_wise,
            "source_differences": differences,
            "ppo_training": ppo_training,
            "cost": {
                **self.evaluation_counter.to_dict(),
                **timing,
            },
        }

def _target_mix(population_size, cfg):
    sources = ("drl", "heuristic", "random")
    raw = {
        "drl": population_size * float(cfg.get("drl_ratio", 0.6)),
        "heuristic": population_size * float(cfg.get("heuristic_ratio", 0.2)),
        "random": population_size * float(cfg.get("random_ratio", 0.2)),
    }
    counts = {source: int(math.floor(raw[source])) for source in sources}
    remaining = population_size - sum(counts.values())
    order = sorted(
        sources,
        key=lambda source: (raw[source] - counts[source], -sources.index(source)),
        reverse=True,
    )
    for source in order[:remaining]:
        counts[source] += 1
    return counts
