"""Training loop for PPO initialization."""
from __future__ import annotations

import time
import numpy as np

from src.rl_init.config_validation import validate_drl_init_config
from src.rl_init.env import InitDeploymentEnv
from src.rl_init.ppo_agent import PPOAgent, action_to_id
from src.rl_init.rollout_buffer import RolloutBuffer
from src.utils.logger import get_logger


logger = get_logger("DRL-INIT")


class PPOTrainer:
    def __init__(self, ctx, config, evaluation_counter=None):
        self.ctx = ctx
        self.config = config
        self.cfg = validate_drl_init_config(config)
        self.env = InitDeploymentEnv(
            ctx,
            config,
            evaluation_counter=evaluation_counter,
            terminal_counter_key="ppo_terminal_evaluations",
        )
        state = self.env.reset(seed=self.cfg.get("seed", 42))
        self.agent = PPOAgent(state, config)
        self.training_log = []
        self.training_context = {}
        self.train_seconds = 0.0
        self.update_count = 0
        self.first_update_episode = None

    def set_training_context(self, context: dict):
        self.training_context = dict(context or {})

    def train(self):
        start = time.time()
        episodes = int(self.cfg.get("train_episodes", 1000))
        update_every = max(int(self.cfg.get("rollout_steps", 512)), 1)
        log_interval = max(int(self.cfg.get("log_interval_episodes", 100)), 1)
        buffer = RolloutBuffer()
        total_steps = 0
        last_stats = _empty_update_stats()
        logger.info(
            f"PPO training started episodes={episodes} rollout_steps={update_every} "
            f"device={self.agent.device}"
        )

        for ep in range(episodes):
            state = self.env.reset(seed=int(self.cfg.get("seed", 42)) + ep)
            done = False
            episode_steps = 0
            total_reward = 0.0
            step_reward_sum = 0.0
            terminal_info = {}
            updated_this_episode = False

            while not done:
                action, log_prob, value, entropy = self.agent.act(state, deterministic=False)
                next_state, reward, done, info = self.env.step(action)
                buffer.add(state, action_to_id(action), log_prob, value, reward, done)
                total_reward += reward
                if done:
                    terminal_info = dict(info)
                else:
                    step_reward_sum += reward
                total_steps += 1
                episode_steps += 1
                state = next_state

                if len(buffer) >= update_every:
                    last_value = rollout_bootstrap_value(self.agent, state, done)
                    buffer.compute_returns_and_advantages(
                        last_value,
                        self.agent.gamma,
                        self.agent.gae_lambda,
                    )
                    last_stats = self.agent.update(buffer)
                    self.update_count += 1
                    updated_this_episode = True
                    if self.first_update_episode is None:
                        self.first_update_episode = ep
                    buffer.clear()

            if len(buffer) and ep == episodes - 1:
                last_value = rollout_bootstrap_value(self.agent, state, done)
                buffer.compute_returns_and_advantages(
                    last_value,
                    self.agent.gamma,
                    self.agent.gae_lambda,
                )
                last_stats = self.agent.update(buffer)
                self.update_count += 1
                updated_this_episode = True
                if self.first_update_episode is None:
                    self.first_update_episode = ep
                buffer.clear()

            row = _episode_log_row(
                ep,
                episode_steps,
                total_steps,
                total_reward,
                step_reward_sum,
                terminal_info,
                last_stats,
            )
            row["updated_this_episode"] = updated_this_episode
            row["ppo_update_count"] = self.update_count
            row.update(self.training_context)
            self.training_log.append(row)

            if (ep + 1) % log_interval == 0 or ep == episodes - 1:
                window = self.training_log[-log_interval:]
                mean_reward = float(np.mean([item["episode_reward"] for item in window]))
                feasible_rate = float(np.mean([bool(item["feasible"]) for item in window]))
                logger.info(
                    f"PPO episode {ep + 1}/{episodes} mean_reward={mean_reward:.4f} "
                    f"feasible_rate={feasible_rate:.3f} "
                    f"loss={last_stats.get('loss', np.nan):.6f}"
                )

        self.train_seconds = time.time() - start
        logger.info(
            f"PPO training finished seconds={self.train_seconds:.3f} "
            f"updates={self.update_count}"
        )
        return self.training_log


def rollout_bootstrap_value(agent, state, done) -> float:
    """Use zero only for terminal rollouts; bootstrap partial rollouts."""
    if done:
        return 0.0
    return float(agent.value(state))

def _empty_update_stats():
    return {
        "loss": np.nan,
        "policy_loss": np.nan,
        "value_loss": np.nan,
        "entropy": np.nan,
        "approx_kl": np.nan,
    }


def _episode_log_row(ep, episode_steps, total_steps, total_reward, step_reward_sum, info, stats):
    return {
        "episode": ep,
        "step": total_steps,
        "episode_steps": episode_steps,
        "episode_reward": total_reward,
        "total_reward": total_reward,
        "step_reward_sum": step_reward_sum,
        "terminal_reward": info.get("terminal_reward", np.nan),
        "feasible": info.get("feasible", False),
        "cv": info.get("cv", np.nan),
        "cv_deploy": info.get("cv_deploy", np.nan),
        "cv_link": info.get("cv_link", np.nan),
        "cv_power": info.get("cv_power", np.nan),
        "cv_energy_sensor": info.get("cv_energy_sensor", np.nan),
        "cv_energy_ap": info.get("cv_energy_ap", np.nan),
        "cv_energy": info.get("cv_energy", np.nan),
        "cv_sink": info.get("cv_sink", np.nan),
        "cv_service": info.get("cv_service", np.nan),
        "coverage": info.get("coverage", np.nan),
        "rsum_capacity": info.get("rsum_capacity", np.nan),
        "rsum_norm": info.get("rsum_norm", np.nan),
        "rsum_ref_min": info.get("rsum_ref_min", np.nan),
        "rsum_ref_max": info.get("rsum_ref_max", np.nan),
        "repair_iter": info.get("repair_iter", np.nan),
        "repair_success": info.get("repair_success", bool(info.get("feasible", False))),
        "num_sensors": info.get("num_sensors", np.nan),
        "num_aps": info.get("num_aps", np.nan),
        "invalid_action_count": info.get("invalid_action_count", 0),
        "loss": stats.get("loss", np.nan),
        "policy_loss": stats.get("policy_loss", np.nan),
        "value_loss": stats.get("value_loss", np.nan),
        "entropy": stats.get("entropy", np.nan),
        "approx_kl": stats.get("approx_kl", np.nan),
        "loaded_checkpoint": False,
    }
