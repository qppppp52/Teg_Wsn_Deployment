"""Training loop for PPO initialization."""
from __future__ import annotations

import time
import numpy as np
from src.rl_init.env import InitDeploymentEnv
from src.rl_init.ppo_agent import PPOAgent, action_to_id
from src.rl_init.rollout_buffer import RolloutBuffer


class PPOTrainer:
    def __init__(self, ctx, config):
        self.ctx = ctx
        self.config = config
        self.cfg = config.get("drl_init", config)
        self.env = InitDeploymentEnv(ctx, config)
        state = self.env.reset(seed=self.cfg.get("seed", 42))
        self.agent = PPOAgent(state, config)
        self.training_log = []
        self.train_seconds = 0.0

    def train(self):
        start = time.time()
        episodes = int(self.cfg.get("train_episodes", 1000))
        update_every = max(int(self.cfg.get("rollout_steps", 512)), 1)
        buffer = RolloutBuffer()
        total_steps = 0
        last_stats = _empty_update_stats()
        for ep in range(episodes):
            state = self.env.reset(seed=int(self.cfg.get("seed", 42)) + ep)
            done = False
            episode_steps = 0
            total_reward = 0.0
            step_reward_sum = 0.0
            terminal_info = {}
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
                    buffer.compute_returns_and_advantages(0.0, self.agent.gamma, self.agent.gae_lambda)
                    last_stats = self.agent.update(buffer)
                    buffer.clear()
            if len(buffer) and (ep == episodes - 1):
                buffer.compute_returns_and_advantages(0.0, self.agent.gamma, self.agent.gae_lambda)
                last_stats = self.agent.update(buffer)
                buffer.clear()
            self.training_log.append(_episode_log_row(
                ep, episode_steps, total_steps, total_reward, step_reward_sum, terminal_info, last_stats
            ))
        self.train_seconds = time.time() - start
        return self.training_log


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
        "cv_capacity": info.get("cv_capacity", np.nan),
        "cv_energy": info.get("cv_energy", np.nan),
        "cv_sink": info.get("cv_sink", np.nan),
        "cv_service": info.get("cv_service", np.nan),
        "coverage": info.get("coverage", np.nan),
        "rsum_actual": info.get("rsum_actual", np.nan),
        "rsum_capacity": info.get("rsum_capacity", np.nan),
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
