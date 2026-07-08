"""Training loop for PPO initialization."""
from __future__ import annotations

import time
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
        for ep in range(episodes):
            state = self.env.reset(seed=int(self.cfg.get("seed", 42)) + ep)
            done = False
            ep_reward = 0.0
            while not done:
                action, log_prob, value, entropy = self.agent.act(state, deterministic=False)
                next_state, reward, done, info = self.env.step(action)
                buffer.add(state, action_to_id(action), log_prob, value, reward, done)
                ep_reward += reward
                total_steps += 1
                state = next_state
                if len(buffer) >= update_every:
                    buffer.compute_returns_and_advantages(0.0, self.agent.gamma, self.agent.gae_lambda)
                    stats = self.agent.update(buffer)
                    buffer.clear()
                    self.training_log.append({"episode": ep, "step": total_steps, "episode_reward": ep_reward, **stats})
            if len(buffer) and (ep == episodes - 1):
                buffer.compute_returns_and_advantages(0.0, self.agent.gamma, self.agent.gae_lambda)
                stats = self.agent.update(buffer)
                buffer.clear()
                self.training_log.append({"episode": ep, "step": total_steps, "episode_reward": ep_reward, **stats})
        self.train_seconds = time.time() - start
        return self.training_log
