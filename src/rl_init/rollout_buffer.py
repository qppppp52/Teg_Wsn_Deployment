"""Rollout buffer with GAE for PPO initialization."""
from __future__ import annotations

import numpy as np


class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.advantages = []
        self.returns = []

    def add(self, state, action_id, log_prob, value, reward, done):
        self.states.append(state)
        self.actions.append(int(action_id))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def compute_returns_and_advantages(self, last_value=0.0, gamma=0.98, gae_lambda=0.95):
        values = self.values + [float(last_value)]
        adv = 0.0
        advantages = []
        for t in reversed(range(len(self.rewards))):
            nonterminal = 0.0 if self.dones[t] else 1.0
            delta = self.rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
            adv = delta + gamma * gae_lambda * nonterminal * adv
            advantages.append(adv)
        advantages.reverse()
        self.advantages = advantages
        self.returns = (np.asarray(advantages) + np.asarray(self.values)).tolist()
        return self.advantages, self.returns

    def __len__(self):
        return len(self.rewards)
