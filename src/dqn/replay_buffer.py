"""Replay buffer for DQN."""
from __future__ import annotations

from collections import deque
import random
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=int(capacity))

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((np.asarray(state, dtype=np.float32), int(action), float(reward), np.asarray(next_state, dtype=np.float32), bool(done)))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, int(batch_size))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.stack(states), np.asarray(actions), np.asarray(rewards, dtype=np.float32), np.stack(next_states), np.asarray(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)
