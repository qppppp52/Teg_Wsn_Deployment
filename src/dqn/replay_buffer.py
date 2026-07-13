"""Replay buffer for DQN."""
from __future__ import annotations

from collections import deque
import random
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=int(capacity))

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        action_mask=None,
        next_action_mask=None,
    ):
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        if action_mask is None and next_action_mask is None:
            action_mask = np.ones(1, dtype=bool)
            next_action_mask = np.ones(1, dtype=bool)
        elif action_mask is None or next_action_mask is None:
            raise ValueError("Both action_mask and next_action_mask must be provided")
        self.buffer.append((
            state,
            int(action),
            float(reward),
            next_state,
            bool(done),
            np.asarray(action_mask, dtype=bool),
            np.asarray(next_action_mask, dtype=bool),
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, int(batch_size))
        states, actions, rewards, next_states, dones, masks, next_masks = zip(*batch)
        return (
            np.stack(states),
            np.asarray(actions),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_states),
            np.asarray(dones, dtype=np.float32),
            np.stack(masks),
            np.stack(next_masks),
        )

    def __len__(self):
        return len(self.buffer)
