"""DQN agent with epsilon-greedy action selection and checkpoints."""
from __future__ import annotations

import os
import random
import numpy as np
from src.dqn.replay_buffer import ReplayBuffer
from src.dqn.q_network import QNetwork, torch


class DQNAgent:
    """Small DQN agent used to control generation-level CR-MODE actions."""

    def __init__(self, state_dim, action_dim, config):
        if torch is None:
            raise ImportError("PyTorch is required for DQNAgent")
        dcfg = config.get("dqn", {})
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dims = tuple(dcfg.get("hidden_dims", [128, 128]))
        self.gamma = float(dcfg.get("gamma", 0.95))
        self.batch_size = int(dcfg.get("batch_size", 64))
        self.min_replay_size = int(dcfg.get("min_replay_size", 200))
        self.target_update_interval = int(dcfg.get("target_update_interval", 10))
        self.epsilon_start = float(dcfg.get("epsilon_start", 1.0))
        self.epsilon_end = float(dcfg.get("epsilon_end", 0.05))
        self.epsilon_decay_generations = max(int(dcfg.get("epsilon_decay_generations", 80)), 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNetwork(self.state_dim, self.action_dim, self.hidden_dims).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.action_dim, self.hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=float(dcfg.get("lr", 1e-3)))
        self.replay = ReplayBuffer(dcfg.get("replay_capacity", 5000))
        self.steps = 0
        self.last_loss = None

    def epsilon(self, gen):
        frac = min(max(float(gen) / self.epsilon_decay_generations, 0.0), 1.0)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def select_action(self, state, gen=0):
        eps = self.epsilon(gen)
        if random.random() < eps:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(torch.argmax(self.q_net(s), dim=1).item())

    def train_step(self):
        if len(self.replay) < max(self.min_replay_size, self.batch_size):
            return None
        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            target = rewards + self.gamma * (1.0 - dones) * self.target_net(next_states).max(dim=1, keepdim=True).values
        loss = torch.nn.functional.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.steps += 1
        if self.steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        self.last_loss = float(loss.item())
        return self.last_loss

    def save(self, path):
        """Save a full training checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dims": self.hidden_dims,
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "last_loss": self.last_loss,
        }, path)

    def load(self, path, map_location=None):
        """Load a checkpoint into this agent."""
        checkpoint = torch.load(path, map_location=map_location or self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint.get("target_net", checkpoint["q_net"]))
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps = int(checkpoint.get("steps", 0))
        self.last_loss = checkpoint.get("last_loss")
        return self

    @classmethod
    def from_checkpoint(cls, path, config):
        """Create an agent from a saved checkpoint."""
        if torch is None:
            raise ImportError("PyTorch is required for DQNAgent")
        checkpoint = torch.load(path, map_location="cpu")
        agent = cls(checkpoint["state_dim"], checkpoint["action_dim"], config)
        return agent.load(path)
