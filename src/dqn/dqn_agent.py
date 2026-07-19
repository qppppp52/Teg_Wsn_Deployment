"""DQN agent with masked Double-DQN learning and strict checkpoints."""
from __future__ import annotations

import hashlib
import json
import os
import random
import numpy as np

from src.dqn.action_space import ACTIONS
from src.decoder.power_decoder import INITIAL_POWER_SEMANTICS_VERSION
from src.heatsink.sink_ownership import SINK_OWNERSHIP_SEMANTICS_VERSION
from src.power.power_repair import POWER_REPAIR_SEMANTICS_VERSION
from src.power.throughput_enhancer import THROUGHPUT_ENHANCER_VERSION
from src.dqn.config import DQNConfig
from src.dqn.replay_buffer import ReplayBuffer
from src.dqn.q_network import QNetwork, torch
from src.dqn.state_builder import STATE_KEYS
from src.dqn.state_normalizer import RunningNormalizer, RunningScalarNormalizer

CHECKPOINT_SCHEMA_VERSION = 4
POLICY_CONTRACT_VERSION = 3


class DQNAgent:
    def __init__(self, state_dim, action_dim, config):
        if torch is None:
            raise ImportError("PyTorch is required for DQNAgent")
        self.config = config if isinstance(config, DQNConfig) else DQNConfig.from_mapping(config)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dims = self.config.hidden_dims
        self.gamma = self.config.gamma
        self.batch_size = self.config.batch_size
        self.min_replay_size = self.config.min_replay_size
        self.target_update_interval = self.config.target_update_interval
        self.grad_clip_norm = self.config.grad_clip_norm
        self.epsilon_start = self.config.epsilon_start
        self.epsilon_end = self.config.epsilon_end
        self.epsilon_decay_steps = self.config.epsilon_decay_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNetwork(self.state_dim, self.action_dim, self.hidden_dims).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.action_dim, self.hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(),
            lr=self.config.learning_rate,
        )
        self.replay = ReplayBuffer(self.config.replay_capacity)
        self.state_normalizer = RunningNormalizer(
            self.state_dim,
            enabled=self.config.state_normalization,
            warmup_steps=self.config.state_warmup_steps,
            clip=self.config.state_clip,
        )
        self.reward_normalizer = RunningScalarNormalizer(
            enabled=self.config.reward_normalization,
            warmup_steps=self.config.reward_warmup_steps,
            clip=max(abs(self.config.reward_clip[0]), abs(self.config.reward_clip[1])),
        )
        self.gradient_steps = 0
        self.interaction_steps = 0
        self.last_loss = None
        self.last_q_mean = None
        self.last_q_max = None
        self.target_updated = False

    @property
    def steps(self):
        return self.gradient_steps

    def epsilon(self, step=None):
        step = self.interaction_steps if step is None else int(step)
        frac = min(max(float(step) / self.epsilon_decay_steps, 0.0), 1.0)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def select_action(self, state, gen=None, action_mask=None, deterministic=False):
        mask = self._normalize_action_mask(action_mask)
        eps = 0.0 if deterministic else self.epsilon(gen)
        if not deterministic and random.random() < eps:
            return int(random.choice(np.flatnonzero(mask).tolist()))
        with torch.no_grad():
            normalized = self.state_normalizer.normalize(state)
            state_tensor = torch.as_tensor(
                normalized, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            q_values = self.q_net(state_tensor).squeeze(0)
            mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            q_values = q_values.masked_fill(~mask_tensor, -torch.inf)
            allowed = q_values[mask_tensor]
            self.last_q_mean = float(allowed.mean().item())
            self.last_q_max = float(allowed.max().item())
            return int(torch.argmax(q_values).item())

    def observe(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        action_mask,
        next_action_mask,
    ):
        self.state_normalizer.update(state)
        if done:
            self.state_normalizer.update(next_state)
        clipped = float(np.clip(reward, *self.config.reward_clip))
        self.reward_normalizer.update(clipped)
        self.replay.add(
            state,
            action,
            clipped,
            next_state,
            done,
            action_mask,
            next_action_mask,
        )
        self.interaction_steps += 1

    def train_step(self):
        if len(self.replay) < max(self.min_replay_size, self.batch_size):
            return None
        states, actions, rewards, next_states, dones, _, next_masks = self.replay.sample(
            self.batch_size
        )
        states = self.state_normalizer.normalize(states)
        next_states = self.state_normalizer.normalize(next_states)
        if self.config.reward_normalization:
            rewards = self.reward_normalizer.normalize(rewards)
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_masks = torch.as_tensor(next_masks, dtype=torch.bool, device=self.device)
        if next_masks.ndim != 2 or next_masks.shape[1] != self.action_dim:
            raise ValueError("Replay next_action_mask shape does not match action_dim")
        if torch.any(~next_masks.any(dim=1)):
            raise ValueError("Replay contains an empty next-action mask")

        q_values = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            online_next = self.q_net(next_states).masked_fill(~next_masks, -torch.inf)
            next_actions = online_next.argmax(dim=1, keepdim=True)
            target_next = self.target_net(next_states).masked_fill(~next_masks, -torch.inf)
            next_q = target_next.gather(1, next_actions)
            targets = rewards + self.gamma * (1.0 - dones) * next_q
        loss = torch.nn.functional.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        self.gradient_steps += 1
        self.target_updated = False
        if self.gradient_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.target_updated = True
        self.last_loss = float(loss.item())
        return self.last_loss

    def q_stats(self, state, action_mask=None):
        mask = self._normalize_action_mask(action_mask)
        with torch.no_grad():
            normalized = self.state_normalizer.normalize(state)
            state_tensor = torch.as_tensor(
                normalized, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            q_values = self.q_net(state_tensor).squeeze(0)
            allowed = q_values[
                torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            ]
            return float(allowed.mean().item()), float(allowed.max().item())

    def _normalize_action_mask(self, action_mask):
        if action_mask is None:
            return np.ones(self.action_dim, dtype=bool)
        mask = np.asarray(action_mask, dtype=bool).reshape(-1)
        if mask.size != self.action_dim:
            raise ValueError(f"action_mask size {mask.size} != action_dim {self.action_dim}")
        if not np.any(mask):
            raise ValueError("action_mask must allow at least one action")
        return mask

    def parameter_hash(self):
        digest = hashlib.sha256()
        for name, tensor in sorted(self.q_net.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def _checkpoint_metadata(self):
        metadata = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "policy_contract_version": POLICY_CONTRACT_VERSION,
            "state_keys": list(STATE_KEYS),
            "action_names": [action["name"] for action in ACTIONS],
            "action_definitions": ACTIONS,
            "hidden_dims": list(self.hidden_dims),
            "gamma": self.config.gamma,
            "state_normalization": {
                "enabled": self.config.state_normalization,
                "warmup_steps": self.config.state_warmup_steps,
                "clip": self.config.state_clip,
            },
            "reward": {
                "clip": list(self.config.reward_clip),
                "normalize": self.config.reward_normalization,
                "warmup_steps": self.config.reward_warmup_steps,
            },
            "action_mask": {
                "enabled": self.config.action_mask_enabled,
                "thresholds": self.config.mask_thresholds,
            },
        }
        metadata["physics_contract"] = {
            "throughput_enhancer_version": THROUGHPUT_ENHANCER_VERSION,
            "heatsink_ownership_semantics_version": SINK_OWNERSHIP_SEMANTICS_VERSION,
            "initial_power_semantics_version": INITIAL_POWER_SEMANTICS_VERSION,
            "power_repair_semantics_version": POWER_REPAIR_SEMANTICS_VERSION,
            "environment": self.config.environment_contract,
        }
        metadata["config_hash"] = hashlib.sha256(
            json.dumps(metadata, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return metadata

    def save(self, path):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        metadata = self._checkpoint_metadata()
        torch.save({
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dims": self.hidden_dims,
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "gradient_steps": self.gradient_steps,
            "interaction_steps": self.interaction_steps,
            "last_loss": self.last_loss,
            "state_normalizer": self.state_normalizer.state_dict(),
            "reward_normalizer": self.reward_normalizer.state_dict(),
            "metadata": metadata,
        }, path)

    def load(self, path, map_location=None):
        checkpoint = torch.load(path, map_location=map_location or self.device)
        self._validate_checkpoint(checkpoint)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint.get("target_net", checkpoint["q_net"]))
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.gradient_steps = int(checkpoint.get("gradient_steps", checkpoint.get("steps", 0)))
        self.interaction_steps = int(checkpoint.get("interaction_steps", 0))
        self.last_loss = checkpoint.get("last_loss")
        if "state_normalizer" in checkpoint:
            self.state_normalizer.load_state_dict(checkpoint["state_normalizer"])
        if "reward_normalizer" in checkpoint:
            self.reward_normalizer.load_state_dict(checkpoint["reward_normalizer"])
        return self

    def set_evaluation_mode(self):
        self.q_net.eval()
        self.target_net.eval()
        self.state_normalizer.freeze()
        self.reward_normalizer.freeze()

    def _validate_checkpoint(self, checkpoint):
        metadata = checkpoint.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("DQN checkpoint schema is obsolete; retrain the DQN controller")
        if metadata.get("policy_contract_version") != POLICY_CONTRACT_VERSION:
            raise ValueError("DQN checkpoint policy contract version is obsolete")
        expected_metadata = self._checkpoint_metadata()
        if metadata.get("config_hash") != expected_metadata["config_hash"]:
            raise ValueError("DQN checkpoint policy contract is incompatible")
        if int(checkpoint.get("state_dim", -1)) != self.state_dim:
            raise ValueError("DQN checkpoint state_dim is incompatible")
        if int(checkpoint.get("action_dim", -1)) != self.action_dim:
            raise ValueError("DQN checkpoint action_dim is incompatible")
        metadata = checkpoint.get("metadata", {})
        expected_actions = [action["name"] for action in ACTIONS]
        if metadata.get("action_names", expected_actions) != expected_actions:
            raise ValueError("DQN checkpoint action ordering is incompatible")
        if metadata.get("state_keys", list(STATE_KEYS)) != list(STATE_KEYS):
            raise ValueError("DQN checkpoint state definition is incompatible")

    @classmethod
    def from_checkpoint(cls, path, config):
        if torch is None:
            raise ImportError("PyTorch is required for DQNAgent")
        checkpoint = torch.load(path, map_location="cpu")
        agent = cls(checkpoint["state_dim"], checkpoint["action_dim"], config)
        return agent.load(path)
