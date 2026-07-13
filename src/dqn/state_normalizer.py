"""Running normalizers used by DQN training and frozen evaluation."""
from __future__ import annotations

import numpy as np


class RunningNormalizer:
    def __init__(self, shape, enabled=True, warmup_steps=0, clip=5.0):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.m2 = np.zeros(shape, dtype=np.float64)
        self.count = 0
        self.enabled = bool(enabled)
        self.warmup_steps = int(warmup_steps)
        self.clip = float(clip)
        self.frozen = False

    def update(self, value):
        if self.frozen or not self.enabled:
            return
        value = np.asarray(value, dtype=np.float64)
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    def normalize(self, value):
        value = np.asarray(value, dtype=np.float32)
        if not self.enabled or self.count < max(self.warmup_steps, 2):
            return value
        variance = self.m2 / max(self.count - 1, 1)
        normalized = (value - self.mean) / np.sqrt(variance + 1.0e-8)
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)

    def freeze(self):
        self.frozen = True

    def state_dict(self):
        return {
            "mean": self.mean.tolist(),
            "m2": self.m2.tolist(),
            "count": self.count,
            "enabled": self.enabled,
            "warmup_steps": self.warmup_steps,
            "clip": self.clip,
        }

    def load_state_dict(self, state):
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.m2 = np.asarray(state["m2"], dtype=np.float64)
        self.count = int(state["count"])
        self.enabled = bool(state.get("enabled", self.enabled))
        self.warmup_steps = int(state.get("warmup_steps", self.warmup_steps))
        self.clip = float(state.get("clip", self.clip))
        return self


class RunningScalarNormalizer(RunningNormalizer):
    def __init__(self, enabled=True, warmup_steps=0, clip=5.0):
        super().__init__((), enabled=enabled, warmup_steps=warmup_steps, clip=clip)

    def normalize_scalar(self, value):
        return float(self.normalize(float(value)))
