"""PPO actor-critic policy for candidate-role initialization."""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    from torch.distributions import Categorical
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    Categorical = None

from src.rl_init.action_space import NUM_ROLES, unflatten_action


class CandidateAttentionEncoder(nn.Module if nn is not None else object):
    def __init__(self, candidate_dim, global_dim, hidden_dim=128, dropout=0.1, num_heads=4):
        if nn is None:
            raise ImportError("PyTorch is required for CandidateAttentionEncoder")
        super().__init__()
        num_heads = max(1, int(num_heads))
        if hidden_dim % num_heads != 0:
            num_heads = 1
        self.candidate_mlp = nn.Sequential(
            nn.Linear(candidate_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.global_mlp = nn.Sequential(nn.Linear(global_dim, hidden_dim), nn.ReLU())
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.fuse = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU())

    def forward(self, candidate_features, global_features):
        cand = self.candidate_mlp(candidate_features)
        glob = self.global_mlp(global_features)
        if glob.dim() == 1:
            glob = glob.unsqueeze(0)
        query = glob.unsqueeze(1)
        attended_global, _ = self.attention(query=query, key=cand, value=cand, need_weights=False)
        context = attended_global.expand(-1, cand.shape[1], -1)
        fused = self.fuse(torch.cat([cand, context], dim=-1))
        pooled = fused.mean(dim=1)
        return fused, pooled


class InitActorCritic(nn.Module if nn is not None else object):
    def __init__(self, candidate_feature_dim, global_feature_dim, hidden_dim=128, dropout=0.1, attention_dim=None):
        if nn is None:
            raise ImportError("PyTorch is required for InitActorCritic")
        super().__init__()
        self.encoder = CandidateAttentionEncoder(candidate_feature_dim, global_feature_dim, hidden_dim, dropout)
        self.actor = nn.Linear(hidden_dim, NUM_ROLES)
        self.value_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, state):
        cand, glob, mask = _state_to_tensors(state, next(self.parameters()).device)
        emb, pooled = self.encoder(cand, glob)
        logits = self.actor(emb)
        masked_logits = logits.masked_fill(~mask, -1.0e9)
        value = self.value_head(pooled).squeeze(-1)
        return {"joint_logits": masked_logits, "value": value}

    def act(self, state, deterministic=False):
        out = self.forward(state)
        logits = out["joint_logits"].reshape(out["joint_logits"].shape[0], -1)
        dist = Categorical(logits=logits)
        action_id = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action_id)
        entropy = dist.entropy()
        action = unflatten_action(int(action_id[0].item()))
        return action, log_prob.squeeze(0), out["value"].squeeze(0), entropy.squeeze(0)

    def evaluate_actions(self, states, actions):
        if isinstance(states, dict):
            states = [states]
        log_probs = []
        values = []
        entropies = []
        for state, action_id in zip(states, actions):
            out = self.forward(state)
            logits = out["joint_logits"].reshape(1, -1)
            dist = Categorical(logits=logits)
            action_tensor = torch.as_tensor([int(action_id)], dtype=torch.long, device=logits.device)
            log_probs.append(dist.log_prob(action_tensor).squeeze(0))
            values.append(out["value"].squeeze(0))
            entropies.append(dist.entropy().squeeze(0))
        return torch.stack(log_probs), torch.stack(values), torch.stack(entropies).mean()


def _state_to_tensors(state, device):
    cand = torch.as_tensor(state["candidate_features"], dtype=torch.float32, device=device)
    glob = torch.as_tensor(state["global_features"], dtype=torch.float32, device=device)
    mask = torch.as_tensor(state["joint_action_mask"], dtype=torch.bool, device=device)
    if cand.dim() == 2:
        cand = cand.unsqueeze(0)
    if glob.dim() == 1:
        glob = glob.unsqueeze(0)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    return cand, glob, mask
