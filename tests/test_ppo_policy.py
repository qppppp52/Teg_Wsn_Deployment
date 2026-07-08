import pytest
from src.rl_init.env import InitDeploymentEnv
from tests.rl_init_test_utils import make_dummy_ctx


def test_ppo_policy_masked_forward_and_act():
    pytest.importorskip("torch")
    from src.rl_init.ppo_policy import InitActorCritic

    ctx = make_dummy_ctx()
    env = InitDeploymentEnv(ctx, ctx.config)
    state = env.reset(seed=1)
    policy = InitActorCritic(state["candidate_features"].shape[1], state["global_features"].shape[0], hidden_dim=16, dropout=0.0)
    out = policy.forward(state)
    assert tuple(out["joint_logits"].shape) == (1, ctx.num_candidates, 4)
    action, log_prob, value, entropy = policy.act(state, deterministic=True)
    assert "candidate_id" in action and "role" in action
    assert log_prob.ndim == 0
    assert value.ndim == 0
    assert entropy.ndim == 0
