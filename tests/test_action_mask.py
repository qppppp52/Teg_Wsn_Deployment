import numpy as np

from src.dqn.action_mask import build_dqn_action_mask
from src.dqn.action_space import ACTIONS
from src.rl_init.env import InitDeploymentEnv
from src.rl_init.action_space import ROLE_STOP
from tests.rl_init_test_utils import make_dummy_ctx


def test_dqn_action_mask_falls_back_when_everything_would_be_blocked():
    tiny_actions = [{"id": 0, "name": "throughput_priority"}]
    mask = build_dqn_action_mask({"FR": 0.0, "CV_mean": 10.0}, tiny_actions, {"dqn": {"action_mask_enabled": True}})
    assert mask.shape == (1,)
    assert bool(mask[0])


def test_rl_init_stop_action_masked_until_minimum_selection():
    ctx = make_dummy_ctx()
    env = InitDeploymentEnv(ctx, ctx.config)
    env.reset(seed=1)
    mask = env.get_action_mask()
    assert not bool(mask[0, ROLE_STOP])
