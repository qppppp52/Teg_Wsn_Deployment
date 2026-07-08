import numpy as np
from src.rl_init.env import InitDeploymentEnv
from src.rl_init.action_space import ROLE_AP, ROLE_SENSOR, ROLE_STOP
from tests.rl_init_test_utils import make_dummy_ctx


def test_rl_init_action_mask_respects_candidate_roles_and_min_stop():
    ctx = make_dummy_ctx()
    env = InitDeploymentEnv(ctx, ctx.config)
    env.reset(seed=1)
    mask = env.get_action_mask()
    assert mask.shape == (ctx.num_candidates, 4)
    assert np.any(mask)
    assert not mask[0, ROLE_AP]
    assert mask[0, ROLE_SENSOR]
    assert not mask[0, ROLE_STOP]
