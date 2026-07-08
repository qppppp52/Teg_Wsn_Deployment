from src.rl_init.env import InitDeploymentEnv
from src.rl_init.action_space import ROLE_SENSOR, ROLE_STOP
from tests.rl_init_test_utils import make_dummy_ctx


def test_rl_init_env_invalid_stop_before_minimum_nodes():
    ctx = make_dummy_ctx()
    env = InitDeploymentEnv(ctx, ctx.config)
    env.reset(seed=1)
    _, reward, done, info = env.step({"candidate_id": 0, "role": ROLE_STOP})
    assert info["invalid_action"] is True
    assert done is False
    assert reward < 0


def test_rl_init_env_selects_sensor():
    ctx = make_dummy_ctx()
    env = InitDeploymentEnv(ctx, ctx.config)
    env.reset(seed=1)
    _, _, _, info = env.step({"candidate_id": 0, "role": ROLE_SENSOR})
    assert not info["invalid_action"]
    assert 0 in env.selected_sensors
