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


def test_rl_init_env_terminal_info_contains_training_metrics(monkeypatch):
    def fake_terminal_reward(env, reward_cfg, norm_cfg):
        return 0.5, {
            "feasible": True,
            "cv": 0.0,
            "cv_deploy": 0.0,
            "cv_link": 0.0,
            "cv_capacity": 0.0,
            "cv_energy": 0.0,
            "cv_sink": 0.0,
            "cv_service": 0.0,
            "coverage": 0.5,
            "rsum": 1.0,
            "rsum_actual": 1.0,
            "rsum_capacity": 2.0,
            "repair_iter": 0,
            "repair_success": True,
        }

    monkeypatch.setattr("src.rl_init.env.compute_terminal_reward", fake_terminal_reward)
    ctx = make_dummy_ctx()
    env = InitDeploymentEnv(ctx, ctx.config)
    env.reset(seed=1)
    done = False
    info = {}
    for _ in range(20):
        _, _, done, info = env.step({"candidate_id": 0, "role": ROLE_STOP})
        if done:
            break
    assert done is True
    for key in [
        "terminal_reward", "step_reward_sum", "feasible", "cv", "cv_deploy", "cv_link",
        "cv_capacity", "cv_energy", "cv_sink", "cv_service", "coverage", "rsum",
        "rsum_actual", "rsum_capacity", "repair_iter", "repair_success", "num_sensors",
        "num_aps", "invalid_action_count",
    ]:
        assert key in info

