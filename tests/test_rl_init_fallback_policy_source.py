import builtins

from src.rl_init.population_generator import DRLInitPopulationGenerator
from tests.rl_init_test_utils import make_dummy_ctx


def test_fallback_policy_source_when_pytorch_trainer_unavailable(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "src.rl_init.ppo_trainer":
            raise ImportError("simulated torch unavailable")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    ctx = make_dummy_ctx()
    generator = DRLInitPopulationGenerator(ctx, ctx.config)
    rows = generator.train()
    assert generator.policy_source == "fallback"
    assert generator.torch_available is False
    assert generator.checkpoint_skip_reason == "torch_unavailable"
    assert rows[0]["policy_source"] == "fallback"
    assert rows[0]["fallback_reason"] == "torch_unavailable"
