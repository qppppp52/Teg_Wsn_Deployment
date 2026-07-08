import numpy as np
from src.rl_init.individual_builder import build_individual_from_orders
from tests.rl_init_test_utils import make_dummy_ctx


def test_rl_init_individual_builder_shapes_and_priority_order():
    ctx = make_dummy_ctx()
    ind = build_individual_from_orders(ctx, [2, 0], [4], priority_noise_std=0.0, rng=np.random.default_rng(1))
    assert ind.rho_s.shape == (ctx.index_mapping.num_sensor_candidates,)
    assert ind.rho_a.shape == (ctx.index_mapping.num_ap_candidates,)
    assert ind.rho_s[ctx.index_mapping.Ls_global_to_local[2]] > ind.rho_s[ctx.index_mapping.Ls_global_to_local[0]]
    assert 0.0 <= ind.rho_a.min() <= ind.rho_a.max() <= 1.0
