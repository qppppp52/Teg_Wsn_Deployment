from src.dqn.reward_function import compute_reward


def test_dqn_reward_components_positive_for_cv_fr_hv_improvement():
    before = {
        "CV_mean": 4.0,
        "FR": 0.2,
        "HV": 0.1,
        "best_coverage": 0.4,
        "best_rsum_norm": 0.3,
        "diversity": 0.1,
        "mean_repair_iter": 3.0,
        "pressure": {"deploy": 0.1, "link": 0.4, "capacity": 0.2, "energy": 0.5, "sink": 0.1, "service": 0.2},
    }
    after = {
        "CV_mean": 2.0,
        "FR": 0.6,
        "HV": 0.3,
        "best_coverage": 0.5,
        "best_rsum_norm": 0.4,
        "diversity": 0.15,
        "mean_repair_iter": 2.0,
        "pressure": {"deploy": 0.05, "link": 0.2, "capacity": 0.1, "energy": 0.25, "sink": 0.05, "service": 0.1},
    }
    reward, parts = compute_reward(before, after, max_repair_iter=5)
    assert reward > 0
    assert parts["R_CV"] > 0
    assert parts["R_FR"] > 0
    assert parts["R_HV"] > 0


def test_dqn_reward_near_zero_or_cost_penalized_when_stable_feasible():
    metrics = {
        "CV_mean": 0.0,
        "FR": 1.0,
        "HV": 0.5,
        "best_coverage": 0.8,
        "best_rsum_norm": 0.8,
        "diversity": 0.2,
        "mean_repair_iter": 1.0,
        "pressure": {"deploy": 0.0, "link": 0.0, "capacity": 0.0, "energy": 0.0, "sink": 0.0, "service": 0.0},
    }
    reward, parts = compute_reward(metrics, metrics, max_repair_iter=5)
    assert reward <= 0.0
    assert parts["R_cost"] > 0
