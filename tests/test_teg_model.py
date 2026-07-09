import numpy as np

from src.physics.teg_model import compute_delta_T_thermal_resistance


def test_thermal_resistance_delta_t_uses_teg_resistance_divider():
    config = {
        "discretization": {"grid_spacing": 0.5},
        "teg": {
            "thermal_resistance": {
                "T_amb": 295.15,
                "R_TEG": 0.12,
                "h_sink": 50.0,
            }
        },
    }
    t_wall = np.array([305.15])
    r_sink = 1.0 / (50.0 * 0.5 * 0.5)
    expected = (t_wall - 295.15) * 0.12 / (0.12 + r_sink)

    actual = compute_delta_T_thermal_resistance(t_wall, config)

    np.testing.assert_allclose(actual, expected)
