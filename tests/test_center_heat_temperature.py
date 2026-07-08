import numpy as np
from src.scene.enclosed_space import EnclosedSpace
from src.scene.surface_discretizer import discretize_surface
from src.scene.center_heat_temperature import compute_center_air_convection_temperature_field


def _config():
    return {
        "space": {"Lx": 3.0, "Ly": 3.0, "Lz": 3.0},
        "temperature": {
            "model": "center_air_convection_synthetic",
            "ambient": 295.15,
            "source_temperature": 343.15,
            "source_position": [1.5, 1.5, 1.5],
            "decay_length": 1.15,
            "buoyancy_strength": 0.35,
            "face_heat_multipliers": {"Top": 1.18, "Bottom": 0.78, "Left": 0.92, "Right": 1.03, "Front": 1.08, "Back": 0.88},
            "plume_direction": [0.25, 0.10, 1.0],
            "plume_strength": 0.18,
            "min_wall_temperature": 296.15,
            "max_wall_temperature": 343.15,
            "min_surface_temp_std": 1.5,
            "min_surface_temp_range": 5.0,
        },
    }


def test_center_heat_temperature_bounds_and_nonuniformity():
    space = EnclosedSpace(3.0, 3.0, 3.0)
    pts = discretize_surface(space, 0.5)
    T = compute_center_air_convection_temperature_field(pts, space, _config())
    assert np.all(T >= 295.15)
    assert np.all(T <= 343.15)
    assert np.std(T) >= 1.5
    assert np.max(T) - np.min(T) >= 5.0
    top = T[pts[:, 3].astype(int) == 0].mean()
    bottom = T[pts[:, 3].astype(int) == 1].mean()
    assert top > bottom
