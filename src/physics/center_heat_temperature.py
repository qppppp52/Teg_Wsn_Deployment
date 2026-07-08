"""Compatibility wrapper for the center heat-source wall-temperature model."""
from src.scene.center_heat_temperature import (
    compute_center_air_convection_temperature,
    compute_center_air_convection_temperature_field,
)

__all__ = [
    "compute_center_air_convection_temperature",
    "compute_center_air_convection_temperature_field",
]
