"""Target point generation utilities."""
import numpy as np


def generate_random_targets(num_targets: int, Lx: float, Ly: float, Lz: float,
                            margin: float = 0.1,
                            seed: int | None = None) -> np.ndarray:
    """Generate random target points inside the enclosed space.

    A local RNG is used so experiment reproducibility is not broken by
    ``np.random.seed(None)`` or other global RNG side effects.
    """
    rng = np.random.default_rng(seed)
    targets = rng.uniform(
        low=[margin, margin, margin],
        high=[Lx - margin, Ly - margin, Lz - margin],
        size=(num_targets, 3)
    )
    return targets
