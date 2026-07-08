"""Utilities for reproducible seeding across optional backends."""
import os
import random

import numpy as np


def set_seed(seed: int, include_torch: bool = False):
    """Set deterministic seeds for Python and NumPy, with optional torch support."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if not include_torch:
        return

    try:
        import torch
    except Exception:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)