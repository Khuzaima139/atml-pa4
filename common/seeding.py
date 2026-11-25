"""Deterministic seeding for Python, NumPy and PyTorch."""
import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> int:
    """Seed random, numpy and torch (CPU and every CUDA device if present); return the seed."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
