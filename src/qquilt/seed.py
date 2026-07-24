"""Deterministic RNG seeding: the single place every run gets its seeds from."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) from a single helper.

    Pass deterministic=True to also enable torch deterministic algorithms and
    the cuBLAS workspace config required for matmul reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # warn_only=True keeps a few non-deterministic transformers fast paths
        # working: a documented warning is preferable to a hard crash on those
        # ops (see ENGINEERING.md, "Determinism and pinning").
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
