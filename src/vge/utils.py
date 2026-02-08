"""
Utility functions for Variance-Gated Ensembles.

Contains:
- device selection, random seed setting, and early stopping.

Reference: "Variance-Gated Ensembles: An Epistemic-Aware Framework for
Uncertainty Estimation"
"""

from __future__ import annotations

import random
from typing import Literal

import numpy as np
import torch


def select_device() -> Literal["cuda", "mps", "cpu"]:
    """
    Automatically select the best available device.

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seeds(seed: int = 42, deterministic: bool = False) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
        deterministic: If True, use deterministic algorithms (slower but reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        # Faster but not strictly reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


class EarlyStopping:
    """
    Early stopping utility for training.

    Monitors a metric (e.g., validation loss) and signals when to stop
    training if no improvement is seen for a specified number of epochs.

    Args:
        min_delta: Minimum change to qualify as an improvement
        patience: Number of epochs to wait before stopping
    """

    def __init__(self, min_delta: float = 1.0e-4, patience: int = 10) -> None:
        self.min_delta = min_delta
        self.patience = patience
        self.best: float = float("inf")
        self.wait: int = 0
        self.stop: bool = False

    def step(self, current: float) -> bool:
        """
        Check if training should stop.

        Args:
            current: Current metric value (e.g., validation loss)

        Returns:
            True if training should stop, False otherwise
        """
        if current < self.best - self.min_delta:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop = True

        return self.stop

    def reset(self) -> None:
        """Reset early stopping state."""
        self.best = float("inf")
        self.wait = 0
        self.stop = False
