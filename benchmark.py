"""
benchmark.py
------------
Single benchmark function used throughout the experiments.
"""

from __future__ import annotations
import numpy as np


class Rastrigin:
    """Multi-modal benchmark; global minimum is f(0)=0."""

    def __init__(self, dim: int = 2, bound: float = 5.12) -> None:
        self.dim, self.bound = dim, bound

    # make the instance directly callable as f(x)
    def __call__(self, x: np.ndarray) -> float:  # noqa: D401
        return 10 * self.dim + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))