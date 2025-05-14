"""
benchmark.py
------------
Single benchmark function used throughout the experiments.
Implements the Rastrigin function: a multi-modal test case with many local minima.
"""

from __future__ import annotations              # enable forward-referenced type hints (PEP 563)
import numpy as np                              # import NumPy for numerical operations


class Rastrigin:
    """Multi-modal benchmark; global minimum is f(0) = 0."""

    def __init__(self, dim: int = 2, bound: float = 5.12) -> None:
        # Store the number of dimensions and the search box bound
        self.dim, self.bound = dim, bound

    def __call__(self, x: np.ndarray) -> float:  # noqa: D401
        """
        Evaluate the Rastrigin function at point x.

        Parameters
        ----------
        x : np.ndarray
            A real-valued vector of shape (dim,).

        Returns
        -------
        float
            The objective value at x.
        """
        # Rastrigin formula: 10·d + Σ [x_i² − 10·cos(2πx_i)]
        return 10 * self.dim + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
