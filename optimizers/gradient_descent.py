"""
Finite-difference gradient-descent optimiser.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .base import BaseOptimizer


class GradientDescent(BaseOptimizer):
    """Constant-learning-rate GD."""

    def __init__(self, problem, lr: float = 0.01, max_iter: int = 2_000) -> None:
        super().__init__(problem, max_iter)
        self.lr = lr
        self.x = np.random.uniform(-problem.bound, problem.bound, problem.dim)
        self.position_history: List[np.ndarray] = []

    # ------------------------------------------------------------------ #
    def _grad(self, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
        g = np.zeros_like(x)
        fx = self.problem(x)
        for i in range(len(x)):
            xh = x.copy()
            xh[i] += h
            g[i] = (self.problem(xh) - fx) / h
        return g

    # ------------------------------------------------------------------ #
    def optimise(
        self, tol: float = 1e-6, patience: int = 50
    ) -> Tuple[np.ndarray, float, int]:
        stall = 0
        for it in range(1, self.max_iter + 1):
            self.x -= self.lr * self._grad(self.x)
            self.x = np.clip(self.x, -self.problem.bound, self.problem.bound)
            f = self.problem(self.x)
            self.position_history.append(self.x.copy())

            if f < self.best_f - tol:
                self.best_f, self.best_x, stall = f, self.x.copy(), 0
            else:
                stall += 1
            if stall >= patience:
                return self.best_x, self.best_f, it
        return self.best_x, self.best_f, self.max_iter
