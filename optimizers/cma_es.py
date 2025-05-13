"""
Covariance-Matrix Adaptation Evolution Strategy (μ/λ, –C variant).
Now logs diversity & σ history for analysis.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from .base import BaseOptimizer


# ------------------------------------------------------------------ #
def mean_pairwise_distance(pop: np.ndarray) -> float:
    """Average Euclidean distance between all pairs (upper-triangular)."""
    if len(pop) < 2:
        return 0.0
    dists = np.linalg.norm(pop[:, None] - pop[None, :], axis=-1)
    return dists[np.triu_indices(len(pop), k=1)].mean()


# ------------------------------------------------------------------ #
class CMAES(BaseOptimizer):
    """CMA-ES with standard Hansen-Ostermeier parameterisation."""

    def __init__(
        self,
        problem,
        sigma: float = 0.5,
        lamb: int = 50,
        max_iter: int = 1_000,
    ) -> None:
        super().__init__(problem, max_iter)

        self.dim, self.sigma, self.lamb = problem.dim, sigma, lamb
        self.mu = lamb // 2

        self.centroid = np.random.uniform(-problem.bound, problem.bound, self.dim)

        self.C = np.eye(self.dim)
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)
        self.chiN = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim**2))

        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = w / np.sum(w)
        self.mueff = 1 / np.sum(self.weights**2)

        self.cc = 4 / (self.dim + 4)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 3)
        self.ccov1 = 2 / ((self.dim + 1.3) ** 2 + self.mueff)
        self.ccovmu = 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.dim + 2) ** 2 + self.mueff)
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs

        # histories for analysis / plotting
        self.population_history: List[np.ndarray] = []
        self.diversity_history: List[float] = []
        self.sigma_history: List[float] = []

    # -------------------------------------------------------------- #
    def _sample(self) -> np.ndarray:
        eigvals, eigvecs = np.linalg.eigh(self.C)
        sqrtC = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 1e-20))) @ eigvecs.T
        return self.centroid + self.sigma * np.random.randn(self.lamb, self.dim) @ sqrtC.T

    # -------------------------------------------------------------- #
    def _update(self, pop: np.ndarray, fit: np.ndarray) -> None:
        idx = np.argsort(fit)
        pop = pop[idx]
        old = self.centroid.copy()

        self.centroid = pop[: self.mu].T @ self.weights
        y = (self.centroid - old) / self.sigma

        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * y
        hsig = (
            np.linalg.norm(self.ps)
            / np.sqrt(1 - (1 - self.cs) ** (2 * len(self.population_history)))
            / self.chiN
            < 1.4 + 2 / (self.dim + 1)
        )
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y

        artmp = (pop[: self.mu] - old) / self.sigma
        self.C = (
            (1 - self.ccov1 - self.ccovmu) * self.C
            + self.ccov1 * np.outer(self.pc, self.pc)
            + self.ccovmu * (self.weights[:, None] * artmp).T @ artmp
        )

        self.sigma *= math.exp((np.linalg.norm(self.ps) / self.chiN - 1) * self.cs / self.damps)

    # -------------------------------------------------------------- #
    def optimise(self, tol: float = 1e-6, patience: int = 30):
        stall = 0
        for gen in range(1, self.max_iter + 1):
            pop = self._sample()
            fit = np.array([self.problem(p) for p in pop])

            # log histories --------------------------------------------------
            self.population_history.append(pop.copy())
            self.diversity_history.append(mean_pairwise_distance(pop))
            self.sigma_history.append(self.sigma)

            best = int(np.argmin(fit))
            if fit[best] < self.best_f - tol:
                self.best_f, self.best_x, stall = float(fit[best]), pop[best].copy(), 0
            else:
                stall += 1
            if stall >= patience or self.best_f < tol:
                return self.best_x, self.best_f, gen

            self._update(pop, fit)
        return self.best_x, self.best_f, self.max_iter
