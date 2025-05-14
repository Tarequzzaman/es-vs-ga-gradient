"""
Covariance-Matrix Adaptation Evolution Strategy  (μ/λ, –C variant)
──────────────────────────────────────────────────────────────────
Adds logging of population diversity and step-size (σ) history for
post-run analysis / visualisation.
"""

from __future__ import annotations       # allow forward-referenced type hints

import math                              # exp, sqrt …
from typing import List                  # typing alias for clearer lists
import numpy as np                       # ndarray maths

from .base import BaseOptimizer          # abstract parent with shared API


# ──────────────────────────────────────────────────────────────────────
# Helper: average Euclidean distance between every unordered pair in pop
# ──────────────────────────────────────────────────────────────────────
def mean_pairwise_distance(pop: np.ndarray) -> float:
    """Return ⟨‖x_i − x_j‖⟩  for all i < j (O(λ²) but λ ≤ 100)."""
    if len(pop) < 2:
        return 0.0
    dists = np.linalg.norm(pop[:, None] - pop[None, :], axis=-1)     # full λ×λ matrix
    return dists[np.triu_indices(len(pop), k=1)].mean()              # upper-triangle mean


# ──────────────────────────────────────────────────────────────────────
class CMAES(BaseOptimizer):
    """CMA-ES using standard Hansen–Ostermeier hyper-parameters."""

    # ------------------------------------------------------------------
    def __init__(
        self,
        problem,                         # objective function (Rastrigin)
        sigma: float = 0.5,              # initial global step-size
        lamb: int = 50,                  # population size λ
        max_iter: int = 1_000,           # hard generation cap
    ) -> None:
        super().__init__(problem, max_iter)

        # store handy aliases
        self.dim, self.sigma, self.lamb = problem.dim, sigma, lamb
        self.mu = lamb // 2              # number of parents selected each gen

        # starting mean m₀ sampled uniformly inside search box
        self.centroid = np.random.uniform(-problem.bound, problem.bound, self.dim)

        # strategy parameters / evolution paths
        self.C  = np.eye(self.dim)                       # covariance matrix
        self.ps = np.zeros(self.dim)                     # σ-path
        self.pc = np.zeros(self.dim)                     # C-path
        self.chiN = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))

        # positive, normalised parent weights  w₁…w_μ
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = w / np.sum(w)
        self.mueff = 1 / np.sum(self.weights**2)         # variance-effective μ

        # time constants / learning rates (Hansen & Ostermeier, 2001)
        self.cc   = 4 / (self.dim + 4)
        self.cs   = (self.mueff + 2) / (self.dim + self.mueff + 3)
        self.ccov1  = 2 / ((self.dim + 1.3)**2 + self.mueff)
        self.ccovmu = 2 * (self.mueff - 2 + 1/self.mueff) / ((self.dim + 2)**2 + self.mueff)
        self.damps  = 1 + 2*max(0, np.sqrt((self.mueff-1)/(self.dim+1)) - 1) + self.cs

        # histories for later diagnostic plots
        self.population_history: List[np.ndarray] = []   # λ×d array per generation
        self.diversity_history:  List[float] = []        # mean pairwise distance
        self.sigma_history:      List[float] = []        # σ after each generation

    # ------------------------------------------------------------------
    def _sample(self) -> np.ndarray:
        """Draw λ offspring from 𝓝(m, σ²C)."""
        
        eigvals, eigvecs = np.linalg.eigh(self.C)                        # diagonalise C
        sqrtC = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 1e-20))) @ eigvecs.T
        return self.centroid + self.sigma * np.random.randn(self.lamb, self.dim) @ sqrtC.T

    # ------------------------------------------------------------------
    def _update(self, pop: np.ndarray, fit: np.ndarray) -> None:
        """Update mean, evolution paths, covariance, and σ."""
        idx = np.argsort(fit)                     # sort by fitness (asc.)
        pop = pop[idx]
        old = self.centroid.copy()                # m ← m′ bookkeeping

        # new mean m′
        self.centroid = pop[:self.mu].T @ self.weights
        y = (self.centroid - old) / self.sigma    # σ-scaled mean shift

        # update σ-path (ps)
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * y
        # Heaviside-like test for successful step
        hsig = (
            np.linalg.norm(self.ps)
            / np.sqrt(1 - (1 - self.cs) ** (2 * len(self.population_history)))
            / self.chiN
            < 1.4 + 2 / (self.dim + 1)
        )

        # update C-path (pc)
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y

        # rank-one + rank-μ covariance update
        artmp = (pop[:self.mu] - old) / self.sigma                       # σ-scaled selected steps
        self.C = (
            (1 - self.ccov1 - self.ccovmu) * self.C
            + self.ccov1 * np.outer(self.pc, self.pc)
            + self.ccovmu * (self.weights[:, None] * artmp).T @ artmp
        )

        # cumulative step-size adaptation
        self.sigma *= math.exp((np.linalg.norm(self.ps) / self.chiN - 1) * self.cs / self.damps)

    # ------------------------------------------------------------------
    def optimise(self, tol: float = 1e-6, patience: int = 30):
        """
        Run CMA-ES until
          • no improvement > tol for `patience` generations,  or
          • `best_f` ≤ tol,                                      or
          • `max_iter` reached.
        Returns (best_x, best_f, generations).
        """
        stall = 0                                      # gens since last improvement
        for gen in range(1, self.max_iter + 1):
            pop = self._sample()                       # λ offspring
            fit = np.array([self.problem(p) for p in pop])

            # ─── bookkeeping for later analysis ─────────────────────────
            self.population_history.append(pop.copy())
            self.diversity_history.append(mean_pairwise_distance(pop))
            self.sigma_history.append(self.sigma)

            # update best-so-far & stall counter
            best = int(np.argmin(fit))
            if fit[best] < self.best_f - tol:          # improvement large enough?
                self.best_f, self.best_x, stall = float(fit[best]), pop[best].copy(), 0
            else:
                stall += 1

            # early-stop tests
            if stall >= patience or self.best_f < tol:
                return self.best_x, self.best_f, gen

            # adapt search distribution for next generation
            self._update(pop, fit)

        # reached generation cap
        return self.best_x, self.best_f, self.max_iter
