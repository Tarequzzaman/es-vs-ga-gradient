"""
Uniform-crossover GA with Gaussian mutation.
Adds diversity history collection.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .base import BaseOptimizer


def mean_pairwise_distance(pop: np.ndarray) -> float:
    if len(pop) < 2:
        return 0.0
    dists = np.linalg.norm(pop[:, None] - pop[None, :], axis=-1)
    return dists[np.triu_indices(len(pop), k=1)].mean()


class GeneticAlgorithm(BaseOptimizer):
    """Simple GA with elitism + tournament selection."""

    def __init__(
        self,
        problem,
        pop_size: int = 50,
        mut_rate: float = 0.1,
        elite: int = 2,
        tournament_k: int = 3,
        max_iter: int = 1_000,
    ) -> None:
        super().__init__(problem, max_iter)
        self.pop_size, self.mut_rate, self.elite, self.k = pop_size, mut_rate, elite, tournament_k
        self.dim, self.bound = problem.dim, problem.bound

        self.population_history: List[np.ndarray] = []
        self.diversity_history: List[float] = []

    # ------------------------------------------------------------------ #
    def _tournament(self, fitness: np.ndarray) -> int:
        idx = np.random.choice(len(fitness), self.k, replace=False)
        return idx[int(np.argmin(fitness[idx]))]

    # ------------------------------------------------------------------ #
    def optimise(self, tol: float = 1e-6, patience: int = 30):
        pop = np.random.uniform(-self.bound, self.bound, (self.pop_size, self.dim))
        stall = 0
        for gen in range(1, self.max_iter + 1):
            fit = np.array([self.problem(ind) for ind in pop])

            self.population_history.append(pop.copy())
            self.diversity_history.append(mean_pairwise_distance(pop))

            if fit.min() < self.best_f - tol:
                self.best_f, stall = float(fit.min()), 0
            else:
                stall += 1
            if stall >= patience or self.best_f < tol:
                return pop[int(np.argmin(fit))].copy(), self.best_f, gen

            new_pop = pop[np.argsort(fit)][: self.elite]
            mut_sigma = 0.1 * self.bound * (0.995**gen)

            while len(new_pop) < self.pop_size:
                p1, p2 = pop[self._tournament(fit)], pop[self._tournament(fit)]
                child = np.where(np.random.rand(self.dim) < 0.5, p1, p2)
                mask = np.random.rand(self.dim) < self.mut_rate
                child[mask] += np.random.randn(mask.sum()) * mut_sigma
                child = np.clip(child, -self.bound, self.bound)
                new_pop = np.vstack((new_pop, child))
            pop = new_pop
        return pop[np.argmin([self.problem(ind) for ind in pop])], self.best_f, self.max_iter
