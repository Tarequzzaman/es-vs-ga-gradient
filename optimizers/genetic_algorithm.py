"""
Uniform-crossover Genetic Algorithm with Gaussian mutation.

Adds:
    • `population_history`  – list of λ×d arrays (one per generation)
    • `diversity_history`   – mean pairwise distance per generation
"""

from __future__ import annotations            # postpone evaluation of type hints (PEP 563)

from typing import List                       # List type alias for annotations
import numpy as np                            # NumPy numerical library

from .base import BaseOptimizer               # Abstract optimiser interface


# --------------------------------------------------------------------------- #
def mean_pairwise_distance(pop: np.ndarray) -> float:
    """
    Return the average Euclidean distance between every unordered pair of
    individuals in the population.

    Parameters
    ----------
    pop : np.ndarray
        Array of shape (λ, dim) representing the current population.

    Returns
    -------
    float
        Mean pairwise distance (0.0 if population has < 2 individuals).
    """
    if len(pop) < 2:                          # handle degenerate case (one or zero individuals)
        return 0.0                            # by definition no distance
    # compute full λ×λ distance matrix in a vectorised manner
    dists = np.linalg.norm(pop[:, None] - pop[None, :], axis=-1)
    # take the upper-triangular part (i < j) to avoid double-counting & zeros
    return dists[np.triu_indices(len(pop), k=1)].mean()


# --------------------------------------------------------------------------- #
class GeneticAlgorithm(BaseOptimizer):
    """Simple GA with elitism and k-tournament parent selection."""

    def __init__(                        # constructor: configure hyper-parameters
        self,
        problem,                         # objective function to minimise
        pop_size: int = 50,              # population size λ
        mut_rate: float = 0.1,           # per-gene mutation probability
        elite: int = 2,                  # # elite survivors copied unchanged
        tournament_k: int = 3,           # tournament size k
        max_iter: int = 1_000,           # generation cap
    ) -> None:
        super().__init__(problem, max_iter)      # initialise shared BaseOptimizer state
        # store GA parameters
        self.pop_size = pop_size                 # λ
        self.mut_rate = mut_rate                 # mutation probability
        self.elite = elite                       # # elites
        self.k = tournament_k                    # k-tournament size
        self.dim, self.bound = problem.dim, problem.bound  # dimension d and search bound

        # histories for post-run analysis
        self.population_history: List[np.ndarray] = []     # λ×d arrays
        self.diversity_history: List[float] = []           # diversity metric

    # --------------------------------------------------------------------- #
    def _tournament(self, fitness: np.ndarray) -> int:
        """
        Draw `k` random individuals without replacement, return the index of
        the one with the lowest (best) fitness.

        Parameters
        ----------
        fitness : np.ndarray
            1-D array of fitness values (lower = better).

        Returns
        -------
        int
            Index of selected parent.
        """
        idx = np.random.choice(len(fitness), self.k, replace=False)   # pick k contestants
        return idx[int(np.argmin(fitness[idx]))]                      # return best among them

    # --------------------------------------------------------------------- #
    def optimise(self, tol: float = 1e-6, patience: int = 30):
        """
        Run GA until:
          • best fitness improves by > `tol` within last `patience` generations,
          • or best fitness ≤ `tol`,
          • or `max_iter` generations reached.

        Parameters
        ----------
        tol : float
            Improvement threshold.
        patience : int
            Early-stop patience in generations.

        Returns
        -------
        tuple
            (best_x: np.ndarray, best_f: float, generations: int)
        """
        # initialise population uniformly inside search box [−bound, +bound]^d
        pop = np.random.uniform(-self.bound, self.bound,
                                (self.pop_size, self.dim))
        stall = 0                                                     # counter of non-improving generations

        for gen in range(1, self.max_iter + 1):                        # generation loop
            # evaluate objective for every individual
            fit = np.array([self.problem(ind) for ind in pop])

            # log current population & diversity
            self.population_history.append(pop.copy())
            self.diversity_history.append(mean_pairwise_distance(pop))

            # track improvement
            if fit.min() < self.best_f - tol:                          # improvement exceeds tol?
                self.best_f = float(fit.min())                         # store new best fitness
                stall = 0                                              # reset stall counter
            else:
                stall += 1                                             # increment stall

            # early-stop check
            if stall >= patience or self.best_f < tol:
                # return best individual, best fitness, generations executed
                return pop[int(np.argmin(fit))].copy(), self.best_f, gen

            # ─── create next generation ─────────────────────────────── #
            # elitism: copy top `elite` individuals
            new_pop = pop[np.argsort(fit)][: self.elite]

            # mutation scale decays geometrically
            mut_sigma = 0.1 * self.bound * (0.995 ** gen)

            # generate offspring until population is full
            while len(new_pop) < self.pop_size:
                # parent selection via tournament
                p1 = pop[self._tournament(fit)]
                p2 = pop[self._tournament(fit)]

                # uniform crossover (50-50 mask)
                child = np.where(np.random.rand(self.dim) < 0.5, p1, p2)

                # Gaussian mutation on masked genes
                mask = np.random.rand(self.dim) < self.mut_rate
                child[mask] += np.random.randn(mask.sum()) * mut_sigma

                # enforce search bounds
                child = np.clip(child, -self.bound, self.bound)

                # add offspring to new population
                new_pop = np.vstack((new_pop, child))

            pop = new_pop                                             # iterate: new pop becomes current pop

        # reached generation cap: return best individual in final pop
        best_idx = np.argmin([self.problem(ind) for ind in pop])
        return pop[best_idx], self.best_f, self.max_iter
