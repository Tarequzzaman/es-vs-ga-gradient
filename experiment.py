"""
Utility that executes multiple random-seed runs and returns a DataFrame.
Includes diversity + σ metrics for richer analysis.
"""

from __future__ import annotations                      # allow postponed type hints

import random                                            # for seeding Python's RNG
from typing import Dict                                  # type for flexible kwargs

import numpy as np                                       # numerical operations
import pandas as pd                                      # tabular result aggregation

from benchmark import Rastrigin                          # benchmark function
from optimizers import CMAES, GeneticAlgorithm, GradientDescent  # optimisers


# ------------------------------------------------------------------ #
def run_experiment(
    prob: Rastrigin,
    runs: int,
    tol: float,
    patience: int,
    *,
    gd_kwargs: Dict | None = None,
    cma_kwargs: Dict | None = None,
    ga_kwargs: Dict | None = None,
) -> pd.DataFrame:
    """
    Run CMA-ES, GA, and GD over multiple random seeds, collecting performance metrics.

    Parameters
    ----------
    prob : Rastrigin
        The benchmark problem instance.
    runs : int
        Number of independent random-seed runs.
    tol : float
        Minimum improvement threshold for early-stop.
    patience : int
        Maximum consecutive non-improving steps allowed before stopping.
    gd_kwargs : dict, optional
        Additional keyword arguments for Gradient Descent.
    cma_kwargs : dict, optional
        Additional keyword arguments for CMA-ES.
    ga_kwargs : dict, optional
        Additional keyword arguments for Genetic Algorithm.

    Returns
    -------
    pd.DataFrame
        One row per run, with columns including best fitness, step counts,
        CMA-ES diversity, and σ history.
    """
    gd_kwargs = gd_kwargs or {}                            # fallback to empty dict
    cma_kwargs = cma_kwargs or {}                          # fallback to empty dict
    ga_kwargs = ga_kwargs or {}                            # fallback to empty dict

    rows = []                                               # collect run results

    for r in range(runs):                                   # repeat for each seed
        np.random.seed(r)                                   # seed NumPy RNG
        random.seed(r)                                      # seed Python RNG

        # ─ Gradient Descent ─
        gd = GradientDescent(prob, **gd_kwargs)             # create GD instance with kwargs
        _, gd_best, gd_steps = gd.optimise(tol, patience)   # run optimiser

        # ─ CMA-ES ─
        cma = CMAES(prob, **cma_kwargs)                     # create CMA-ES instance
        _, cma_best, cma_steps = cma.optimise(tol, patience)

        # ─ Genetic Algorithm ─
        ga = GeneticAlgorithm(prob, **ga_kwargs)            # create GA instance
        _, ga_best, ga_steps = ga.optimise(tol, patience)

        # ─ Collect all stats into a dictionary row ─
        rows.append(
            dict(
                Run=r + 1,                                  # 1-based run index

                # Gradient Descent
                GD_best=gd_best,                            # best fitness achieved
                GD_steps=gd_steps,                          # total steps used

                # CMA-ES
                CMA_best=cma_best,                          # best fitness
                CMA_steps=cma_steps,                        # steps used
                CMA_diversity=float(np.mean(cma.diversity_history)),  # mean diversity across generations
                CMA_sigma_last=cma.sigma_history[-1],       # final σ value

                # Genetic Algorithm
                GA_best=ga_best,                            # best GA fitness
                GA_steps=ga_steps,                          # steps used
                GA_diversity=float(np.mean(ga.diversity_history)),    # mean pairwise distance
            )
        )

    return pd.DataFrame(rows)                               # return all results as a table
