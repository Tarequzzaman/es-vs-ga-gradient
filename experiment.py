"""
Utility that executes multiple random-seed runs and returns a DataFrame.
Includes diversity + σ metrics for richer analysis.
"""

from __future__ import annotations

import random
from typing import Dict

import numpy as np
import pandas as pd

from benchmark import Rastrigin
from optimizers import CMAES, GeneticAlgorithm, GradientDescent


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
    gd_kwargs = gd_kwargs or {}
    cma_kwargs = cma_kwargs or {}
    ga_kwargs = ga_kwargs or {}

    rows = []
    for r in range(runs):
        np.random.seed(r)
        random.seed(r)

        gd = GradientDescent(prob, **gd_kwargs)
        _, gd_best, gd_steps = gd.optimise(tol, patience)

        cma = CMAES(prob, **cma_kwargs)
        _, cma_best, cma_steps = cma.optimise(tol, patience)

        ga = GeneticAlgorithm(prob, **ga_kwargs)
        _, ga_best, ga_steps = ga.optimise(tol, patience)

        rows.append(
            dict(
                Run=r + 1,
                # GD
                GD_best=gd_best,
                GD_steps=gd_steps,
                # CMA-ES
                CMA_best=cma_best,
                CMA_steps=cma_steps,
                CMA_diversity=float(np.mean(cma.diversity_history)),
                CMA_sigma_last=cma.sigma_history[-1],
                # GA
                GA_best=ga_best,
                GA_steps=ga_steps,
                GA_diversity=float(np.mean(ga.diversity_history)),
            )
        )
    return pd.DataFrame(rows)
