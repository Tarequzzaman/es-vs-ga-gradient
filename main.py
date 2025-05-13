#!/usr/bin/env python
"""
main.py
-------
Command-line entry point for the optimisation study.

• Compares CMA-ES, GA, and GD on 2-D Rastrigin
• Early-stop when fitness stalls (tol, patience)
• Optional 3-D visual demo + σ plot
• Exposes key hyper-parameters via CLI flags
"""

from __future__ import annotations

import argparse

import pandas as pd
from scipy.stats import mannwhitneyu

from benchmark import Rastrigin
from experiment import run_experiment
from optimizers import CMAES, GeneticAlgorithm, GradientDescent
from visualizer import grafic_rastrigin_gens, plot_sigma_history


# ------------------------------------------------------------------ #
def main() -> None:
    parser = argparse.ArgumentParser()
    # study-wide
    parser.add_argument("--runs", type=int, default=20, help="number of random seeds")
    parser.add_argument("--plots", type=int, default=40, help="gens to plot (0 = none)")
    # optimiser hyper-parameters
    parser.add_argument("--cma_sigma", type=float, default=0.5, help="initial σ for CMA-ES")
    parser.add_argument("--cma_lambda", type=int, default=50, help="population size λ for CMA-ES")
    parser.add_argument("--ga_pop", type=int, default=50, help="GA population size")
    parser.add_argument("--gd_lr", type=float, default=0.01, help="GD learning rate")
    parser.add_argument("--max_iter", type=int, default=2000, help="max iterations for ES / GA")
    args = parser.parse_args()

    tol, patience = 1e-6, 30
    rastrigin = Rastrigin(dim=2)

    # ─────────────────────────────────────────────────────────── #
    # optional visual demo
    if args.plots:
        demo_cma = CMAES(
            rastrigin, sigma=args.cma_sigma, lamb=args.cma_lambda, max_iter=args.max_iter
        )
        demo_gd = GradientDescent(
            rastrigin, lr=args.gd_lr, max_iter=2 * args.max_iter  # GD gets more steps
        )
        demo_ga = GeneticAlgorithm(rastrigin, pop_size=args.ga_pop, max_iter=args.max_iter)

        demo_cma.optimise(tol, patience)
        demo_gd.optimise(tol, patience)
        demo_ga.optimise(tol, patience)

        grafic_rastrigin_gens(
            rastrigin,
            demo_cma.population_history,
            demo_gd.position_history,
            demo_ga.population_history,
            max_gens=args.plots,
        )
        plot_sigma_history(demo_cma.sigma_history)

    # ─────────────────────────────────────────────────────────── #
    # statistical experiment
    df = run_experiment(
        rastrigin,
        runs=args.runs,
        tol=tol,
        patience=patience,
        gd_kwargs=dict(lr=args.gd_lr, max_iter=2 * args.max_iter),
        cma_kwargs=dict(sigma=args.cma_sigma, lamb=args.cma_lambda, max_iter=args.max_iter),
        ga_kwargs=dict(pop_size=args.ga_pop, max_iter=args.max_iter),
    )

    summary = pd.DataFrame(
        {
            "Algorithm": ["CMA-ES", "Genetic Alg", "Gradient Descent"],
            "Median best f": [
                df["CMA_best"].median(),
                df["GA_best"].median(),
                df["GD_best"].median(),
            ],
            "Median steps": [
                df["CMA_steps"].median(),
                df["GA_steps"].median(),
                df["GD_steps"].median(),
            ],
            "Median diversity": [
                df["CMA_diversity"].median(),
                df["GA_diversity"].median(),
                None,
            ],
            "Median σ_final": [df["CMA_sigma_last"].median(), None, None],
            "Success-rate (f≤tol)": [
                (df["CMA_best"] <= tol).mean(),
                (df["GA_best"] <= tol).mean(),
                (df["GD_best"] <= tol).mean(),
            ],
        }
    )

    print("\n===== Detailed runs =====")
    print(df.to_string(index=False, float_format=lambda x: f"{x: .4g}"))

    print("\n===== Summary (medians over runs) =====")
    print(summary.to_string(index=False, float_format=lambda x: f"{x: .4g}"))

    for a, b in [("CMA_best", "GA_best"), ("CMA_best", "GD_best")]:
        stat, p = mannwhitneyu(df[a], df[b])
        print(f"U({a[:3]},{b[:3]}) p={p:.3g}")


# ------------------------------------------------------------------ #
if __name__ == "__main__":
    main()
