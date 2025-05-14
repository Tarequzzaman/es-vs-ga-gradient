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

from __future__ import annotations                          # Enable forward-referenced type hints

import argparse                                              # For parsing command-line arguments
import pandas as pd                                          # For result summarisation
from scipy.stats import mannwhitneyu                         # Non-parametric statistical test

# Local modules
from benchmark import Rastrigin                              # Objective function
from experiment import run_experiment                        # Batch runner
from optimizers import CMAES, GeneticAlgorithm, GradientDescent  # All three optimisers
from visualizer import grafic_rastrigin_gens, plot_sigma_history  # Visualisation helpers


def main() -> None:
    """
    Entry point for running optimisation experiments. Configurable via CLI.
    Runs statistical comparisons between optimisers and optionally shows 3D plots.
    """
    parser = argparse.ArgumentParser()                        # Create argument parser

    # Study-wide configuration
    parser.add_argument("--runs", type=int, default=20, help="number of random seeds")
    parser.add_argument("--plots", type=int, default=40, help="gens to plot (0 = none)")

    # Optimiser-specific hyper-parameters
    parser.add_argument("--cma_sigma", type=float, default=0.5, help="initial σ for CMA-ES")
    parser.add_argument("--cma_lambda", type=int, default=50, help="population size λ for CMA-ES")
    parser.add_argument("--ga_pop", type=int, default=50, help="GA population size")
    parser.add_argument("--gd_lr", type=float, default=0.01, help="GD learning rate")
    parser.add_argument("--max_iter", type=int, default=2000, help="max iterations for ES / GA")

    args = parser.parse_args()                                # Parse command-line arguments

    tol, patience = 1e-6, 30                                   # Convergence and early-stop settings
    rastrigin = Rastrigin(dim=2)                               # 2-D Rastrigin problem instance

    # Optional visual demonstration of optimiser trajectories
    if args.plots:
        # Create instances with user-specified hyper-parameters
        demo_cma = CMAES(
            rastrigin, sigma=args.cma_sigma, lamb=args.cma_lambda, max_iter=args.max_iter
        )
        demo_gd = GradientDescent(
            rastrigin, lr=args.gd_lr, max_iter=2 * args.max_iter  # GD gets extra steps
        )
        demo_ga = GeneticAlgorithm(
            rastrigin, pop_size=args.ga_pop, max_iter=args.max_iter
        )

        # Run each optimiser
        demo_cma.optimise(tol, patience)
        demo_gd.optimise(tol, patience)
        demo_ga.optimise(tol, patience)

        # Show 3D wireframe with population/trajectory overlays
        grafic_rastrigin_gens(
            rastrigin,
            demo_cma.population_history,
            demo_gd.position_history,
            demo_ga.population_history,
            max_gens=args.plots,
        )

        # Show how σ evolves across generations (CMA-ES only)
        plot_sigma_history(demo_cma.sigma_history)

    # ─────────────────────────────────────────────────────────── #
    # Run full statistical experiment over multiple seeds
    df = run_experiment(
        rastrigin,
        runs=args.runs,
        tol=tol,
        patience=patience,
        gd_kwargs=dict(lr=args.gd_lr, max_iter=2 * args.max_iter),  # GD gets more iterations
        cma_kwargs=dict(sigma=args.cma_sigma, lamb=args.cma_lambda, max_iter=args.max_iter),
        ga_kwargs=dict(pop_size=args.ga_pop, max_iter=args.max_iter),
    )

    # Create a summary table (medians and success rates)
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
                None,  # GD doesn't track diversity
            ],
            "Median σ_final": [df["CMA_sigma_last"].median(), None, None],  # σ is only relevant for CMA-ES
            "Success-rate (f≤tol)": [
                (df["CMA_best"] <= tol).mean(),
                (df["GA_best"] <= tol).mean(),
                (df["GD_best"] <= tol).mean(),
            ],
        }
    )

    # Print full per-run table
    print("\n===== Detailed runs =====")
    print(df.to_string(index=False, float_format=lambda x: f"{x: .4g}"))

    # Print summary statistics
    print("\n===== Summary (medians over runs) =====")
    print(summary.to_string(index=False, float_format=lambda x: f"{x: .4g}"))

    # Mann-Whitney U-tests: CMA vs GA, CMA vs GD
    for a, b in [("CMA_best", "GA_best"), ("CMA_best", "GD_best")]:
        stat, p = mannwhitneyu(df[a], df[b])                   # non-parametric test
        print(f"U({a[:3]},{b[:3]}) p={p:.3g}")                 # print p-value


if __name__ == "__main__":           # Run main() if executed directly
    main()
