"""
optimizers/__init__.py
──────────────────────
Package facade: collects the four concrete optimiser classes so users can
import them with a single, tidy statement:

    from optimizers import CMAES, GeneticAlgorithm, GradientDescent
"""

# ──────────────────────────────────────────────────────────────────────
# Re-export concrete classes from their sub-modules
# ──────────────────────────────────────────────────────────────────────
from .base import BaseOptimizer            # abstract parent (interface only)
from .gradient_descent import GradientDescent  # finite-difference GD
from .cma_es import CMAES                      # Covariance-Matrix Adaptation ES
from .genetic_algorithm import GeneticAlgorithm  # uniform-crossover GA

# ──────────────────────────────────────────────────────────────────────
# __all__  controls  “from optimizers import *”
# Only the names listed here will be imported in that case, keeping the
# public surface clean and preventing accidental leakage of helpers.
# ──────────────────────────────────────────────────────────────────────
__all__: list[str] = [
    "BaseOptimizer",
    "GradientDescent",
    "CMAES",
    "GeneticAlgorithm",
]
