"""
optimizers package façade — re-export the concrete optimisers.
"""

from .base import BaseOptimizer
from .gradient_descent import GradientDescent
from .cma_es import CMAES
from .genetic_algorithm import GeneticAlgorithm

__all__ = [
    "BaseOptimizer",
    "GradientDescent",
    "CMAES",
    "GeneticAlgorithm",
]
