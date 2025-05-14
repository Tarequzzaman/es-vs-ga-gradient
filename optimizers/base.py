"""
Base class that unifies the optimiser interface / shared state.
Every concrete optimiser must inherit from ‟BaseOptimizer”.
"""

from __future__ import annotations          # allow forward-referencing of types (PEP 563)

from abc import ABC, abstractmethod         # ABC = Abstract Base Class machinery
from typing import Tuple                    # Tuple type-hint alias
import numpy as np                           # NumPy array maths

# ──────────────────────────────────────────────────────────────────────────
# Forward reference to avoid a runtime import cycle:
# When TYPE_CHECKING is True (static type-check phase) we can safely import
# Rastrigin for hints; at runtime the block is skipped.
# ──────────────────────────────────────────────────────────────────────────
from typing import TYPE_CHECKING             # special constant set by typing tools
if TYPE_CHECKING:                            # executed only by type-checkers / linters
    from benchmark import Rastrigin          # referenced purely for static typing

# ──────────────────────────────────────────────────────────────────────────
class BaseOptimizer(ABC):                    # abstract parent for all optimisers
    """
    Contract:
      • constructor saves a problem instance and iteration cap
      • .optimise() must return (best_x, best_f, steps)
    """

    def __init__(self, problem: "Rastrigin", max_iter: int) -> None:
        self.problem = problem               # objective function to minimise
        self.max_iter = max_iter             # hard stop after this many iterations
        self.best_f: float = float("inf")    # running best fitness (start at +∞)
        self.best_x: np.ndarray | None = None  # solution vector achieving best_f

    # ------------------------------------------------------------------ #
    @abstractmethod                          # subclasses MUST override this
    def optimise(
        self,
        tol: float,                          # minimum improvement to reset patience
        patience: int                        # early-stop after this many non-improving steps
    ) -> Tuple[np.ndarray | None, float, int]:
        """
        Execute the optimiser’s main loop.

        Returns
        -------
        best_x : ndarray | None
            Coordinate vector of the best solution found.
        best_f : float
            Objective value of `best_x`.
        steps  : int
            Iterations (or generations) actually performed.
        """
        ...
