"""
Base class that unifies the optimiser interface / shared state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

# forward reference to avoid import cycle; only used for typing
from typing import TYPE_CHECKING
if TYPE_CHECKING:  # pragma: no cover
    from benchmark import Rastrigin


class BaseOptimizer(ABC):
    """
    All optimisers must return ``best_x, best_f, steps`` from ``optimise``.
    """

    def __init__(self, problem: "Rastrigin", max_iter: int) -> None:
        self.problem = problem
        self.max_iter = max_iter
        self.best_f: float = float("inf")
        self.best_x: np.ndarray | None = None

    # --------------------------------------------------------------------- #
    @abstractmethod
    def optimise(
        self, tol: float, patience: int
    ) -> Tuple[np.ndarray | None, float, int]:
        ...
