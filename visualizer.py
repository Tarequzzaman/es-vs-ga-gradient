"""
3-D wire-frame visualiser for optimiser trajectories.
Also includes a helper plot for σ adaptation.
"""

from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (keeps mpl happy)
import matplotlib.pyplot as plt


# ------------------------------------------------------------------ #
def grafic_rastrigin_gens(
    rastrigin,
    cma_hist: List[np.ndarray],
    gd_hist: List[np.ndarray],
    ga_hist: List[np.ndarray] | None = None,
    max_gens: int = 10,
    mesh_step: float = 0.25,
) -> None:
    b = rastrigin.bound
    X, Y = np.mgrid[-b : b : mesh_step, -b : b : mesh_step]
    Z = 10 * 2 + (X**2 - 10 * np.cos(2 * np.pi * X)) + (Y**2 - 10 * np.cos(2 * np.pi * Y))

    for g in range(min(max_gens, len(cma_hist))):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(f"Population {g + 1}")
        ax.plot_wireframe(X, Y, Z, linewidth=0.4, color="grey", alpha=0.7)

        pop = cma_hist[g]
        ax.scatter(pop[:, 0], pop[:, 1], [rastrigin(p) for p in pop], c="red", s=18, label="CMA-ES")

        if ga_hist and g < len(ga_hist):
            gpop = ga_hist[g]
            ax.scatter(
                gpop[:, 0], gpop[:, 1], [rastrigin(p) for p in gpop], c="green", marker="x", s=28, label="GA"
            )

        if g < len(gd_hist):
            gd_pt = gd_hist[g]
            ax.scatter(gd_pt[0], gd_pt[1], rastrigin(gd_pt), marker="^", c="blue", s=40, label="GD")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("f(x,y)")
        ax.legend()
        plt.tight_layout()
        plt.show()


# ------------------------------------------------------------------ #
def plot_sigma_history(sigmas: List[float]):
    """Quick line plot of σ adaptation over generations."""

    plt.figure()
    plt.plot(sigmas)
    plt.title("CMA-ES step-size σ")
    plt.xlabel("generation")
    plt.ylabel("σ")
    plt.tight_layout()
    plt.show()
