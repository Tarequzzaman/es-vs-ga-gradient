"""
3-D wire-frame visualiser for optimiser trajectories.
Also includes a helper plot for σ adaptation.
"""

from __future__ import annotations                    # Enable postponed evaluation of annotations

from typing import List                                # For type annotations
import matplotlib.pyplot as plt                        # Plotting library
import numpy as np                                     # Numerical computations
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # Needed to enable 3D plotting in matplotlib

import matplotlib.pyplot as plt                        # Redundant import, but harmless


# ------------------------------------------------------------------ #
def grafic_rastrigin_gens(
    rastrigin,
    cma_hist: List[np.ndarray],
    gd_hist: List[np.ndarray],
    ga_hist: List[np.ndarray] | None = None,
    max_gens: int = 10,
    mesh_step: float = 0.25,
) -> None:
    """
    Plot the optimisation trajectory of each algorithm over the 3D Rastrigin surface.

    Parameters
    ----------
    rastrigin : callable
        The benchmark function being optimised (must be callable like f(x)).
    cma_hist : List[np.ndarray]
        History of CMA-ES populations over generations.
    gd_hist : List[np.ndarray]
        History of GD iterates (1 per step).
    ga_hist : List[np.ndarray] | None
        Optional: history of GA populations over generations.
    max_gens : int
        Number of generations to visualise (capped by history length).
    mesh_step : float
        Grid spacing for surface rendering.
    """
    b = rastrigin.bound                                                  # Domain bound (±)
    X, Y = np.mgrid[-b : b : mesh_step, -b : b : mesh_step]              # Create grid
    # Evaluate Rastrigin over the mesh
    Z = 10 * 2 + (X**2 - 10 * np.cos(2 * np.pi * X)) + (Y**2 - 10 * np.cos(2 * np.pi * Y))

    # Iterate through each generation
    for g in range(min(max_gens, len(cma_hist))):
        fig = plt.figure()                                              # New figure for each generation
        ax = fig.add_subplot(111, projection="3d")                      # 3D plot
        ax.set_title(f"Population {g + 1}")                             # Generation title
        ax.plot_wireframe(X, Y, Z, linewidth=0.4, color="grey", alpha=0.7)  # Plot Rastrigin surface

        pop = cma_hist[g]                                               # CMA-ES population at generation g
        ax.scatter(pop[:, 0], pop[:, 1], [rastrigin(p) for p in pop],   # Plot CMA-ES samples in red
                   c="red", s=18, label="CMA-ES")

        # If GA data available and includes this generation
        if ga_hist and g < len(ga_hist):
            gpop = ga_hist[g]                                           # GA population at generation g
            ax.scatter(gpop[:, 0], gpop[:, 1], [rastrigin(p) for p in gpop],  # Plot GA in green (x)
                       c="green", marker="x", s=28, label="GA")

        # If GD step exists for this generation
        if g < len(gd_hist):
            gd_pt = gd_hist[g]                                          # GD point at step g
            ax.scatter(gd_pt[0], gd_pt[1], rastrigin(gd_pt),            # Plot GD in blue (^) marker
                       marker="^", c="blue", s=40, label="GD")

        ax.set_xlabel("x")                                              # X axis label
        ax.set_ylabel("y")                                              # Y axis label
        ax.set_zlabel("f(x,y)")                                         # Z axis label (fitness)
        ax.legend()                                                     # Show legend
        plt.tight_layout()                                              # Reduce padding
        plt.show()                                                      # Display the plot


# ------------------------------------------------------------------ #
def plot_sigma_history(sigmas: List[float]):
    """
    Plot the evolution of CMA-ES step-size σ across generations.

    Parameters
    ----------
    sigmas : List[float]
        History of σ values (1 per generation).
    """
    plt.figure()                                                        # Create new figure
    plt.plot(sigmas)                                                    # Plot σ values
    plt.title("CMA-ES step-size σ")                                     # Title of the plot
    plt.xlabel("generation")                                            # X axis label
    plt.ylabel("σ")                                                     # Y axis label
    plt.tight_layout()                                                  # Layout adjustment
    plt.show()                                                          # Display the plot
