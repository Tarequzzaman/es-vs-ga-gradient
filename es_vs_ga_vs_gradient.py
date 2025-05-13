"""
es_experiment.py
────────────────────────────────────────────────────────────────────────────
CMA-ES  vs  Genetic Algorithm  vs  Gradient Descent on the 2-D Rastrigin
• Early-stop when best fitness hasn’t improved by > tol for < patience > steps
• Optional grafic-style 3-D visualisation of first N generations
• Prints detailed per-run results, median summary, and Mann-Whitney U-tests
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import argparse, math, random
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D      # noqa: F401  (3-D backend)
from scipy.stats import mannwhitneyu


# ╔══════════════════════════════════════════════════════════════╗
# 1. BENCHMARK FUNCTION
# ╚══════════════════════════════════════════════════════════════╝
class Rastrigin:
    """Multi-modal benchmark (global minimum f(0)=0)."""

    def __init__(self, dim: int = 2, bound: float = 5.12):
        self.dim, self.bound = dim, bound

    def __call__(self, x: np.ndarray) -> float:              # type: ignore
        return 10*self.dim + np.sum(x**2 - 10*np.cos(2*np.pi*x))


# ╔══════════════════════════════════════════════════════════════╗
# 2.  ABSTRACT OPTIMISER BASE
# ╚══════════════════════════════════════════════════════════════╝
class BaseOptimizer(ABC):
    """Parent class for all optimisers."""


    def __init__(self, problem: Rastrigin, max_iter: int):
        self.problem, self.max_iter = problem, max_iter
        self.best_f, self.best_x = float("inf"), None

    @abstractmethod
    def optimise(self, tol: float, patience: int) -> Tuple[np.ndarray, float, int]: 
        pass


# ╔══════════════════════════════════════════════════════════════╗
# 3.  GRADIENT DESCENT
# ╚══════════════════════════════════════════════════════════════╝
class GradientDescent(BaseOptimizer):
    """Constant-LR finite-difference GD."""


    def __init__(self, problem, lr=0.01, max_iter=2000):
        super().__init__(problem, max_iter)
        self.lr = lr
        self.x = np.random.uniform(-problem.bound, problem.bound, problem.dim)
        self.position_history: List[np.ndarray] = []


    def _grad(self, x, h=1e-5):
        g = np.zeros_like(x); fx = self.problem(x)
        for i in range(len(x)):
            xh = x.copy(); xh[i] += h
            g[i] = (self.problem(xh) - fx) / h
        return g
    
    
    def optimise(self, tol=1e-6, patience=50):
        stall = 0
        for it in range(1, self.max_iter+1):
            self.x -= self.lr * self._grad(self.x)
            self.x = np.clip(self.x, -self.problem.bound, self.problem.bound)
            f = self.problem(self.x); self.position_history.append(self.x.copy())
            if f < self.best_f - tol: self.best_f, self.best_x, stall = f, self.x.copy(), 0
            else: stall += 1
            if stall >= patience: return self.best_x, self.best_f, it
        return self.best_x, self.best_f, self.max_iter


# ╔══════════════════════════════════════════════════════════════╗
# 4.  CMA-ES  (μ/λ, –C)
# ╚══════════════════════════════════════════════════════════════╝
class CMAES(BaseOptimizer):
    """Covariance-Matrix Adaptation Evolution Strategy."""

    def __init__(self, problem, sigma=0.5, lamb=50, max_iter=1000):
        super().__init__(problem, max_iter)
        self.dim, self.sigma, self.lamb = problem.dim, sigma, lamb
        self.mu = lamb // 2
        self.centroid = np.random.uniform(-problem.bound, problem.bound, self.dim)
        self.C = np.eye(self.dim); self.ps = np.zeros(self.dim); self.pc = np.zeros(self.dim)
        self.chiN = np.sqrt(self.dim)*(1-1/(4*self.dim)+1/(21*self.dim**2))
        w = np.log(self.mu+0.5) - np.log(np.arange(1, self.mu+1))
        self.weights = w/np.sum(w); self.mueff = 1/np.sum(self.weights**2)
        self.cc   = 4/(self.dim+4)
        self.cs   = (self.mueff+2)/(self.dim+self.mueff+3)
        self.ccov1 = 2/((self.dim+1.3)**2+self.mueff)
        self.ccovmu= 2*(self.mueff-2+1/self.mueff)/((self.dim+2)**2+self.mueff)
        self.damps= 1+2*max(0,np.sqrt((self.mueff-1)/(self.dim+1))-1)+self.cs
        self.population_history: List[np.ndarray] = []


    # ––– helpers ––––––––––––––––––––––––––––––––––––––––––––––––
    def _sample(self):
        eigvals, eigvecs = np.linalg.eigh(self.C)
        sqrtC = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals,1e-20))) @ eigvecs.T
        return self.centroid + self.sigma * np.random.randn(self.lamb, self.dim) @ sqrtC.T
    

    def _update(self, pop, fit):
        idx = np.argsort(fit); pop = pop[idx]
        old = self.centroid.copy(); self.centroid = pop[:self.mu].T @ self.weights
        y = (self.centroid - old)/self.sigma
        self.ps = (1-self.cs)*self.ps + np.sqrt(self.cs*(2-self.cs)*self.mueff)*y
        hsig = (np.linalg.norm(self.ps) /
                np.sqrt(1-(1-self.cs)**(2*len(self.population_history))))/self.chiN < 1.4+2/(self.dim+1)
        self.pc = (1-self.cc)*self.pc + hsig*np.sqrt(self.cc*(2-self.cc)*self.mueff)*y
        artmp = (pop[:self.mu] - old)/self.sigma
        self.C = ((1-self.ccov1-self.ccovmu)*self.C +
                  self.ccov1*np.outer(self.pc, self.pc) +
                  self.ccovmu*(self.weights[:,None]*artmp).T @ artmp)
        self.sigma *= math.exp((np.linalg.norm(self.ps)/self.chiN-1)*self.cs/self.damps)
    # ––– run ––––––––––––––––––––––––––––––––––––––––––––––––––––


    def optimise(self, tol=1e-6, patience=30):
        stall = 0
        for gen in range(1, self.max_iter+1):
            pop = self._sample(); fit = np.array([self.problem(p) for p in pop])
            self.population_history.append(pop.copy())
            best = int(np.argmin(fit))
            if fit[best] < self.best_f - tol: self.best_f, self.best_x, stall = float(fit[best]), pop[best].copy(), 0
            else: stall += 1
            if stall >= patience or self.best_f < tol: return self.best_x, self.best_f, gen
            self._update(pop, fit)
        return self.best_x, self.best_f, self.max_iter


# ╔══════════════════════════════════════════════════════════════╗
# 5.  GENETIC ALGORITHM
# ╚══════════════════════════════════════════════════════════════╝
class GeneticAlgorithm(BaseOptimizer):
    """Uniform-crossover GA with decaying Gaussian mutation."""

    def __init__(self, problem, pop_size=50, mut_rate=0.1,
                 elite=2, tournament_k=3, max_iter=1000):
        super().__init__(problem, max_iter)
        self.pop_size, self.mut_rate, self.elite, self.k = pop_size, mut_rate, elite, tournament_k
        self.dim, self.bound = problem.dim, problem.bound
        self.population_history: List[np.ndarray] = []


    def _tournament(self, fitness):
        idx = np.random.choice(len(fitness), self.k, replace=False)
        return idx[int(np.argmin(fitness[idx]))]
    

    def optimise(self, tol=1e-6, patience=30):
        pop = np.random.uniform(-self.bound, self.bound, (self.pop_size, self.dim)); stall = 0
        for gen in range(1, self.max_iter+1):
            fit = np.array([self.problem(ind) for ind in pop])
            self.population_history.append(pop.copy())
            if fit.min() < self.best_f - tol: self.best_f, stall = float(fit.min()), 0
            else: stall += 1
            if stall >= patience or self.best_f < tol:
                return pop[int(np.argmin(fit))].copy(), self.best_f, gen
            # elitism
            new_pop = pop[np.argsort(fit)][:self.elite]
            mut_sigma = 0.1*self.bound*(0.995**gen)
            while len(new_pop) < self.pop_size:
                p1, p2 = pop[self._tournament(fit)], pop[self._tournament(fit)]
                child = np.where(np.random.rand(self.dim) < 0.5, p1, p2)  # crossover
                mask = np.random.rand(self.dim) < self.mut_rate
                child[mask] += np.random.randn(mask.sum()) * mut_sigma
                child = np.clip(child, -self.bound, self.bound)
                new_pop = np.vstack((new_pop, child))
            pop = new_pop
        return pop[np.argmin([self.problem(ind) for ind in pop])], self.best_f, self.max_iter


# ╔══════════════════════════════════════════════════════════════╗
# 6.  WIREFRAME VISUALISER  (now shows GA in green ×)
# ╚══════════════════════════════════════════════════════════════╝
def grafic_rastrigin_gens(
        rastrigin: Rastrigin,
        cma_hist: List[np.ndarray],
        gd_hist:  List[np.ndarray],
        ga_hist:  List[np.ndarray] | None = None,
        max_gens=10, mesh_step=0.25
) -> None:
    

    b = rastrigin.bound
    X, Y = np.mgrid[-b:b:mesh_step, -b:b:mesh_step]
    Z = 10*2 + (X**2 - 10*np.cos(2*np.pi*X)) + (Y**2 - 10*np.cos(2*np.pi*Y))
    for g in range(min(max_gens, len(cma_hist))):
        fig = plt.figure(); ax = fig.add_subplot(111, projection="3d")
        ax.set_title(f"Population {g+1}")
        ax.plot_wireframe(X, Y, Z, linewidth=0.4, color="grey", alpha=0.7)
        # CMA-ES
        pop = cma_hist[g]
        ax.scatter(pop[:,0], pop[:,1], [rastrigin(p) for p in pop],
                   c="red", s=18, label="CMA-ES")
        # GA (green ×)
        if ga_hist and g < len(ga_hist):
            gpop = ga_hist[g]
            ax.scatter(gpop[:,0], gpop[:,1], [rastrigin(p) for p in gpop],
                       c="green", marker="x", s=28, label="GA")
        # GD
        if g < len(gd_hist):
            gd_pt = gd_hist[g]
            ax.scatter(gd_pt[0], gd_pt[1], rastrigin(gd_pt),
                       marker="^", c="blue", s=40, label="GD")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("f(x,y)")
        ax.legend(); plt.tight_layout(); plt.show()


# ╔══════════════════════════════════════════════════════════════╗
# 7.  EXPERIMENT DRIVER
# ╚══════════════════════════════════════════════════════════════╝
def run_experiment(
        prob: Rastrigin,
        runs: int,
        tol: float,
        patience: int
) -> pd.DataFrame:
    
    rows = []
    for r in range(runs):
        np.random.seed(r); random.seed(r)
        gd = GradientDescent(prob, lr=0.01, max_iter=2000)
        _, gd_best, gd_steps = gd.optimise(tol, patience)
        cma = CMAES(prob, sigma=0.5, lamb=50, max_iter=1000)
        _, cma_best, cma_steps = cma.optimise(tol, patience)
        ga = GeneticAlgorithm(prob, pop_size=50, max_iter=1000)
        _, ga_best, ga_steps = ga.optimise(tol, patience)
        rows.append(dict(Run=r+1, GD_best=gd_best, GD_steps=gd_steps,
                         CMA_best=cma_best, CMA_steps=cma_steps,
                         GA_best=ga_best, GA_steps=ga_steps))
    return pd.DataFrame(rows)


# ╔══════════════════════════════════════════════════════════════╗
# 8.  MAIN
# ╚══════════════════════════════════════════════════════════════╝
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs",  type=int, default=20, help="number of seeds")
    parser.add_argument("--plots", type=int, default=0,  help="gens to plot (0=none)")
    args = parser.parse_args()

    tol, patience = 1e-6, 30
    rastrigin = Rastrigin(dim=2)

    # ─ visual demo ─
    if args.plots:
        demo_cma = CMAES(rastrigin, sigma=0.5, lamb=50, max_iter=1000)
        demo_gd  = GradientDescent(rastrigin, lr=0.01, max_iter=2000)
        demo_ga  = GeneticAlgorithm(rastrigin, pop_size=50, max_iter=1000)
        demo_cma.optimise(tol, patience); demo_gd.optimise(tol, patience); demo_ga.optimise(tol, patience)
        grafic_rastrigin_gens(rastrigin, demo_cma.population_history,
                              demo_gd.position_history, demo_ga.population_history,
                              max_gens=args.plots)

    # ─ stats run ─
    df = run_experiment(rastrigin, runs=args.runs, tol=tol, patience=patience)

    summary = pd.DataFrame({
        "Algorithm": ["CMA-ES", "Genetic Alg", "Gradient Descent"],
        "Median best f": [df["CMA_best"].median(),
                          df["GA_best"].median(),
                          df["GD_best"].median()],
        "Median steps":  [df["CMA_steps"].median(),
                          df["GA_steps"].median(),
                          df["GD_steps"].median()],
        "Success-rate (f≤tol)": [(df["CMA_best"] <= tol).mean(),
                                 (df["GA_best"]  <= tol).mean(),
                                 (df["GD_best"]  <= tol).mean()]
    })

    print("\n===== Detailed runs ====="); print(df.to_string(index=False, float_format=lambda x:f"{x: .4g}"))
    print("\n===== Summary (medians over runs) ====="); print(summary.to_string(index=False, float_format=lambda x:f"{x: .4g}"))

    for b in [("CMA_best","GA_best"), ("CMA_best","GD_best")]:
        stat, p = mannwhitneyu(df[b[0]], df[b[1]]); print(f"U({b[0][:3]},{b[1][:3]}) p={p:.3g}")
