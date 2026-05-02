"""
NSGA-II multi-objective evolutionary algorithm.
Ref: Deb, Pratap, Agarwal & Meyarivan (2002), IEEE TEC 6(2):182–197.

Operators:
  Selection   — binary tournament (crowded comparison)
  Crossover   — Simulated Binary Crossover SBX (Deb & Agrawal 1995)
  Mutation    — Polynomial mutation (Deb & Goyal 1996)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np


@dataclass
class Individual:
    genes: np.ndarray
    objectives: np.ndarray = field(default_factory=lambda: np.empty(0))
    rank: int = 0
    crowding_distance: float = 0.0
    # temporary fields for non-dominated sort (reset each generation)
    _dom_count: int = field(default=0, repr=False)
    _dom_set: list = field(default_factory=list, repr=False)


# ---------------------------------------------------------------------------
# Non-dominated sorting
# ---------------------------------------------------------------------------

def _dominates(a: Individual, b: Individual) -> bool:
    """True iff a weakly dominates b on all objectives and strictly on one (minimisation)."""
    return (np.all(a.objectives <= b.objectives)
            and np.any(a.objectives < b.objectives))


def fast_non_dominated_sort(pop: list[Individual]) -> list[list[Individual]]:
    """O(M·N²) fast non-dominated sorting (Deb 2002 §III-A)."""
    for p in pop:
        p._dom_count = 0
        p._dom_set = []

    fronts: list[list[Individual]] = [[]]
    for p in pop:
        for q in pop:
            if p is q:
                continue
            if _dominates(p, q):
                p._dom_set.append(q)
            elif _dominates(q, p):
                p._dom_count += 1
        if p._dom_count == 0:
            p.rank = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front: list[Individual] = []
        for p in fronts[i]:
            for q in p._dom_set:
                q._dom_count -= 1
                if q._dom_count == 0:
                    q.rank = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return [f for f in fronts if f]  # drop empty trailing front


def crowding_distance_assignment(front: list[Individual]) -> None:
    """Assign crowding distances in-place (Deb 2002 §III-B)."""
    n = len(front)
    if n < 3:
        for ind in front:
            ind.crowding_distance = np.inf
        return
    n_obj = len(front[0].objectives)
    for ind in front:
        ind.crowding_distance = 0.0
    for m in range(n_obj):
        front.sort(key=lambda x: x.objectives[m])
        front[0].crowding_distance = np.inf
        front[-1].crowding_distance = np.inf
        span = front[-1].objectives[m] - front[0].objectives[m]
        if span < 1e-12:
            continue
        for i in range(1, n - 1):
            front[i].crowding_distance += (
                (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / span
            )


# ---------------------------------------------------------------------------
# Genetic operators
# ---------------------------------------------------------------------------

def _tournament(pop: list[Individual]) -> Individual:
    """Binary tournament with crowded-comparison operator."""
    idx = np.random.choice(len(pop), 2, replace=False)
    a, b = pop[idx[0]], pop[idx[1]]
    if a.rank < b.rank:
        return a
    if b.rank < a.rank:
        return b
    return a if a.crowding_distance >= b.crowding_distance else b


def sbx_crossover(p1: np.ndarray, p2: np.ndarray,
                   bounds: np.ndarray,
                   eta_c: float = 20.0,
                   p_c: float = 0.90) -> tuple[np.ndarray, np.ndarray]:
    """Simulated Binary Crossover."""
    c1, c2 = p1.copy(), p2.copy()
    if np.random.rand() > p_c:
        return c1, c2
    for i in range(len(p1)):
        if np.random.rand() > 0.5:
            continue
        xl, xu = bounds[i, 0], bounds[i, 1]
        y1, y2 = min(p1[i], p2[i]), max(p1[i], p2[i])
        if y2 - y1 < 1e-10:
            continue
        r = np.random.rand()
        beta = 1.0 + 2.0 * min(y1 - xl, xu - y2) / (y2 - y1)
        alpha = 2.0 - beta ** (-(eta_c + 1.0))
        if r <= 1.0 / alpha:
            betaq = (r * alpha) ** (1.0 / (eta_c + 1.0))
        else:
            betaq = (1.0 / (2.0 - r * alpha)) ** (1.0 / (eta_c + 1.0))
        half_diff = 0.5 * betaq * (y2 - y1)
        mid = 0.5 * (y1 + y2)
        c1[i] = np.clip(mid - half_diff, xl, xu)
        c2[i] = np.clip(mid + half_diff, xl, xu)
    return c1, c2


def polynomial_mutation(genes: np.ndarray,
                         bounds: np.ndarray,
                         eta_m: float = 20.0,
                         p_m: Optional[float] = None) -> np.ndarray:
    """Polynomial mutation."""
    if p_m is None:
        p_m = 1.0 / len(genes)
    mutant = genes.copy()
    for i in range(len(genes)):
        if np.random.rand() > p_m:
            continue
        xl, xu = bounds[i, 0], bounds[i, 1]
        d1 = (genes[i] - xl) / (xu - xl + 1e-12)
        d2 = (xu - genes[i]) / (xu - xl + 1e-12)
        r = np.random.rand()
        mp = 1.0 / (eta_m + 1.0)
        if r < 0.5:
            xy = 1.0 - d1
            val = 2.0 * r + (1.0 - 2.0 * r) * xy ** (eta_m + 1.0)
            dq = val ** mp - 1.0
        else:
            xy = 1.0 - d2
            val = 2.0 * (1.0 - r) + 2.0 * (r - 0.5) * xy ** (eta_m + 1.0)
            dq = 1.0 - val ** mp
        mutant[i] = np.clip(genes[i] + dq * (xu - xl), xl, xu)
    return mutant


# ---------------------------------------------------------------------------
# NSGA-II engine
# ---------------------------------------------------------------------------

class NSGA2:
    """
    NSGA-II solver for real-valued multi-objective optimisation.

    Parameters
    ----------
    problem  : object with .bounds, .evaluate(genes), .is_feasible(genes)
    pop_size : population size (even number)
    n_gen    : number of generations
    seed     : random seed for reproducibility
    """

    def __init__(self, problem, pop_size: int = 100,
                 n_gen: int = 200, seed: int = 42):
        self.problem  = problem
        self.pop_size = pop_size
        self.n_gen    = n_gen
        np.random.seed(seed)
        self._bounds = np.array(problem.bounds)

    # ------------------------------------------------------------------
    def _sample(self) -> Individual:
        genes = np.array([
            np.random.uniform(lo, hi)
            for lo, hi in self._bounds
        ])
        return Individual(genes=genes)

    def _evaluate(self, ind: Individual) -> None:
        ind.objectives = self.problem.evaluate(ind.genes)

    def _initial_population(self) -> list[Individual]:
        pop: list[Individual] = []
        attempts = 0
        while len(pop) < self.pop_size:
            ind = self._sample()
            if self.problem.is_feasible(ind.genes):
                self._evaluate(ind)
                pop.append(ind)
            attempts += 1
            if attempts > self.pop_size * 500:
                raise RuntimeError(
                    "Cannot generate enough feasible individuals — "
                    "check problem bounds and constraints."
                )
        return pop

    def _offspring(self, parents: list[Individual]) -> list[Individual]:
        children: list[Individual] = []
        while len(children) < self.pop_size:
            p1 = _tournament(parents)
            p2 = _tournament(parents)
            g1, g2 = sbx_crossover(p1.genes, p2.genes, self._bounds)
            for g in (g1, g2):
                g = polynomial_mutation(g, self._bounds)
                if self.problem.is_feasible(g):
                    children.append(Individual(genes=g))
        return children[:self.pop_size]

    def _select_next(self, combined: list[Individual]) -> list[Individual]:
        fronts = fast_non_dominated_sort(combined)
        next_pop: list[Individual] = []
        for front in fronts:
            crowding_distance_assignment(front)
            if len(next_pop) + len(front) <= self.pop_size:
                next_pop.extend(front)
            else:
                remaining = self.pop_size - len(next_pop)
                front.sort(key=lambda x: -x.crowding_distance)
                next_pop.extend(front[:remaining])
                break
        return next_pop

    # ------------------------------------------------------------------
    def run(self, callback: Optional[Callable] = None
            ) -> tuple[list[Individual], list[dict]]:
        """
        Run NSGA-II.

        Returns
        -------
        pareto_front : list of non-dominated individuals
        history      : per-generation stats for convergence plots
        """
        population = self._initial_population()
        fronts = fast_non_dominated_sort(population)
        for front in fronts:
            crowding_distance_assignment(front)

        history: list[dict] = []

        for gen in range(self.n_gen):
            offspring = self._offspring(population)
            for ind in offspring:
                self._evaluate(ind)

            population = self._select_next(population + offspring)
            pareto = [ind for ind in population if ind.rank == 0]

            hist_entry = {
                'gen': gen,
                'pareto_size': len(pareto),
                'best_objectives': np.array([ind.objectives for ind in pareto]),
            }
            history.append(hist_entry)

            if callback:
                callback(gen, pareto, history)

        pareto_front = [ind for ind in population if ind.rank == 0]
        return pareto_front, history
