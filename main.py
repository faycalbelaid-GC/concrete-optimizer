"""
Concrete mix optimisation + durability simulation — main entry point.

Usage
-----
  python main.py              # standard run (100 generations)
  python main.py --fast       # quick demo (30 generations)
  python main.py --sensitivity  # sensitivity plots only, no optimisation
"""
import argparse
import os
import time
import numpy as np

from concrete_optimizer.optimization.problem import ConcreteMixProblem
from concrete_optimizer.optimization.nsga2 import NSGA2
from concrete_optimizer.models.strength import (
    compressive_strength_28d, concrete_class, characteristic_strength)
from concrete_optimizer.models.carbon_footprint import embodied_co2
from concrete_optimizer.durability.service_life import estimate_service_life
from concrete_optimizer.visualization.plots import (
    plot_pareto_front,
    plot_durability_profiles,
    plot_convergence,
    plot_sensitivity,
)

os.makedirs('results', exist_ok=True)

DIVIDER = "=" * 68


def print_solution(label: str, sol: tuple, problem: ConcreteMixProblem) -> None:
    _, mix, fc, co2, sl = sol
    fck = characteristic_strength(fc)
    print(f"\n  {'─' * 56}")
    print(f"  {label}")
    print(f"  {'─' * 56}")
    print(f"  Composition (kg/m³)")
    print(f"    Ciment           : {mix.cement:>6.0f}")
    print(f"    Eau              : {mix.water:>6.0f}   (w/b = {mix.w_b:.3f})")
    print(f"    Cendres volantes : {mix.fly_ash:>6.0f}   ({mix.fly_ash/max(mix.total_binder,1)*100:4.1f} % liant)")
    print(f"    Laitier GGBS     : {mix.slag:>6.0f}   ({mix.slag/max(mix.total_binder,1)*100:4.1f} % liant)")
    print(f"    Fumée de silice  : {mix.silica_fume:>6.0f}   ({mix.silica_fume/max(mix.total_binder,1)*100:4.1f} % liant)")
    print(f"    Sable            : {mix.fine_agg:>6.0f}")
    print(f"    Gravier          : {mix.coarse_agg:>6.0f}")
    print(f"  Performances")
    print(f"    f'c (28j)        : {fc:>6.1f} MPa  →  {concrete_class(fck)}")
    print(f"    fck estimé       : {fck:>6.1f} MPa")
    print(f"    CO₂ embarqué     : {co2:>6.0f} kg CO₂eq/m³")
    print(f"    SCM total        : {mix.scm_fraction*100:>5.1f} %")
    print(f"  Durabilité  (mécanisme : {sl.governing_mechanism})")
    print(f"    t_init carb.     : {sl.t_initiation_carbonation:>6.0f} ans")
    print(f"    t_init Cl⁻       : {sl.t_initiation_chloride:>6.0f} ans")
    print(f"    t_propagation    : {sl.t_propagation:>6.1f} ans")
    print(f"    Durée de vie     : {sl.t_service_total:>6.0f} ans")
    print(f"    i_corr actif     : {sl.corrosion_rate_ua:>6.2f} μA/cm²")


def main() -> None:
    parser = argparse.ArgumentParser(description='Concrete mix optimiser')
    parser.add_argument('--fast',        action='store_true',
                        help='Quick demo with 30 generations')
    parser.add_argument('--sensitivity', action='store_true',
                        help='Sensitivity analysis only')
    parser.add_argument('--pop',   type=int, default=120, help='Population size')
    parser.add_argument('--gen',   type=int, default=100, help='Generations')
    parser.add_argument('--fc-min',type=float, default=25.0, help='Min f\'c (MPa)')
    parser.add_argument('--cover', type=float, default=40.0, help='Cover (mm)')
    parser.add_argument('--carb',  default='XC3', help='Carbonation class')
    parser.add_argument('--cl',    default='XS2', help='Chloride class')
    args = parser.parse_args()

    print(DIVIDER)
    print("  OPTIMISATION BÉTON — NSGA-II  |  Durabilité & Empreinte CO₂")
    print("  Modèles : fib Bulletin 34, DuraCrete, EN 1992, EN 206")
    print(DIVIDER)

    problem = ConcreteMixProblem(
        fc_min=args.fc_min,
        cover=args.cover,
        exposure_carb=args.carb,
        exposure_cl=args.cl,
        rh=0.70,
        T=293.15,
    )

    if args.sensitivity:
        print("\n  Génération des courbes de sensibilité...")
        fig = plot_sensitivity(problem, save_path='results/sensitivity.png')
        print("  → results/sensitivity.png")
        return

    n_gen = 30 if args.fast else args.gen
    pop_s = 80 if args.fast else args.pop

    print(f"\n  Paramètres du problème")
    print(f"    f'c_min = {problem.fc_min} MPa  |  Enrobage = {problem.cover} mm")
    print(f"    Exposition : {problem.exposure_carb} (carb.) / {problem.exposure_cl} (Cl⁻)")
    print(f"  Paramètres NSGA-II")
    print(f"    Population = {pop_s}  |  Générations = {n_gen}")
    print(f"    Variables = 6D  |  Objectifs = 3  (CO₂, f'c, SL)\n")

    optimizer = NSGA2(problem=problem, pop_size=pop_s, n_gen=n_gen, seed=42)

    def progress(gen, pareto, history):
        if gen % max(1, n_gen // 10) == 0:
            objs = np.array([ind.objectives for ind in pareto])
            print(
                f"  gen {gen:4d} | Pareto={len(pareto):3d} | "
                f"CO₂_min={objs[:,0].min():.0f} | "
                f"f'c_max={-objs[:,1].min():.1f} MPa | "
                f"SL_max={-objs[:,2].min():.0f} ans"
            )

    t0 = time.time()
    pareto_front, history = optimizer.run(callback=progress)
    elapsed = time.time() - t0
    print(f"\n  Terminé en {elapsed:.1f} s  |  {len(pareto_front)} solutions Pareto\n")

    # ── Analyse du front de Pareto ────────────────────────────────────────
    solutions = []
    for ind in pareto_front:
        mix = problem.decode(ind.genes)
        fc  = compressive_strength_28d(mix)
        co2 = embodied_co2(mix)
        sl  = estimate_service_life(
            mix, problem.cover, problem.exposure_carb, problem.exposure_cl,
            problem.rh, problem.T,
        )
        solutions.append((ind, mix, fc, co2, sl))

    best_co2 = min(solutions, key=lambda s: s[3])
    best_fc  = max(solutions, key=lambda s: s[2])
    best_sl  = max(solutions, key=lambda s: s[4].t_service_total)

    # Utopia-point compromise (normalised Euclidean distance)
    co2_arr = np.array([s[3] for s in solutions])
    fc_arr  = np.array([s[2] for s in solutions])
    sl_arr  = np.array([s[4].t_service_total for s in solutions])
    eps = 1e-9
    n_co2 = (co2_arr - co2_arr.min()) / (co2_arr.max() - co2_arr.min() + eps)
    n_fc  = (fc_arr.max()  - fc_arr)  / (fc_arr.max()  - fc_arr.min()  + eps)
    n_sl  = (sl_arr.max()  - sl_arr)  / (sl_arr.max()  - sl_arr.min()  + eps)
    compromise = solutions[int(np.argmin(np.sqrt(n_co2**2 + n_fc**2 + n_sl**2)))]

    print(DIVIDER)
    print("  RÉSULTATS — TOP SOLUTIONS DU FRONT DE PARETO")
    print(DIVIDER)
    print_solution("★  MEILLEURE EMPREINTE CO₂  (CO₂ minimal)", best_co2, problem)
    print_solution("★  MEILLEURE RÉSISTANCE     (f'c maximal)", best_fc,  problem)
    print_solution("★  MEILLEURE DURABILITÉ     (SL maximale)", best_sl,  problem)
    print_solution("★  SOLUTION COMPROMIS       (distance utopie minimale)", compromise, problem)

    # ── Figures ───────────────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("  GRAPHIQUES")
    print(DIVIDER)

    plot_convergence(history, save_path='results/convergence.png')
    print("  → results/convergence.png")

    plot_pareto_front(pareto_front, problem, save_path='results/pareto_front.png')
    print("  → results/pareto_front.png")

    for tag, sol in [('compromise', compromise), ('best_co2', best_co2),
                      ('best_sl', best_sl)]:
        _, mix, _, _, sl = sol
        plot_durability_profiles(
            mix, problem.cover, problem.exposure_carb, problem.exposure_cl,
            problem.rh,
            save_path=f'results/durability_{tag}.png',
        )
        print(f"  → results/durability_{tag}.png")

    plot_sensitivity(problem, save_path='results/sensitivity.png')
    print("  → results/sensitivity.png")

    print(f"\n  Tous les résultats dans ./results/")
    print(DIVIDER)


if __name__ == '__main__':
    main()
