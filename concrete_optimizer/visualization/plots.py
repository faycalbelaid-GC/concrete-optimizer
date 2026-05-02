"""
Publication-quality matplotlib figures for the concrete optimisation project.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection
import numpy as np

from ..models.mix_design import MixDesign
from ..models.strength import compressive_strength_28d, concrete_class
from ..models.carbon_footprint import embodied_co2
from ..durability.carbonation import carbonation_coefficient_k, carbonation_depth_array
from ..durability.chloride import (chloride_profile,
                                    chloride_diffusion_coefficient,
                                    CS_BY_CLASS)
from ..durability.service_life import estimate_service_life

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
})


# ──────────────────────────────────────────────────────────────────────────────
def plot_pareto_front(pareto: list, problem, save_path: str | None = None):
    """3-D Pareto front + three 2-D projections."""
    data = np.array([ind.objectives for ind in pareto])
    co2 = data[:, 0]
    fc  = -data[:, 1]
    sl  = -data[:, 2]

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        "Front de Pareto NSGA-II — Optimisation Béton",
        fontsize=14, fontweight='bold', y=0.98,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

    # --- 3D scatter
    ax3d = fig.add_subplot(gs[0, :2], projection='3d')
    sc   = ax3d.scatter(co2, fc, sl, c=sl, cmap='RdYlGn',
                         s=55, alpha=0.85, depthshade=True)
    ax3d.set_xlabel('CO₂  (kg/m³)', labelpad=8)
    ax3d.set_ylabel("f'c  (MPa)",    labelpad=8)
    ax3d.set_zlabel('Durée de vie  (ans)', labelpad=8)
    ax3d.set_title('Front de Pareto 3D', pad=10)
    fig.colorbar(sc, ax=ax3d, label='Durée de vie (ans)', shrink=0.55, pad=0.1)

    # --- Info box
    ax_info = fig.add_subplot(gs[0, 2])
    ax_info.axis('off')
    txt = (
        f"Solutions Pareto : {len(pareto)}\n\n"
        f"CO₂\n  min = {co2.min():.0f}  max = {co2.max():.0f} kg/m³\n\n"
        f"Résistance\n  min = {fc.min():.1f}  max = {fc.max():.1f} MPa\n\n"
        f"Durée de vie\n  min = {sl.min():.0f}  max = {sl.max():.0f} ans\n\n"
        f"f'c_min = {problem.fc_min} MPa\n"
        f"Enrobage = {problem.cover} mm\n"
        f"Exp. carb. = {problem.exposure_carb}\n"
        f"Exp. Cl⁻  = {problem.exposure_cl}"
    )
    ax_info.text(0.05, 0.97, txt, transform=ax_info.transAxes,
                  fontsize=9, va='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round,pad=0.6',
                             facecolor='#ddeeff', alpha=0.7))

    # --- 2-D projections
    def proj(ax, xv, yv, cv, xl, yl, ttl, clbl):
        sc2 = ax.scatter(xv, yv, c=cv, cmap='RdYlGn', s=35, alpha=0.8)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(ttl)
        ax.grid(True, alpha=0.25)
        fig.colorbar(sc2, ax=ax, label=clbl, shrink=0.85)

    proj(fig.add_subplot(gs[1, 0]), co2, fc,  sl,
         'CO₂ (kg/m³)', "f'c (MPa)", 'CO₂ vs Résistance', 'SL (ans)')
    proj(fig.add_subplot(gs[1, 1]), co2, sl,  fc,
         'CO₂ (kg/m³)', 'Durée de vie (ans)', 'CO₂ vs Durée de vie', "f'c (MPa)")
    proj(fig.add_subplot(gs[1, 2]), fc,  sl, co2,
         "f'c (MPa)", 'Durée de vie (ans)', 'Résistance vs Durée de vie',
         'CO₂ (kg/m³)')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
def plot_durability_profiles(mix: MixDesign,
                              cover: float = 35.0,
                              exposure_carb: str = 'XC3',
                              exposure_cl: str = 'XS2',
                              rh: float = 0.70,
                              save_path: str | None = None):
    """
    Four-panel durability figure:
      (a) Carbonation depth vs time
      (b) Chloride concentration profiles at several ages
      (c) Mix composition bar chart
      (d) Service life decomposition (initiation + propagation)
    """
    from ..durability.service_life import _RH_BY_CLASS, estimate_service_life

    rh_carb = _RH_BY_CLASS.get(exposure_carb, rh)
    rh_cl   = _RH_BY_CLASS.get(exposure_cl,   rh)
    sl_res  = estimate_service_life(mix, cover, exposure_carb, exposure_cl, rh)

    fc  = compressive_strength_28d(mix)
    co2 = embodied_co2(mix)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Profils de Durabilité — f'c={fc:.1f} MPa | "
        f"CO₂={co2:.0f} kg/m³ | SL={sl_res.t_service_total:.0f} ans",
        fontsize=13, fontweight='bold',
    )

    # (a) Carbonation depth
    ax = axes[0, 0]
    k  = carbonation_coefficient_k(mix, rh_carb)
    t  = np.linspace(0, 120, 600)
    xc = carbonation_depth_array(t, k)

    ax.fill_between(t, xc, alpha=0.12, color='steelblue')
    ax.plot(t, xc, 'steelblue', lw=2,
             label=f'Profondeur carbonatation  (k={k:.2f} mm/an⁰·⁵)')
    ax.axhline(cover, color='crimson', ls='--', lw=1.5,
                label=f'Enrobage = {cover:.0f} mm')

    ti_c = sl_res.t_initiation_carbonation
    if ti_c < 120:
        ax.axvline(ti_c, color='darkorange', ls=':', lw=1.5,
                    label=f't_init = {ti_c:.0f} ans')
    ax.set_xlabel('Temps (ans)');  ax.set_ylabel('Profondeur (mm)')
    ax.set_title(f'(a) Carbonatation — classe {exposure_carb}')
    ax.legend(); ax.grid(True, alpha=0.25)
    ax.set_xlim(0, 120); ax.set_ylim(bottom=0)

    # (b) Chloride profiles
    ax = axes[0, 1]
    D_a, m_age = chloride_diffusion_coefficient(mix)
    Cs  = CS_BY_CLASS.get(exposure_cl, 3.0)
    x_d = np.linspace(0, 100, 400)
    cmap = plt.get_cmap('Blues')
    t_years = [2, 5, 10, 25, 50, 100]
    for idx, t_y in enumerate(t_years):
        C = chloride_profile(x_d, t_y, D_a, m_age, Cs)
        ax.plot(x_d, C, color=cmap(0.25 + 0.75 * idx / (len(t_years) - 1)),
                lw=1.5, label=f't = {t_y} ans')

    ax.axvline(cover, color='crimson', ls='--', lw=1.5,
                label=f'Enrobage {cover:.0f} mm')
    ax.axhline(0.40, color='darkorange', ls=':', lw=1.5,
                label='C_crit = 0.40 %')
    ax.set_xlabel('Profondeur (mm)')
    ax.set_ylabel('Teneur Cl⁻ (% masse ciment)')
    ax.set_title(f'(b) Chlorures — classe {exposure_cl}  (D_a={D_a:.1f} mm²/an)')
    ax.legend(ncol=2); ax.grid(True, alpha=0.25)
    ax.set_xlim(0, 100); ax.set_ylim(0, Cs * 1.12)

    # (c) Composition waterfall
    ax = axes[1, 0]
    labels = ['Ciment', 'Eau', 'Cendres\nvolantes', 'Laitier',
              'Fumée\nde silice', 'Sable', 'Gravier']
    qties  = [mix.cement, mix.water, mix.fly_ash, mix.slag,
              mix.silica_fume, mix.fine_agg, mix.coarse_agg]
    colors = ['#e74c3c', '#3498db', '#95a5a6', '#7f8c8d',
              '#bdc3c7', '#f39c12', '#e67e22']
    bars = ax.barh(labels, qties, color=colors, edgecolor='white', linewidth=0.6)
    for bar, qty in zip(bars, qties):
        if qty > 5:
            ax.text(qty + 8, bar.get_y() + bar.get_height() / 2,
                     f'{qty:.0f}', va='center', fontsize=8)
    ax.set_xlabel('Quantité (kg/m³)')
    ax.set_title(f'(c) Composition  —  {concrete_class(fc * 0.85)}  |  w/b={mix.w_b:.3f}')
    ax.grid(True, alpha=0.25, axis='x')

    # (d) Service life decomposition
    ax = axes[1, 1]
    mechanisms = ['Carbonatation', 'Chlorures']
    t_inits = [sl_res.t_initiation_carbonation, sl_res.t_initiation_chloride]
    t_props = [sl_res.t_propagation, sl_res.t_propagation]

    t_inits_plot = [min(v, 500) for v in t_inits]
    ax.bar(mechanisms, t_inits_plot, label='Phase initiation',
            color=['#3498db', '#e74c3c'], alpha=0.85)
    ax.bar(mechanisms, t_props, bottom=t_inits_plot,
            label='Phase propagation',
            color=['#2980b9', '#c0392b'], alpha=0.85)

    gov_idx = 0 if sl_res.governing_mechanism == 'carbonation' else 1
    gov_total = min(t_inits[gov_idx] + t_props[gov_idx], 500)
    ax.annotate(f"⚑ Gouvernant\n{gov_total:.0f} ans",
                 xy=(gov_idx, gov_total),
                 xytext=(gov_idx + 0.3, gov_total * 0.8),
                 fontsize=9, color='darkred',
                 arrowprops=dict(arrowstyle='->', color='darkred'))

    ax.set_ylabel('Durée de vie (ans)')
    ax.set_title('(d) Décomposition de la durée de vie')
    ax.legend(); ax.grid(True, alpha=0.25, axis='y')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
def plot_convergence(history: list, save_path: str | None = None):
    """Four convergence indicators over NSGA-II generations."""
    gens  = [h['gen'] for h in history]
    sizes = [h['pareto_size'] for h in history]

    def safe_stat(attr_idx, agg):
        vals = []
        for h in history:
            obj = h['best_objectives']
            if len(obj):
                vals.append(agg(obj[:, attr_idx]))
            else:
                vals.append(np.nan)
        return vals

    min_co2 = safe_stat(0, np.min)
    max_fc  = [-v for v in safe_stat(1, np.min)]   # stored as -fc
    max_sl  = [-v for v in safe_stat(2, np.min)]   # stored as -sl

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Convergence NSGA-II', fontsize=13, fontweight='bold')

    cfg = [
        (axes[0, 0], gens, sizes,   'g',  'Taille du front de Pareto',        'Nb solutions'),
        (axes[0, 1], gens, min_co2, 'b',  'CO₂ minimum du front de Pareto',   'CO₂ (kg/m³)'),
        (axes[1, 0], gens, max_fc,  'r',  "f'c maximum du front de Pareto",   "f'c (MPa)"),
        (axes[1, 1], gens, max_sl,  'm',  'Durée de vie max du front de Pareto','Durée de vie (ans)'),
    ]
    for ax, x, y, col, title, ylabel in cfg:
        ax.plot(x, y, color=col, lw=1.5)
        ax.set_title(title); ax.set_xlabel('Génération'); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
def plot_sensitivity(problem, cover_values: list | None = None,
                     wc_values: list | None = None,
                     save_path: str | None = None):
    """
    Parametric sensitivity: service life and CO2 vs cover depth and w/b.
    Useful for design charts.
    """
    if cover_values is None:
        cover_values = [20, 30, 40, 50, 60, 75]
    if wc_values is None:
        wc_values = np.linspace(0.35, 0.65, 40)

    from ..models.mix_design import MixDesign

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Analyse de Sensibilité — Enrobage et Rapport w/b',
                  fontsize=12, fontweight='bold')

    cmap = plt.get_cmap('viridis')
    for idx, cov in enumerate(cover_values):
        sls, co2s = [], []
        for wc in wc_values:
            mix = MixDesign(cement=350, water=350 * wc,
                             fine_agg=750, coarse_agg=1050)
            sl  = estimate_service_life(
                mix, cov, problem.exposure_carb, problem.exposure_cl, problem.rh)
            sls.append(sl.t_service_total)
            co2s.append(embodied_co2(mix))
        col = cmap(idx / (len(cover_values) - 1))
        axes[0].plot(wc_values, sls,  color=col, lw=1.8, label=f'enrobage {cov}mm')
        axes[1].plot(wc_values, co2s, color=col, lw=1.8, label=f'enrobage {cov}mm')

    for ax, ylabel, title in [
        (axes[0], 'Durée de vie (ans)',  'Durée de vie vs w/b'),
        (axes[1], 'CO₂ (kg/m³)',         'Empreinte CO₂ vs w/b'),
    ]:
        ax.set_xlabel('Rapport w/b'); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig
