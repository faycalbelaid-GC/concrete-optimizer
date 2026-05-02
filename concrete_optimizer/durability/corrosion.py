"""
Corrosion propagation model.

References:
  Rodriguez et al. (1996), Liu & Weyers (1998),
  Broomfield (2007) — Corrosion of Steel in Concrete.
"""
import numpy as np
from ..models.mix_design import MixDesign


def corrosion_current_density(mix: MixDesign,
                               rh: float = 0.75,
                               T: float = 293.15,
                               mechanism: str = 'chloride') -> float:
    """
    Active corrosion current density i_corr (μA/cm²).

    Passive current (pre-initiation):   < 0.1 μA/cm²
    Carbonation-induced active:         1–10 μA/cm²
    Chloride-induced pitting:           5–100 μA/cm²

    Base rates from Rodriguez et al. (1996).
    """
    base = {'carbonation': 2.5, 'chloride': 12.0}.get(mechanism, 2.5)

    # Humidity: electrolyte needed for ionic transport
    if rh < 0.60:
        f_rh = 0.05
    elif rh < 0.75:
        f_rh = 0.5 + 0.5 * (rh - 0.60) / 0.15
    else:
        f_rh = 1.0

    # Arrhenius temperature correction, Ea = 40 kJ/mol
    f_T = float(np.exp((40000.0 / 8.314) * (1.0 / 293.15 - 1.0 / max(T, 250.0))))

    # Dense microstructure from SCMs → higher electrical resistivity → lower i_corr
    tb = max(mix.total_binder, 1.0)
    scm_r = (mix.fly_ash + mix.slag + mix.silica_fume) / tb
    f_resist = np.exp(-0.8 * scm_r)

    return float(base * f_rh * f_T * f_resist)


def time_to_cover_cracking(i_corr: float,
                             cover: float = 40.0,
                             diameter: float = 16.0) -> float:
    """
    Propagation time from corrosion onset to first visible cover crack (years).

    Empirical formula calibrated from Liu & Weyers (1998):
        t_prop ≈ W_cr / (α · i_corr)
    where W_cr is the critical rust mass per unit bar length (kg/m).

    Simplified to:  t_prop = C₀ / i_corr
    with C₀ = 25 + 0.4·cover  (larger cover → more rust before cracking).
    """
    if i_corr <= 0:
        return 999.0
    C0 = 22.0 + 0.35 * float(cover)
    t_prop = C0 / i_corr
    return float(np.clip(t_prop, 0.5, 200.0))


def rebar_cross_section_loss(i_corr: float,
                              t_prop: float,
                              diameter_0: float = 16.0) -> float:
    """
    Rebar diameter reduction Δd (mm) over propagation period.
    Faraday's law → Rodriguez (1996): Δd = 0.0232 · i_corr · t_prop
    """
    return 0.0232 * i_corr * t_prop


def residual_diameter(i_corr: float, t_prop: float,
                       diameter_0: float = 16.0) -> float:
    """Residual rebar diameter (mm) at end of service life."""
    return max(0.0, diameter_0 - rebar_cross_section_loss(i_corr, t_prop, diameter_0))
