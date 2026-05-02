"""
Carbonation-induced corrosion initiation model.

Model: simplified Fick's 1st law  →  x_c(t) = k · √t
k calibrated from Papadakis (1991) + fib Bulletin 34 (2006) §4.1.
"""
import numpy as np
from ..models.mix_design import MixDesign


def carbonation_coefficient_k(mix: MixDesign,
                               rh: float = 0.65,
                               co2_conc: float = 4e-4) -> float:
    """
    Carbonation coefficient k (mm / year^0.5).

    Parameters
    ----------
    mix       : concrete mix design
    rh        : ambient relative humidity [0–1]
    co2_conc  : atmospheric CO2 volume fraction (0.04% ≈ 4e-4 outdoor)

    Returns
    -------
    k  in mm / year^0.5
    """
    wc = float(np.clip(mix.w_c, 0.28, 0.75))

    # Base coefficient for plain OPC (Papadakis 1991 empirical fit)
    # k increases exponentially with w/c
    k_base = 5.0 * np.exp(2.0 * (wc - 0.45))

    # SCM fractions of total binder
    tb = max(mix.total_binder, 1.0)
    fa_r = mix.fly_ash / tb
    sl_r = mix.slag / tb
    sf_r = mix.silica_fume / tb

    # SCM correction:
    #   FA & slag ↑ k  (lower CaO binding capacity → faster front advance)
    #   SF ↓ k         (pore refinement → lower D_CO2)
    f_scm = 1.0 + 0.55 * fa_r + 0.45 * sl_r - 0.35 * sf_r

    # Humidity factor: maximum carbonation rate at RH ≈ 55 %
    # (too dry → no reaction; too wet → pores filled → blocked diffusion)
    if rh < 0.45:
        f_rh = 0.05
    elif rh < 0.75:
        f_rh = 1.0 - 0.55 * (rh - 0.45) / 0.30
        f_rh = max(f_rh, 0.25)
    else:
        f_rh = 0.25 * max(1.0 - (rh - 0.75) / 0.25, 0.1)

    # CO2 concentration scaling (√ law from diffusion flux)
    f_co2 = np.sqrt(co2_conc / 4e-4)

    k = float(np.clip(k_base * f_scm * f_rh * f_co2, 0.05, 30.0))
    return k


def carbonation_depth(t: float, k: float) -> float:
    """Carbonation front depth (mm) at time t (years)."""
    return k * np.sqrt(max(t, 0.0))


def carbonation_depth_array(t: np.ndarray, k: float) -> np.ndarray:
    """Vectorised carbonation depth."""
    return k * np.sqrt(np.maximum(t, 0.0))


def time_to_carbonation_front(depth_mm: float, k: float) -> float:
    """Time (years) for carbonation front to reach depth (mm)."""
    if k <= 0:
        return 999.0
    return float(np.clip((depth_mm / k) ** 2, 0.0, 999.0))


def carbonation_service_life(mix: MixDesign,
                              cover: float = 30.0,
                              rh: float = 0.65,
                              co2_conc: float = 4e-4) -> float:
    """
    Initiation service life (years) for carbonation-induced corrosion.
    cover : reinforcement cover depth (mm)
    """
    k = carbonation_coefficient_k(mix, rh, co2_conc)
    return time_to_carbonation_front(cover, k)
