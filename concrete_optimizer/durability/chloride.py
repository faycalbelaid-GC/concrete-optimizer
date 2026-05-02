"""
Chloride ingress model — Fick's 2nd law with time-dependent diffusivity.

References:
  Bamforth (1999), Nilsson & Tang (1996), DuraCrete (2000),
  fib Model Code for Service Life Design (2010) §4.3.
"""
import numpy as np
from scipy.special import erfc, erfcinv
from ..models.mix_design import MixDesign

# Surface chloride concentrations by EN 206 exposure class (% mass cement)
CS_BY_CLASS: dict[str, float] = {
    'XS1': 1.5,   # aerial sea salt
    'XS2': 3.0,   # permanently submerged
    'XS3': 4.5,   # tidal / splash zone
    'XD1': 1.0,   # moderate humidity, de-icing
    'XD2': 2.0,   # wet/dry cycles, de-icing
    'XD3': 2.5,   # heavily trafficked road
}


def chloride_diffusion_coefficient(mix: MixDesign,
                                    T: float = 293.15) -> tuple[float, float]:
    """
    Apparent chloride diffusion coefficient D_a (mm² / year) and age factor m.

    D28 from Bamforth (1999):
        log10(D28 / 1e-12 m²/s) = -3.07 + 5.55 · (w/c)

    Age factor m (time-dependent diffusivity): DuraCrete (2000).
    Temperature correction: Arrhenius, Ea = 35 kJ/mol.

    Returns
    -------
    D_a : apparent diffusion coefficient (mm²/year)
    m   : age factor (dimensionless, 0.2 – 0.65)
    """
    wc = float(np.clip(mix.w_c, 0.30, 0.75))

    # D28 in m²/s, then convert to mm²/year
    D28_m2s = 10.0 ** (-3.07 + 5.55 * wc) * 1e-12
    D28 = D28_m2s * 3.156e13  # 1 m²/s = 3.156×10¹³ mm²/year

    # SCM reduction of diffusivity (Hooton 2001)
    tb = max(mix.total_binder, 1.0)
    fa_r = mix.fly_ash / tb
    sl_r = mix.slag / tb
    sf_r = mix.silica_fume / tb
    D28 *= float(np.exp(-3.0 * fa_r - 4.5 * sl_r - 9.0 * sf_r))

    # Age factor (higher SCM content → stronger time-dependence)
    m = float(np.clip(0.25 + 0.4 * (fa_r + sl_r), 0.20, 0.65))

    # Temperature correction (Arrhenius)
    f_T = float(np.exp((35000.0 / 8.314) * (1.0 / 293.15 - 1.0 / max(T, 250.0))))

    # Apparent diffusion at t=1 year reference: D_a = D28·(28/365)^m · f_T
    D_a = D28 * (28.0 / 365.0) ** m * f_T
    return D_a, m


def chloride_profile(x: np.ndarray,
                     t: float,
                     D_a: float,
                     m: float,
                     Cs: float = 3.0,
                     Ci: float = 0.02) -> np.ndarray:
    """
    Chloride concentration profile C(x, t) [% mass of cement].

    DuraCrete solution to Fick's 2nd law with time-dependent D(t):
        C(x,t) = Ci + (Cs - Ci) · erfc( x / (2·√(D_a·t)) )

    Parameters
    ----------
    x   : depth array (mm)
    t   : time (years)
    D_a : apparent diffusion coefficient (mm²/year)
    m   : age factor
    Cs  : surface chloride concentration (% mass cement)
    Ci  : initial chloride content (% mass cement)
    """
    denom = 2.0 * np.sqrt(max(D_a * t, 1e-12))
    return Ci + (Cs - Ci) * erfc(x / denom)


def time_to_corrosion_initiation(cover: float,
                                  D_a: float,
                                  Cs: float = 3.0,
                                  Ci: float = 0.02,
                                  Ccr: float = 0.40) -> float:
    """
    Time (years) for chloride concentration at rebar depth to reach
    the critical threshold Ccr (% mass cement).

    Inverted from Fick's 2nd law:
        t_i = (cover / (2 · erfcinv(ratio)))² / D_a

    Ccr = 0.40 % (uncoated steel, EN 206 / fib MC 2010 §6.1.1).
    """
    if Ccr >= Cs:
        return 999.0
    ratio = (Ccr - Ci) / max(Cs - Ci, 1e-9)
    ratio = float(np.clip(ratio, 1e-6, 1.0 - 1e-6))

    beta = float(erfcinv(ratio))
    if beta <= 0 or D_a <= 0:
        return 999.0

    t_i = (cover / (2.0 * beta)) ** 2 / D_a
    return float(np.clip(t_i, 0.0, 999.0))


def chloride_service_life(mix: MixDesign,
                           cover: float = 40.0,
                           exposure: str = 'XS2',
                           T: float = 293.15) -> float:
    """
    Initiation service life (years) against chloride-induced corrosion.
    """
    Cs = CS_BY_CLASS.get(exposure, 3.0)
    D_a, m = chloride_diffusion_coefficient(mix, T)
    return time_to_corrosion_initiation(cover, D_a, Cs)
