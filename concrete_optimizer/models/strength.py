"""
Compressive strength models.
Ref: Abrams (1919), CEB-fip 1990, EN 1992-1-1.
"""
import numpy as np
from .mix_design import MixDesign

# EN 206 characteristic cylinder strength classes
_CLASSES = [
    (8,  'C8/10'),  (12, 'C12/15'), (16, 'C16/20'), (20, 'C20/25'),
    (25, 'C25/30'), (30, 'C30/37'), (35, 'C35/45'), (40, 'C40/50'),
    (45, 'C45/55'), (50, 'C50/60'), (55, 'C55/67'), (60, 'C60/75'),
    (70, 'C70/85'), (80, 'C80/95'), (90, 'C90/105'),
]


def compressive_strength_28d(mix: MixDesign) -> float:
    """
    Mean 28-day cylinder compressive strength (MPa).
    Abrams' law variant calibrated for OPC + SCMs:  fc = A / B^(w/beff)
    Constants A=96.5, B=8.65 from Popovics (1998) regression.
    Silica fume pozzolanic bonus capped at 15% replacement.
    """
    A, B = 96.5, 8.65
    w_b = float(np.clip(mix.w_b, 0.20, 0.90))
    fc = A / (B ** w_b)

    # SF micro-filling + pozzolanic densification (Duval & Kadri 1998)
    sf_frac = mix.silica_fume / max(mix.total_binder, 1.0)
    fc *= (1.0 + 0.30 * min(sf_frac, 0.15))

    return float(fc)


def compressive_strength_t(fc_28: float, t: float,
                            cement_class: str = 'n') -> float:
    """
    Strength at age t (days) — EN 1992-1-1 §3.1.2.
    cement_class: 'r' rapid, 'n' normal, 's' slow.
    """
    s = {'r': 0.20, 'n': 0.25, 's': 0.38}.get(cement_class, 0.25)
    beta_cc = float(np.exp(s * (1.0 - np.sqrt(28.0 / max(t, 0.5)))))
    return beta_cc * fc_28


def characteristic_strength(fc_mean: float, cov: float = 0.15) -> float:
    """fck = fcm - 1.645·σ  (EN 206 §7.4)."""
    return fc_mean - 1.645 * cov * fc_mean


def concrete_class(fck: float) -> str:
    """EN 206 exposure class label from fck (cylinder, MPa)."""
    for limit, label in _CLASSES:
        if fck <= limit:
            return label
    return 'C90/105'
