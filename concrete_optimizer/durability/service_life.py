"""
Service life estimation — coupled carbonation + chloride + corrosion model.

Two-phase approach (fib Model Code for SLD §3):
  Phase 1 — Initiation: time for depassivation front to reach rebar.
  Phase 2 — Propagation: time from active corrosion to structural limit state.
"""
from dataclasses import dataclass
import numpy as np
from ..models.mix_design import MixDesign
from .carbonation import carbonation_service_life
from .chloride import chloride_service_life
from .corrosion import corrosion_current_density, time_to_cover_cracking

# Relative humidity representative of each EN 206 exposure class
_RH_BY_CLASS: dict[str, float] = {
    'XC1': 0.60, 'XC2': 0.85, 'XC3': 0.70, 'XC4': 0.65,
    'XS1': 0.75, 'XS2': 0.92, 'XS3': 0.95,
    'XD1': 0.65, 'XD2': 0.80, 'XD3': 0.88,
}


@dataclass
class ServiceLifeResult:
    t_initiation_carbonation: float   # years — Phase 1 (carbonation path)
    t_initiation_chloride: float      # years — Phase 1 (chloride path)
    t_propagation: float              # years — Phase 2
    t_service_total: float            # years — total service life
    governing_mechanism: str          # 'carbonation' or 'chloride'
    corrosion_rate_ua: float          # active i_corr (μA/cm²)


def estimate_service_life(mix: MixDesign,
                           cover: float = 35.0,
                           exposure_carbonation: str = 'XC3',
                           exposure_chloride: str = 'XS2',
                           rh: float = 0.70,
                           T: float = 293.15) -> ServiceLifeResult:
    """
    Full service life estimate combining both degradation paths.

    Parameters
    ----------
    mix                  : concrete mix
    cover                : nominal rebar cover (mm)
    exposure_carbonation : EN 206 carbonation class (XC1–XC4)
    exposure_chloride    : EN 206 chloride class  (XS1–XS3, XD1–XD3)
    rh                   : site relative humidity [0–1]
    T                    : mean annual temperature (K)
    """
    rh_carb = _RH_BY_CLASS.get(exposure_carbonation, rh)

    t_carb = carbonation_service_life(mix, cover, rh_carb)
    t_cl   = chloride_service_life(mix, cover, exposure_chloride, T)

    # Governing mechanism = first to trigger depassivation
    if t_carb <= t_cl:
        mechanism = 'carbonation'
        rh_active = _RH_BY_CLASS.get(exposure_carbonation, rh)
    else:
        mechanism = 'chloride'
        rh_active = _RH_BY_CLASS.get(exposure_chloride, rh)

    t_init = min(t_carb, t_cl)

    i_corr  = corrosion_current_density(mix, rh_active, T, mechanism)
    t_prop  = time_to_cover_cracking(i_corr, cover)

    total = float(np.clip(t_init + t_prop, 0.0, 999.0))

    return ServiceLifeResult(
        t_initiation_carbonation=float(np.clip(t_carb, 0.0, 999.0)),
        t_initiation_chloride=float(np.clip(t_cl,  0.0, 999.0)),
        t_propagation=float(np.clip(t_prop, 0.0, 200.0)),
        t_service_total=total,
        governing_mechanism=mechanism,
        corrosion_rate_ua=float(i_corr),
    )
