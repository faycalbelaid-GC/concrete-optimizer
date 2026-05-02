"""
Embodied CO2 of concrete mixes.
Emission factors: IPCC AR5 / Ecoinvent 3.8 / ICE Database v3.0.
"""
from .mix_design import MixDesign

# kg CO2eq per kg of material (cradle-to-gate)
CO2_FACTORS: dict[str, float] = {
    'cement':     0.830,   # CEM I Portland  (IPCC 2021)
    'fly_ash':    0.027,   # power-plant by-product
    'slag':       0.052,   # blast-furnace GGBS
    'silica_fume': 0.014,  # ferrosilicon smelting by-product
    'fine_agg':   0.0048,  # quarry sand
    'coarse_agg': 0.0048,  # crushed aggregate
    'water':      0.0003,  # municipal supply
}


def embodied_co2(mix: MixDesign) -> float:
    """Total embodied CO2 (kg CO2eq / m³ concrete)."""
    return (mix.cement       * CO2_FACTORS['cement']
            + mix.fly_ash    * CO2_FACTORS['fly_ash']
            + mix.slag       * CO2_FACTORS['slag']
            + mix.silica_fume * CO2_FACTORS['silica_fume']
            + mix.fine_agg   * CO2_FACTORS['fine_agg']
            + mix.coarse_agg * CO2_FACTORS['coarse_agg']
            + mix.water      * CO2_FACTORS['water'])


def co2_per_mpa(mix: MixDesign, fc: float) -> float:
    """CO2 efficiency index (kg CO2eq / MPa / m³)."""
    return embodied_co2(mix) / max(fc, 1.0)
