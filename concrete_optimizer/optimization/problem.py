"""
Multi-objective concrete mix optimisation problem for NSGA-II.

Decision variables (6D real vector)
────────────────────────────────────
  x[0]  cement         (kg/m³)              [200, 550]
  x[1]  fly_ash_ratio  (fraction of binder) [0.00, 0.50]
  x[2]  slag_ratio     (fraction of binder) [0.00, 0.70]
  x[3]  sf_ratio       (fraction of binder) [0.00, 0.15]
  x[4]  w_b            (water/eff-binder)   [0.30, 0.65]
  x[5]  fine_ratio     (sand / total agg)   [0.40, 0.65]

Objectives (all minimised by convention)
────────────────────────────────────────
  f1  =  CO2 embodied (kg CO2eq / m³)          → minimise
  f2  = −f'c (MPa)                              → minimise  (= maximise f'c)
  f3  = −service life (years)                   → minimise  (= maximise SL)

Constraints (hard)
──────────────────
  g1  total SCM fraction ≤ 0.85
  g2  f'c ≥ fc_min  (during search: ≥ 0.80 · fc_min for diversity)
  g3  cement ≥ 200 kg/m³
"""
import numpy as np
from ..models.mix_design import MixDesign
from ..models.strength import compressive_strength_28d
from ..models.carbon_footprint import embodied_co2
from ..durability.service_life import estimate_service_life


class ConcreteMixProblem:

    BOUNDS: list[tuple[float, float]] = [
        (200.0, 550.0),   # cement
        (0.00,  0.50),    # FA ratio
        (0.00,  0.70),    # slag ratio
        (0.00,  0.15),    # SF ratio
        (0.30,  0.65),    # w/b
        (0.40,  0.65),    # fine_ratio
    ]

    def __init__(self,
                 fc_min: float = 25.0,
                 cover: float = 35.0,
                 exposure_carb: str = 'XC3',
                 exposure_cl: str = 'XS2',
                 rh: float = 0.70,
                 T: float = 293.15):
        self.fc_min        = fc_min
        self.cover         = cover
        self.exposure_carb = exposure_carb
        self.exposure_cl   = exposure_cl
        self.rh            = rh
        self.T             = T
        self.bounds        = self.BOUNDS

    # ------------------------------------------------------------------
    def decode(self, x: np.ndarray) -> MixDesign:
        """Map gene vector → MixDesign ensuring volume ≈ 1 m³."""
        cement, fa_r, sl_r, sf_r, w_b, fine_r = float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])

        # Clip SCM fractions so they sum ≤ 0.85
        total_scm = fa_r + sl_r + sf_r
        if total_scm > 0.85:
            scale = 0.85 / total_scm
            fa_r, sl_r, sf_r = fa_r * scale, sl_r * scale, sf_r * scale

        cement_frac = max(1.0 - fa_r - sl_r - sf_r, 0.15)
        total_binder = cement / cement_frac
        fly_ash = fa_r * total_binder
        slag    = sl_r * total_binder
        sf      = sf_r * total_binder

        # Effective binder for water demand (k-values as in MixDesign defaults)
        b_eff = cement + 0.40 * fly_ash + 0.60 * slag + 2.00 * sf
        water = w_b * b_eff

        # Aggregate volume fills the balance (2 % air voids)
        rho = dict(cement=3150, fa=2200, sl=2900, sf=2200, water=1000,
                   fine=2650, coarse=2700)
        v_paste = (cement / rho['cement'] + fly_ash / rho['fa']
                   + slag / rho['sl']    + sf / rho['sf']
                   + water / rho['water'])
        v_agg = max(1.0 - v_paste - 0.02, 0.30)

        fine_agg   = fine_r * v_agg * rho['fine']
        coarse_agg = (1.0 - fine_r) * v_agg * rho['coarse']

        return MixDesign(
            cement=cement, water=water,
            fly_ash=fly_ash, slag=slag, silica_fume=sf,
            fine_agg=fine_agg, coarse_agg=coarse_agg,
        )

    # ------------------------------------------------------------------
    def is_feasible(self, x: np.ndarray) -> bool:
        mix = self.decode(x)
        if not mix.is_valid():
            return False
        # Soft lower-bound on strength during search (wider exploration)
        if compressive_strength_28d(mix) < self.fc_min * 0.75:
            return False
        return True

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Return objective vector [CO2, -fc, -SL] (all minimised)."""
        mix = self.decode(x)
        fc  = compressive_strength_28d(mix)
        co2 = embodied_co2(mix)
        sl  = estimate_service_life(
            mix, self.cover,
            self.exposure_carb, self.exposure_cl,
            self.rh, self.T,
        )
        return np.array([co2, -fc, -sl.t_service_total])
