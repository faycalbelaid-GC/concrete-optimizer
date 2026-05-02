"""
Concrete mix design — composition and derived properties.
k-value concept for SCMs: EN 206 / fib Bulletin 34.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class MixDesign:
    """Concrete mix design per m³ of fresh concrete."""
    cement: float           # kg/m³  CEM I Portland
    water: float            # kg/m³
    fly_ash: float = 0.0    # kg/m³  Class F or C
    slag: float = 0.0       # kg/m³  GGBS
    silica_fume: float = 0.0  # kg/m³  condensed SF
    fine_agg: float = 0.0   # kg/m³  sand
    coarse_agg: float = 0.0 # kg/m³  crushed stone / gravel

    # k-value efficiency factors (EN 206-1 §5.2.5)
    K_FA: float = field(default=0.40, repr=False)
    K_SL: float = field(default=0.60, repr=False)
    K_SF: float = field(default=2.00, repr=False)

    @property
    def w_c(self) -> float:
        """Water-to-cement ratio."""
        return self.water / max(self.cement, 1.0)

    @property
    def effective_binder(self) -> float:
        """Equivalent cement content using k-values."""
        return (self.cement
                + self.K_FA * self.fly_ash
                + self.K_SL * self.slag
                + self.K_SF * self.silica_fume)

    @property
    def w_b(self) -> float:
        """Water-to-effective-binder ratio."""
        return self.water / max(self.effective_binder, 1.0)

    @property
    def total_binder(self) -> float:
        return self.cement + self.fly_ash + self.slag + self.silica_fume

    @property
    def scm_fraction(self) -> float:
        """Supplementary cementitious material mass fraction."""
        tb = self.total_binder
        if tb <= 0:
            return 0.0
        return (self.fly_ash + self.slag + self.silica_fume) / tb

    @property
    def volume(self) -> float:
        """Total mix volume (m³). Density references from Neville 2011."""
        rho = {
            'cement': 3150, 'water': 1000, 'fa': 2200,
            'sl': 2900, 'sf': 2200, 'fine': 2650, 'coarse': 2700,
        }
        return (self.cement / rho['cement']
                + self.water / rho['water']
                + self.fly_ash / rho['fa']
                + self.slag / rho['sl']
                + self.silica_fume / rho['sf']
                + self.fine_agg / rho['fine']
                + self.coarse_agg / rho['coarse'])

    def is_valid(self) -> bool:
        if any(v < 0 for v in [self.cement, self.water, self.fly_ash,
                                self.slag, self.silica_fume]):
            return False
        if self.cement < 100:
            return False
        if not 0.25 <= self.w_c <= 0.80:
            return False
        return True
