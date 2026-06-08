"""
base_amm.py
-----------
Abstract base class defining the shared interface for all AMM implementations.

Every AMM must implement:
  - pool_value(P)          : mark-to-market value of pool reserves at price P
  - x_star(P)              : risky-asset demand curve  x*(P)
  - y_star(P)              : numeraire demand curve    y*(P)
  - marginal_liquidity(P)  : |dx*/dP|  — key input to LVR formula
  - lvr_rate(sigma, P)     : instantaneous LVR  ℓ(σ,P) = σ²P²/2 · |x*'(P)|
  - slippage(delta_x)      : price impact for a trade of size delta_x
  - impermanent_loss(P)    : LVH as a fraction of V0  (always >= 0)

Design notes
------------
- `initial_price` and `initial_tvl` are stored so that every AMM starts with
  an identical $TVL at P0=initial_price, enabling fair cross-AMM comparisons.
- `fee_tier` is a decimal (e.g. 0.003 for 30 bps).  Slippage methods take it
  into account; LVR formulas deliberately ignore fees (matching Milionis et al.).
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseAMM(ABC):
    """Abstract base for all AMM implementations."""

    def __init__(self, initial_price: float, initial_tvl: float, fee_tier: float, name: str):
        self.P0 = initial_price
        self.V0 = initial_tvl
        self.fee_tier = fee_tier
        self.name = name
        self._initialize(initial_price, initial_tvl)

    # ------------------------------------------------------------------
    # Initialization (called by __init__ after storing common fields)
    # ------------------------------------------------------------------
    @abstractmethod
    def _initialize(self, initial_price: float, initial_tvl: float) -> None:
        """Set pool-specific parameters (reserves, liquidity, etc.)."""

    # ------------------------------------------------------------------
    # Core AMM interface
    # ------------------------------------------------------------------
    @abstractmethod
    def pool_value(self, P: float) -> float:
        """Mark-to-market value of pool reserves: V(P) = P·x*(P) + y*(P)."""

    @abstractmethod
    def x_star(self, P: float) -> float:
        """Optimal risky-asset holding at price P:  x*(P) = V'(P)."""

    @abstractmethod
    def y_star(self, P: float) -> float:
        """Optimal numeraire holding at price P."""

    @abstractmethod
    def marginal_liquidity(self, P: float) -> float:
        """
        |x*'(P)| = |d x*(P) / dP| — the slope of the demand curve.
        This is V''(P) in absolute value and is the key parameter for LVR.
        """

    # ------------------------------------------------------------------
    # Derived metrics (have sensible defaults, may be overridden)
    # ------------------------------------------------------------------
    def lvr_rate(self, sigma: float, P: float) -> float:
        """
        Instantaneous LVR rate:
            ℓ(σ, P) = σ²P²/2 · |x*'(P)|    [Milionis et al., Theorem 1, eq. 8]
        Units: same as pool_value / time-step.
        """
        return 0.5 * sigma**2 * P**2 * self.marginal_liquidity(P)

    def lvr_rate_normalized(self, sigma: float, P: float) -> float:
        """ℓ(σ,P) / V(P) — LVR per dollar of TVL."""
        v = self.pool_value(P)
        if v <= 0:
            return 0.0
        return self.lvr_rate(sigma, P) / v

    def impermanent_loss(self, P: float) -> float:
        """
        Loss-versus-holding (LVH) as a *fraction* of V0:
            LVH(P) = [V0 + x*(P0)·(P - P0)] - V(P)
        Always >= 0.  Returns a non-negative number; multiply by 100 for %.
        """
        x0 = self.x_star(self.P0)
        y0 = self.y_star(self.P0)
        hodl_value = P * x0 + y0           # hold original basket at new price
        amm_value  = self.pool_value(P)
        return max(hodl_value - amm_value, 0.0) / self.V0

    def impermanent_loss_pct(self, P: float) -> float:
        """IL as a percentage."""
        return 100.0 * self.impermanent_loss(P)

    # ------------------------------------------------------------------
    # Slippage
    # ------------------------------------------------------------------
    @abstractmethod
    def get_amount_out(self, delta_x: float) -> float:
        """
        Amount of numeraire y received when selling `delta_x` of the risky asset
        (positive delta_x = sell risky asset into pool).
        Fees are applied internally using self.fee_tier.
        Returns the *gross* amount of y received (before any external fees).
        """

    def slippage_pct(self, delta_x: float) -> float:
        """
        Price impact as % deviation from spot price P0:
            slippage = (P_eff - P0) / P0 * 100
        where P_eff = delta_y / delta_x (average execution price).
        Positive delta_x = LP sells risky asset (price should drop → negative slippage).
        We return the absolute % impact for display convenience.
        """
        if abs(delta_x) < 1e-18:
            return 0.0
        delta_y = self.get_amount_out(delta_x)
        p_eff = delta_y / abs(delta_x)
        return abs(p_eff - self.P0) / self.P0 * 100.0

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"{self.name}(P0={self.P0}, TVL={self.V0:.0f}, fee={self.fee_tier*100:.2f}%)"