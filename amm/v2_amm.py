"""
v2_amm.py
---------
Uniswap V2 (full-range constant-product market maker).

Invariant: sqrt(x·y) = L   ⟺   x·y = L²

Pool value function:
    V(P) = 2·L·√P              [Milionis et al., Example 3, eq. 16]

Demand curves:
    x*(P) = L / √P
    y*(P) = L · √P

Marginal liquidity:
    |x*'(P)| = L / (2·P^{3/2})

Instantaneous LVR:
    ℓ(σ,P) = L·σ²/(4·√P)
    ℓ/V    = σ²/8   (constant — independent of price)

Extends BaseAMM and is consistent with the reference v2_math.py.
"""

from math import sqrt
import numpy as np
from .base_amm import BaseAMM


class UniswapV2AMM(BaseAMM):
    """Full-range CPMM (Uniswap V2)."""

    def __init__(self, initial_price: float, initial_tvl: float, fee_tier: float):
        super().__init__(initial_price, initial_tvl, fee_tier, "UniswapV2")

    def _initialize(self, initial_price: float, initial_tvl: float) -> None:
        # V(P0) = 2·L·√P0 = initial_tvl  →  L = initial_tvl / (2·√P0)
        self.L = initial_tvl / (2.0 * sqrt(initial_price))

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------
    def pool_value(self, P: float) -> float:
        """V(P) = 2·L·√P"""
        return 2.0 * self.L * sqrt(P)

    def x_star(self, P: float) -> float:
        """x*(P) = L / √P"""
        return self.L / sqrt(P)

    def y_star(self, P: float) -> float:
        """y*(P) = L·√P"""
        return self.L * sqrt(P)

    def marginal_liquidity(self, P: float) -> float:
        """|x*'(P)| = L / (2·P^{3/2})"""
        return self.L / (2.0 * P ** 1.5)

    # Override for closed-form speed
    def lvr_rate(self, sigma: float, P: float) -> float:
        """ℓ(σ,P) = L·σ²/(4·√P)"""
        return self.L * sigma**2 / (4.0 * sqrt(P))

    def lvr_rate_normalized(self, sigma: float, P: float) -> float:
        """ℓ/V = σ²/8  (closed-form constant)"""
        return sigma**2 / 8.0

    # impermanent_loss is intentionally not overridden here.
    # The base-class implementation computes (HODL − AMM) / V0, which for V2 gives
    #   IL(r) = (r+1)/2 − √r  where r = P/P0.
    # The closed-form 1 − 2√r/(1+r) equals IL/HODL_VALUE (not IL/V0) and would
    # be inconsistent with every other AMM's normalisation.  [Bug fix]

    # ------------------------------------------------------------------
    # Slippage: x*y = k  →  y_out = k/(x+Δx) - y
    # ------------------------------------------------------------------
    def get_amount_out(self, delta_x: float) -> float:
        """
        Sell `delta_x` of the risky asset for numeraire.
        Applies fee to the input before executing the CPMM swap.
        delta_x > 0 → selling X (LP receives Y).
        Returns Y received.
        """
        x0 = self.x_star(self.P0)
        y0 = self.y_star(self.P0)
        k  = x0 * y0

        dx_after_fee = delta_x * (1.0 - self.fee_tier)
        y_out = y0 - k / (x0 + dx_after_fee)
        return y_out

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (f"UniswapV2(L={self.L:.4f}, P0={self.P0}, "
                f"TVL={self.V0:.0f}, fee={self.fee_tier*100:.2f}%)")