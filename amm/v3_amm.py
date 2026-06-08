"""
v3_amm.py
---------
Uniswap V3 concentrated-liquidity AMM (single-range position).

Invariant (range order over [Pa, Pb)):
    f(x,y) = (x + L/√Pb)^(1/2) · (y + L·√Pa)^(1/2)

Pool value function (Milionis et al., Example 4):
    V(P) = L·(2√P - P/√Pb - √Pa)   for P ∈ [Pa, Pb)
    V(P) = L·(1/√Pa - 1/√Pb)·P     for P < Pa   (all in Y)
    V(P) = L·(√Pb - √Pa)            for P >= Pb  (all in X… wait: all in Y)

Demand curves (in-range):
    x*(P) = L·(1/√P - 1/√Pb)
    y*(P) = L·(√P  - √Pa)

Marginal liquidity (in-range):
    |x*'(P)| = L / (2·P^{3/2})    ← same coefficient as V2, but V(P) is smaller

Instantaneous LVR (in-range):
    ℓ(σ,P) = L·σ²/(4·√P)

Out-of-range: marginal_liquidity = 0 → ℓ = 0.

LVR/TVL can be very large for narrow ranges (capital efficiency ↔ LVR tradeoff).
"""

from math import sqrt
import numpy as np
from .base_amm import BaseAMM


class UniswapV3AMM(BaseAMM):
    """
    Single-tick-range Uniswap V3 position.

    Parameters
    ----------
    initial_price : float   current (and initial) price P0
    initial_tvl   : float   target TVL at P0 (used to solve for L)
    fee_tier      : float   fee as decimal
    price_low     : float   Pa — lower range boundary
    price_high    : float   Pb — upper range boundary
    range_label   : str     human-readable tag (e.g. "wide", "narrow")
    """

    def __init__(self, initial_price, initial_tvl, fee_tier,
                 price_low, price_high, range_label=""):
        self.Pa = price_low
        self.Pb = price_high
        self.range_label = range_label
        label = f"UniswapV3[{range_label}]" if range_label else "UniswapV3"
        super().__init__(initial_price, initial_tvl, fee_tier, label)

    # ------------------------------------------------------------------
    def _initialize(self, initial_price: float, initial_tvl: float) -> None:
        # Solve for L such that V(P0) = initial_tvl
        # V(P0) = L·(2√P0 - P0/√Pb - √Pa)
        sp  = sqrt(initial_price)
        sa  = sqrt(self.Pa)
        sb  = sqrt(self.Pb)
        # clamp to range
        sp_c = max(min(sp, sb), sa)
        denom = 2.0 * sp_c - sp_c**2 / sb - sa
        if denom <= 0:
            raise ValueError(
                f"V3 denom <= 0 at P0={initial_price} with range [{self.Pa},{self.Pb}]. "
                "Ensure Pa < P0 < Pb."
            )
        self.L = initial_tvl / denom

    # ------------------------------------------------------------------
    def _in_range(self, P: float) -> bool:
        return self.Pa <= P < self.Pb

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------
    def pool_value(self, P: float) -> float:
        sa = sqrt(self.Pa)
        sb = sqrt(self.Pb)
        if P < self.Pa:
            # fully in Y — no, fully in X
            return self.L * (1.0 / sa - 1.0 / sb) * P
        elif P >= self.Pb:
            # fully in Y
            return self.L * (sb - sa)
        else:
            sp = sqrt(P)
            return self.L * (2.0 * sp - P / sb - sa)

    def x_star(self, P: float) -> float:
        sb = sqrt(self.Pb)
        if P < self.Pa:
            sa = sqrt(self.Pa)
            return self.L * (1.0 / sa - 1.0 / sb)
        elif P >= self.Pb:
            return 0.0
        else:
            return self.L * (1.0 / sqrt(P) - 1.0 / sb)

    def y_star(self, P: float) -> float:
        sa = sqrt(self.Pa)
        if P < self.Pa:
            return 0.0
        elif P >= self.Pb:
            sb = sqrt(self.Pb)
            return self.L * (sb - sa)
        else:
            return self.L * (sqrt(P) - sa)

    def marginal_liquidity(self, P: float) -> float:
        """|x*'(P)| = L/(2P^{3/2}) when in range, 0 otherwise."""
        if not self._in_range(P):
            return 0.0
        return self.L / (2.0 * P ** 1.5)

    # Override with closed-form
    def lvr_rate(self, sigma: float, P: float) -> float:
        if not self._in_range(P):
            return 0.0
        return self.L * sigma**2 / (4.0 * sqrt(P))

    def lvr_rate_normalized(self, sigma: float, P: float) -> float:
        if not self._in_range(P):
            return 0.0
        v = self.pool_value(P)
        if v <= 0:
            return 0.0
        return self.lvr_rate(sigma, P) / v

    # ------------------------------------------------------------------
    # Slippage (in-range CPMM with virtual reserves)
    # ------------------------------------------------------------------
    def get_amount_out(self, delta_x: float) -> float:
        """
        Sell `delta_x` risky asset using V3 virtual-reserves formula.
        Virtual reserves at P0:
            x_virt = x*(P0) + L/√Pb
            y_virt = y*(P0) + L·√Pa
        Then CPMM:  x_virt·y_virt = const
        """
        sa = sqrt(self.Pa)
        sb = sqrt(self.Pb)
        sp = sqrt(self.P0)

        x_real = self.L * (1.0 / sp - 1.0 / sb)
        y_real = self.L * (sp - sa)

        x_virt = x_real + self.L / sb
        y_virt = y_real + self.L * sa
        k = x_virt * y_virt

        dx_after_fee = delta_x * (1.0 - self.fee_tier)
        y_virt_new   = k / (x_virt + dx_after_fee)
        dy           = y_virt - y_virt_new          # numeraire received (positive)
        return dy

    # ------------------------------------------------------------------
    def range_utilization(self, prices: np.ndarray) -> float:
        """Fraction of time steps where price is inside [Pa, Pb)."""
        return np.mean((prices >= self.Pa) & (prices < self.Pb))

    def capital_efficiency_vs_v2(self) -> float:
        """
        Ratio of V3 fee per unit liquidity to V2 fee per unit liquidity.
        Approximated as √(Pb/Pa) for symmetric ranges around P0.
        """
        return sqrt(self.Pb / self.Pa)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (f"UniswapV3(L={self.L:.4f}, range=[{self.Pa},{self.Pb}], "
                f"P0={self.P0}, TVL={self.V0:.0f}, fee={self.fee_tier*100:.2f}%)")