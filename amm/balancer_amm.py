"""
balancer_amm.py
---------------
Balancer Weighted Pool (2-asset, fixed weights).

Invariant:  x^θ · y^(1-θ) = k   (weighted geometric mean)

Pool value function (Milionis et al., Example 2):
    V(P) = L · θ^θ · (1-θ)^(1-θ) · P^θ

where L = k / (θ^θ · (1-θ)^(1-θ))  normalisation constant.

Demand curves:
    x*(P) = L · (θ/(1-θ))^(1-θ) · P^(-(1-θ))   = θ·V(P)/P
    y*(P) = L · ((1-θ)/θ)^θ   · P^θ              = (1-θ)·V(P)

Marginal liquidity:
    |x*'(P)| = θ(1-θ)·V(P)/P²

Instantaneous LVR:
    ℓ(σ,P) = σ²·θ·(1-θ)/2 · V(P)
    ℓ/V    = σ²·θ·(1-θ)/2   (constant — independent of price)

Special case θ=0.5: identical to Uniswap V2 (ℓ/V = σ²/8).

Slippage: derived from power-law bonding curve.
"""

from math import sqrt
import numpy as np
from .base_amm import BaseAMM


class BalancerWeightedAMM(BaseAMM):
    """
    2-asset Balancer-style weighted pool.

    Parameters
    ----------
    theta : float
        Weight of the risky asset X (0 < theta < 1).
        theta=0.5  → 50/50 pool (equivalent to Uniswap V2)
        theta=0.8  → 80/20 pool (LP holds more of the risky asset)
        theta=0.2  → 20/80 pool (LP holds less of the risky asset)
    """

    def __init__(self, initial_price: float, initial_tvl: float,
                 fee_tier: float, theta: float):
        if not (0.0 < theta < 1.0):
            raise ValueError("theta must be in (0, 1)")
        self.theta = theta
        pct_x = int(round(theta * 100))
        pct_y = 100 - pct_x
        super().__init__(initial_price, initial_tvl, fee_tier,
                         f"Balancer({pct_x}/{pct_y})")

    # ------------------------------------------------------------------
    def _initialize(self, initial_price: float, initial_tvl: float) -> None:
        """
        At P0: V(P0) = initial_tvl
        x*(P0) = θ·V(P0)/P0
        y*(P0) = (1-θ)·V(P0)

        V(P) = C · P^θ  where C = L · θ^θ·(1-θ)^(1-θ)
        At P0: C = initial_tvl / P0^θ
        """
        self.C = initial_tvl / (initial_price ** self.theta)

        # Store initial reserves for swap calculations
        self.x0_init = self.theta * initial_tvl / initial_price
        self.y0_init = (1.0 - self.theta) * initial_tvl
        # Invariant: x^θ · y^(1-θ) = k
        self.k = (self.x0_init ** self.theta) * (self.y0_init ** (1.0 - self.theta))

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------
    def pool_value(self, P: float) -> float:
        """V(P) = C · P^θ"""
        return self.C * (P ** self.theta)

    def x_star(self, P: float) -> float:
        """x*(P) = θ·V(P)/P"""
        return self.theta * self.pool_value(P) / P

    def y_star(self, P: float) -> float:
        """y*(P) = (1-θ)·V(P)"""
        return (1.0 - self.theta) * self.pool_value(P)

    def marginal_liquidity(self, P: float) -> float:
        """|x*'(P)| = θ(1-θ)·V(P)/P²"""
        return self.theta * (1.0 - self.theta) * self.pool_value(P) / (P ** 2)

    # Override with closed-form LVR
    def lvr_rate(self, sigma: float, P: float) -> float:
        """ℓ(σ,P) = σ²·θ·(1-θ)/2 · V(P)"""
        return 0.5 * sigma**2 * self.theta * (1.0 - self.theta) * self.pool_value(P)

    def lvr_rate_normalized(self, sigma: float, P: float) -> float:
        """ℓ/V = σ²·θ·(1-θ)/2  (constant)"""
        return 0.5 * sigma**2 * self.theta * (1.0 - self.theta)

    # ------------------------------------------------------------------
    # IL: closed-form for weighted pool
    # IL(r) = r^θ / [θ^θ·(1-θ)^(1-θ)·(θ·r + (1-θ))] - 1
    # where r = P/P0
    # ------------------------------------------------------------------
    def impermanent_loss(self, P: float) -> float:
        r = P / self.P0
        th = self.theta
        # Value of LP position (normalised to 1 at P0)
        lp_normalised = r ** th
        # Value of HODL basket (normalised to 1 at P0)
        hodl_normalised = th * r + (1.0 - th)
        return max(hodl_normalised - lp_normalised, 0.0)

    # ------------------------------------------------------------------
    # Slippage: Balancer swap uses constant-product in weighted form
    # For 2 assets: y_out = y0·(1 - (x0/(x0+dx))^(θ/(1-θ)))
    # ------------------------------------------------------------------
    def get_amount_out(self, delta_x: float) -> float:
        """
        Sell `delta_x` of risky asset using Balancer's weighted swap formula.
        y_out = y0 · [1 - (x0 / (x0 + dx*(1-fee)))^(θ/(1-θ))]
        """
        x0 = self.x0_init
        y0 = self.y0_init
        th = self.theta

        dx_after_fee = delta_x * (1.0 - self.fee_tier)
        exponent = th / (1.0 - th)
        y_out = y0 * (1.0 - (x0 / (x0 + dx_after_fee)) ** exponent)
        return max(y_out, 0.0)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (f"Balancer(θ={self.theta}, C={self.C:.4f}, k={self.k:.4f}, "
                f"P0={self.P0}, TVL={self.V0:.0f}, fee={self.fee_tier*100:.2f}%)")