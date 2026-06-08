"""
curve_amm.py
------------
Curve Finance StableSwap AMM (2-asset pool).

Invariant (Egorov 2019):
    A·n^n·Σxi + D = A·n^n·D + D^(n+1) / (n^n · Πxi)

For n=2:
    4A·(x + y) + D = 4A·D + D³/(4xy)

where D is the total invariant (≈ total value when x=y=D/2, i.e. at peg)
and A is the amplification coefficient.

Limiting behaviour:
  A → 0   : constant-product market maker  (x·y = (D/2)²)
  A → ∞   : constant-sum market maker      (x + y = D)

Price (implicit):  ∂F/∂x / ∂F/∂y = 1  at peg; off-peg solved numerically.

We implement:
  - Numerical solve for D given (x,y) via Newton iteration
  - Numerical solve for y given (x, D) for swaps
  - Numerical x*(P), y*(P) via root-finding (portfolio optimization)
  - Marginal liquidity via finite difference
  - LVR via base-class formula (numerical)
"""

from math import sqrt
import numpy as np
from scipy.optimize import brentq
from .base_amm import BaseAMM


_N = 2          # number of assets (always 2 in this implementation)
_N_N = _N ** _N  # 4


# ---------------------------------------------------------------------------
# Utility: Curve invariant D
# ---------------------------------------------------------------------------
def _compute_D(x: float, y: float, A: float) -> float:
    """
    Solve for D given reserves (x, y) and amplification A.
    Uses Newton-Raphson (converges very fast for these equations).
    """
    S = x + y
    if S == 0:
        return 0.0
    D = S  # initial guess
    Ann = A * _N_N
    for _ in range(256):
        D_P = D
        D_P = D_P * D / (_N * x)
        D_P = D_P * D / (_N * y)
        D_prev = D
        D = (Ann * S + D_P * _N) * D / ((Ann - 1) * D + (_N + 1) * D_P)
        if abs(D - D_prev) <= 1:
            break
    return D


def _compute_y_from_x(x_new: float, D: float, A: float) -> float:
    """
    Given new x balance and invariant D, solve for the corresponding y.
    Newton-Raphson on the StableSwap invariant.
    """
    Ann = A * _N_N
    # Coefficients for: y² + b·y - c = 0
    b = x_new + D / Ann - D
    c = D**3 / (_N_N * x_new)

    # Newton: f(y) = y² + b·y - c
    y = D  # initial guess
    for _ in range(256):
        y_prev = y
        y = (y * y + c) / (2 * y + b)
        if abs(y - y_prev) <= 1e-12:
            break
    return max(y, 0.0)


# ---------------------------------------------------------------------------
class CurveStableSwapAMM(BaseAMM):
    """
    Two-asset Curve StableSwap pool.

    Parameters
    ----------
    A : float
        Amplification coefficient (e.g. 100 for typical USDC/DAI).
    """

    def __init__(self, initial_price: float, initial_tvl: float,
                 fee_tier: float, A: float):
        self.A = A
        super().__init__(initial_price, initial_tvl, fee_tier,
                         f"Curve(A={A:.0f})")

    # ------------------------------------------------------------------
    def _initialize(self, initial_price: float, initial_tvl: float) -> None:
        """
        At P0 the pool is balanced: x0·P0 + y0 = initial_tvl,
        and x0 = y0/P0 (equal weights in value terms).
        → y0 = initial_tvl/2,  x0 = initial_tvl/(2·P0)
        """
        y0 = initial_tvl / 2.0
        x0 = initial_tvl / (2.0 * initial_price)
        self.x0 = x0
        self.y0 = y0
        self.D  = _compute_D(x0, y0, self.A)

    # ------------------------------------------------------------------
    # Internal: solve x*(P) by minimizing pool value subject to invariant
    # We use the property: at optimum, the ratio ∂F/∂y / ∂F/∂x = P
    # where F is the invariant. We solve for x given the implicit price.
    # ------------------------------------------------------------------
    def _price_from_x(self, x: float) -> float:
        """
        Implicit pool price (marginal rate of substitution dy/dx) at reserve x.
        ∂F/∂x = 4A·1 + D³/(4x²y) * (1/y + ... )
        Easier: price = -dy_dx | invariant fixed
            = (∂F/∂x) / (∂F/∂y)
        We compute this numerically via finite difference on y(x, D).
        """
        y  = _compute_y_from_x(x, self.D, self.A)
        eps = x * 1e-7
        y2 = _compute_y_from_x(x + eps, self.D, self.A)
        dy_dx = (y2 - y) / eps    # negative
        return -dy_dx             # positive price

    def _x_at_price(self, P: float) -> float:
        """
        Find x such that the implicit pool price equals P.
        Uses Brent's method on [x_min, x_max].
        """
        # bounds: x ∈ (0, 2·D)
        x_lo = 1e-9 * self.D
        x_hi = 1.9999 * self.D

        # f(x) = _price_from_x(x) - P
        try:
            x_sol = brentq(lambda x: self._price_from_x(x) - P,
                           x_lo, x_hi, xtol=1e-12, maxiter=200)
        except ValueError:
            # P outside the feasible range of the pool — clamp to boundary
            p_lo = self._price_from_x(x_lo * 1.001)
            p_hi = self._price_from_x(x_hi * 0.999)
            if P <= p_lo:
                x_sol = x_lo * 1.001
            else:
                x_sol = x_hi * 0.999
        return x_sol

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------
    def x_star(self, P: float) -> float:
        return self._x_at_price(P)

    def y_star(self, P: float) -> float:
        x = self._x_at_price(P)
        return _compute_y_from_x(x, self.D, self.A)

    def pool_value(self, P: float) -> float:
        x = self.x_star(P)
        y = self.y_star(P)
        return P * x + y

    def marginal_liquidity(self, P: float) -> float:
        """
        |dx*/dP| via central finite difference.
        """
        h = P * 1e-5
        x_plus  = self._x_at_price(P + h)
        x_minus = self._x_at_price(P - h)
        return abs((x_plus - x_minus) / (2.0 * h))

    # ------------------------------------------------------------------
    # Slippage: sell delta_x risky asset
    # ------------------------------------------------------------------
    def get_amount_out(self, delta_x: float) -> float:
        """
        Swap delta_x of risky asset (X) for numeraire (Y).
        Applies fee_tier to the input.
        """
        x0 = self.x0
        y0 = self.y0

        dx_after_fee = delta_x * (1.0 - self.fee_tier)
        x_new = x0 + dx_after_fee
        y_new = _compute_y_from_x(x_new, self.D, self.A)
        dy    = y0 - y_new          # Y received by trader
        return max(dy, 0.0)

    # ------------------------------------------------------------------
    # Extra: pool price at current reserves (for simulation)
    # ------------------------------------------------------------------
    def current_price(self) -> float:
        return self._price_from_x(self.x0)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (f"CurveStableSwap(A={self.A}, D={self.D:.4f}, "
                f"P0={self.P0}, TVL={self.V0:.0f}, fee={self.fee_tier*100:.2f}%)")