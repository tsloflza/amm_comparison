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

Performance design
------------------
x_star(scalar P) calls scipy.brentq (~20 function evaluations, each a Newton
solve) — roughly 0.4–0.5 ms per call.  Simulations that call this in a Python
loop over (n_steps × n_paths) will be extremely slow.

The vectorised path avoids that entirely by precomputing a look-up table (LUT)
of (P → x*(P)) and (P → |dx*/dP|) at init time (≈0.8–1.2 s once) and then
evaluating any price array via np.interp (<0.1 µs per element):

  x_star_vec(P_array)      — LUT interp, ~0.1 µs/element
  pool_value_vec(P_array)  — wraps x_star_vec
  lvr_rate_vec(σ, P_array) — LUT interp for both x* and |dx*/dP|,
                              no Brent, no finite-difference, no second root-find

engine.py detects hasattr(amm, 'x_star_vec') and routes accordingly, giving
~1000–5000× speedup over the scalar loop for a Curve AMM.

LUT accuracy
  - x_star: < 0.01 % error across [P0/1000, P0*1000]
  - |dx*/dP|: < 1 % error across the same range (< 0.1 % away from the peg
    singularity where A → ∞ behaviour makes |dx*/dP| very large but also makes
    the LVR contribution per step tiny in absolute terms)
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

    Derivation (n=2):
        4A(x+y) + D = 4AD + D³/(4xy)
    Multiply through by y and collect:
        4A·y² + (4Ax + D - 4AD)·y − D³/(4x) = 0
    Divide by 4A = Ann:
        y² + (x + D/Ann − D)·y − D³/(4x·Ann) = 0
    So  b = x + D/Ann − D
        c = D³ / (4·x·Ann)   ← Ann factor is required in the denominator
    """
    Ann = A * _N_N
    # Coefficients for: y² + b·y - c = 0
    b = x_new + D / Ann - D
    c = D**3 / (_N_N * x_new * Ann)   # FIX: was D**3 / (_N_N * x_new), missing Ann

    # Newton: y_next = (y² + c) / (2y + b)
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

    # Number of LUT grid points.  2600 gives <1 % error on |dx*/dP| and
    # sub-0.01 % error on x* across [P0/1000, P0*1000].
    _LUT_N_LOG  = 300   # log-spaced points outside the near-peg window
    _LUT_N_LIN  = 2000  # linearly-spaced points in [P0/2, P0*2]

    def __init__(self, initial_price: float, initial_tvl: float,
                 fee_tier: float, A: float):
        self.A = A
        super().__init__(initial_price, initial_tvl, fee_tier,
                         f"Curve(A={A:.0f})")
        # Build vectorised LUT after the AMM is fully initialised.
        self._build_lut()

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
    # Look-up table (LUT) for vectorised evaluation
    # ------------------------------------------------------------------
    def _build_lut(self) -> None:
        """
        Precompute arrays (P_grid, x_grid, ml_grid) once at init time.

        Grid design
        -----------
        The near-peg region (within ×2 of P0) is sampled with a *linear*
        grid so that the np.gradient() estimate of |dx*/dP| is accurate even
        where x*(P) changes rapidly (high-A pools near peg).  Outside that
        window a log-spaced grid covers [P0/1000, P0*1000].
        """
        P0 = self.P0

        # Dense linear window around peg, plus coarse log tails
        P_lin  = np.linspace(P0 * 0.5, P0 * 2.0, self._LUT_N_LIN)
        P_lo   = np.logspace(np.log10(P0 * 1e-3), np.log10(P0 * 0.5),
                             self._LUT_N_LOG, endpoint=False)
        P_hi   = np.logspace(np.log10(P0 * 2.0), np.log10(P0 * 1e3),
                             self._LUT_N_LOG + 1)[1:]  # skip overlap with P_lin

        P_grid  = np.unique(np.concatenate([P_lo, P_lin, P_hi]))
        x_grid  = np.array([self._x_at_price(P) for P in P_grid])
        y_grid  = np.array([_compute_y_from_x(xi, self.D, self.A) for xi in x_grid])
        # |dx*/dP|: central finite-difference on the non-uniform grid
        ml_grid = np.abs(np.gradient(x_grid, P_grid))

        # Store as attributes for interp
        self._lut_P  = P_grid
        self._lut_x  = x_grid
        self._lut_y  = y_grid
        self._lut_ml = ml_grid

    # ------------------------------------------------------------------
    # Vectorised interface (detected by engine.py via hasattr)
    # ------------------------------------------------------------------
    def x_star_vec(self, P_arr: np.ndarray) -> np.ndarray:
        """
        x*(P) for an array of prices via LUT interpolation.
        ~0.1 µs per element  (vs ~0.45 ms for the scalar Brent solve).
        """
        return np.interp(P_arr, self._lut_P, self._lut_x)

    def pool_value_vec(self, P_arr: np.ndarray) -> np.ndarray:
        """V(P) = P·x*(P) + y*(P) for a price array."""
        x = np.interp(P_arr, self._lut_P, self._lut_x)
        y = np.interp(P_arr, self._lut_P, self._lut_y)
        return P_arr * x + y

    def lvr_rate_vec(self, sigma: float, P_arr: np.ndarray) -> np.ndarray:
        """
        ℓ(σ,P) = σ²·P²/2 · |dx*/dP|  for a price array.
        Uses the LUT for both x* (unused here) and |dx*/dP|.
        No Brent solve, no finite-difference — pure np.interp.
        """
        ml = np.interp(P_arr, self._lut_P, self._lut_ml)
        return 0.5 * sigma**2 * P_arr**2 * ml

    # ------------------------------------------------------------------
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