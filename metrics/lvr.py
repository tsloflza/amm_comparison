"""
metrics/lvr.py
--------------
Loss-Versus-Rebalancing (LVR) metrics.

Two complementary approaches [Milionis et al., Section 4 & 6]:

1. ANALYTICAL LVR  (Theorem 1)
   LVR_T = ∫₀ᵀ ℓ(σₜ, Pₜ) dt
   where ℓ(σ,P) = σ²P²/2 · |x*'(P)|
   Computed by summing the closed-form (or numerical) per-step rate over the path.

2. EMPIRICAL (DELTA-HEDGE) LVR  (Section 6, eq. 21)
   LVR_T^emp = R_T - V_T
   = [V_0 + Σₜ x*(P_{t_k})·(P_{t+1} - P_t)] - V(P_T)
   = Rebalancing P&L - Pool P&L
   This is path-observable and requires no knowledge of σ.

Both are normalised by V0 so results are in units of "fraction of TVL".

Public API
----------
    analytical_lvr_path(amm, prices, sigma, dt_minutes)   ->  (n_paths,)
    analytical_lvr_rate_series(amm, prices, sigma, dt_min) ->  (n_steps, n_paths)
    empirical_lvr(amm, prices, rebal_freq_min, dt_min)     ->  (n_paths,)
    lvr_per_tvl(amm, sigma, P)                             ->  float
    lvr_vs_sigma(amm, sigma_list, P)                       ->  np.ndarray
    break_even_volume(amm, sigma, fee_per_vol)             ->  float
"""

import numpy as np


# -----------------------------------------------------------------------
# Analytical LVR
# -----------------------------------------------------------------------
def analytical_lvr_path(
    amm,
    prices: np.ndarray,
    sigma: float,
    dt_minutes: float = 1.0,
) -> np.ndarray:
    """
    Sum instantaneous LVR over each simulated price path.

    LVR_T ≈ Σₜ ℓ(σ, Pₜ) · Δt

    Parameters
    ----------
    prices     : (n_steps+1, n_paths)
    sigma      : daily volatility (constant)
    dt_minutes : step size in minutes

    Returns
    -------
    lvr : (n_paths,) total LVR normalised by V0
    """
    dt_days = dt_minutes / (60.0 * 24.0)
    # prices[:-1] = steps 0…n_steps-1  (shape: n_steps × n_paths)
    lvr_matrix = np.array(
        [[amm.lvr_rate(sigma, prices[t, j]) * dt_days
          for j in range(prices.shape[1])]
         for t in range(prices.shape[0] - 1)]
    )                                    # (n_steps, n_paths)
    return lvr_matrix.sum(axis=0) / amm.V0


def analytical_lvr_rate_series(
    amm,
    prices: np.ndarray,
    sigma: float,
    dt_minutes: float = 1.0,
) -> np.ndarray:
    """
    Per-step LVR increments (useful for time-series plots).

    Returns
    -------
    lvr_series : (n_steps, n_paths) normalised by V0
    """
    dt_days = dt_minutes / (60.0 * 24.0)
    n_steps, n_paths = prices.shape[0] - 1, prices.shape[1]
    out = np.zeros((n_steps, n_paths))
    for t in range(n_steps):
        for j in range(n_paths):
            out[t, j] = amm.lvr_rate(sigma, prices[t, j]) * dt_days / amm.V0
    return out


# -----------------------------------------------------------------------
# Empirical (delta-hedge) LVR
# -----------------------------------------------------------------------
def empirical_lvr(
    amm,
    prices: np.ndarray,
    rebal_freq_min: int = 1,
    dt_minutes: int = 1,
) -> np.ndarray:
    """
    Empirical LVR via the delta-hedge decomposition [eq. 21].

    LVR^emp = Rebalancing P&L - Pool P&L   (both normalised by V0)

    Pool P&L    = V(P_T) - V(P_0)
    Rebalncing P&L = Σₜ x*(P_{t_k}) · (P_{t+1} - P_t)   [eq. 22]

    Parameters
    ----------
    prices         : (n_steps+1, n_paths)
    rebal_freq_min : rebalancing interval in minutes
    dt_minutes     : simulation step size in minutes

    Returns
    -------
    lvr_emp : (n_paths,) normalised by V0
    """
    n_steps, n_paths = prices.shape[0] - 1, prices.shape[1]
    # round() avoids collision when rebal_freq_min < dt_minutes (e.g. fast mode)
    steps_per_rebal  = max(1, round(rebal_freq_min / dt_minutes))

    pool_pnl = np.array([
        amm.pool_value(prices[-1, j]) - amm.pool_value(prices[0, j])
        for j in range(n_paths)
    ]) / amm.V0

    rebal_pnl = np.zeros(n_paths)
    for j in range(n_paths):
        pnl_j = 0.0
        for t in range(n_steps):
            last_rebal = (t // steps_per_rebal) * steps_per_rebal
            x_rb = amm.x_star(prices[last_rebal, j])
            pnl_j += x_rb * (prices[t + 1, j] - prices[t, j])
        rebal_pnl[j] = pnl_j / amm.V0

    return rebal_pnl - pool_pnl


# -----------------------------------------------------------------------
# Scalar helpers
# -----------------------------------------------------------------------
def lvr_per_tvl(amm, sigma: float, P: float) -> float:
    """Instantaneous LVR / V(P) — dimensionless, per unit time (daily)."""
    return amm.lvr_rate_normalized(sigma, P)


def lvr_vs_sigma(amm, sigma_list, P: float = None) -> np.ndarray:
    """
    LVR/TVL as a function of sigma at price P (default P0).

    Returns
    -------
    lvr_norm : (len(sigma_list),) daily LVR / TVL
    """
    P = P or amm.P0
    return np.array([amm.lvr_rate_normalized(s, P) for s in sigma_list])


def daily_lvr_analytical(amm, sigma: float) -> float:
    """
    Expected daily LVR / TVL under constant-σ GBM at price P0.
    For V2: σ²/8.  For Balancer(θ): σ²·θ·(1-θ)/2.
    """
    return amm.lvr_rate_normalized(sigma, amm.P0)


# -----------------------------------------------------------------------
# Break-even volume
# -----------------------------------------------------------------------
def break_even_volume(
    amm,
    sigma: float,
    fee_per_dollar_volume: float = None,
) -> float:
    """
    Minimum daily trading volume (as fraction of TVL) such that
    fee revenue equals LVR.

    fee_revenue_per_day = fee_tier × daily_volume_frac × TVL / P0 × P0
                        = fee_tier × daily_volume_frac × TVL

    Break-even condition:  fee_tier × vol_frac = LVR/TVL
    → vol_frac = LVR/TVL / fee_tier

    Parameters
    ----------
    fee_per_dollar_volume : override fee rate (defaults to amm.fee_tier)

    Returns
    -------
    vol_frac : break-even daily volume as a fraction of TVL
    """
    fee = fee_per_dollar_volume if fee_per_dollar_volume is not None else amm.fee_tier
    lvr_norm = daily_lvr_analytical(amm, sigma)
    if fee <= 0:
        return np.inf
    return lvr_norm / fee