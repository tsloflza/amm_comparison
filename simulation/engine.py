"""
engine.py
---------
Main simulation engine.  Orchestrates price paths, noise traders, and
metric collection for a single (AMM, sigma, volume_scenario) combination.

For each Monte Carlo path the engine computes:
  - pool P&L            (raw, unhedged)
  - rebalancing P&L     (delta-hedge at a given frequency)
  - hedged P&L          = pool P&L − rebalancing P&L  ≈ fees − LVR
  - analytical LVR      (closed-form / numerical from amm.lvr_rate)
  - fee revenue         (from noise traders)

All values are in USD and normalised by initial TVL.

Performance
-----------
If the AMM exposes vectorised methods (x_star_vec, pool_value_vec,
lvr_rate_vec) the engine evaluates the entire (n_steps, n_paths) price
array in a handful of numpy calls instead of looping over every (t, j).
This gives ~1000–5000× speedup for CurveStableSwapAMM which otherwise
calls scipy.brentq on every step.

For AMMs that do not expose those methods the engine falls back to the
original scalar loops, so all other AMM types are unaffected.
"""

import numpy as np
from typing import Optional

from .price_path import simulate
from .noise_trader import batch_fee_revenue


# -----------------------------------------------------------------------
# Vectorised helpers
# -----------------------------------------------------------------------
def _has_vec(amm) -> bool:
    """Return True if the AMM exposes the vectorised interface."""
    return hasattr(amm, "x_star_vec") and hasattr(amm, "lvr_rate_vec")


def _pool_pnl_vec(amm, prices: np.ndarray) -> np.ndarray:
    """
    Pool P&L for every path using amm.pool_value_vec.
    prices : (n_steps+1, n_paths)
    returns: (n_paths,)  un-normalised
    """
    v_T = amm.pool_value_vec(prices[-1])   # (n_paths,)
    v_0 = amm.pool_value_vec(prices[0])    # (n_paths,)
    return v_T - v_0


def _analytical_lvr_vec(amm, sigma: float, prices: np.ndarray,
                         dt_days: float) -> np.ndarray:
    """
    Analytical LVR for every path using amm.lvr_rate_vec.
    Evaluates all (n_steps × n_paths) prices in a single numpy call.
    prices : (n_steps+1, n_paths)  — we use [:-1] (n_steps rows)
    returns: (n_paths,)  un-normalised
    """
    P_steps = prices[:-1]                        # (n_steps, n_paths)
    ell = amm.lvr_rate_vec(sigma, P_steps.ravel())  # (n_steps*n_paths,)
    ell = ell.reshape(P_steps.shape)             # (n_steps, n_paths)
    return ell.sum(axis=0) * dt_days             # (n_paths,)


def _rebalancing_pnl_vec(amm, prices: np.ndarray,
                          steps_per_rebal: int) -> np.ndarray:
    """
    Rebalancing P&L for every path using amm.x_star_vec.

    For each step t the holding is x*(P_{t_k}) where t_k is the most
    recent rebalancing index ≤ t.  We build the rebalancing-price array
    by index arithmetic (no loop over t or j) and then do a single
    x_star_vec call on the whole array.

    prices : (n_steps+1, n_paths)
    returns: (n_paths,)  un-normalised
    """
    n_steps = prices.shape[0] - 1
    n_paths = prices.shape[1]

    # Index of the last rebalancing event for each step t
    rebal_idx = (np.arange(n_steps) // steps_per_rebal) * steps_per_rebal
    # prices at the last rebal for every (t, j): shape (n_steps, n_paths)
    P_rebal = prices[rebal_idx, :]               # fancy indexing, no copy loop

    # x* at each rebalancing price — one vectorised call
    x_rb = amm.x_star_vec(P_rebal.ravel()).reshape(n_steps, n_paths)

    # Price changes
    dP = prices[1:] - prices[:-1]               # (n_steps, n_paths)

    # Rebalancing P&L = Σ_t x_rb[t,j] · dP[t,j]  for each j
    return (x_rb * dP).sum(axis=0)              # (n_paths,)


# -----------------------------------------------------------------------
def run_simulation(
    amm,
    sigma_daily: float,
    T_days: int = 30,
    dt_minutes: int = 1,
    n_paths: int = 1000,
    seed: int = 42,
    # noise trader params
    lambda_per_day: float = 500,
    mean_trade_frac: float = 0.001,
    # rebalancing frequencies (in minutes) for hedged P&L
    rebal_freqs_min: tuple = (1, 5, 60, 240, 1440),
) -> dict:
    """
    Run a full Monte Carlo simulation for one AMM at one volatility level.

    Returns
    -------
    dict with keys:
        'prices'           : (n_steps+1, n_paths)
        'pool_pnl'         : (n_paths,)  — raw LP P&L / V0
        'hedged_pnl'       : dict freq → (n_paths,)
        'analytical_lvr'   : (n_paths,)
        'fee_revenue'      : (n_paths,)
        'rebal_pnl'        : dict freq → (n_paths,)
        'sigma_daily'      : float
        'tvl'              : float
    """
    prices = simulate(
        P0=amm.P0,
        sigma_daily=sigma_daily,
        T_days=T_days,
        dt_minutes=dt_minutes,
        n_paths=n_paths,
        seed=seed,
    )
    n_steps, n_p = prices.shape[0] - 1, prices.shape[1]
    dt_days = dt_minutes / (60.0 * 24.0)
    use_vec = _has_vec(amm)

    # ── Pool P&L ────────────────────────────────────────────────────────
    # V(P_T) - V(P_0)  normalised by V0
    if use_vec:
        pool_pnl = _pool_pnl_vec(amm, prices) / amm.V0
    else:
        pool_pnl = np.array([
            (amm.pool_value(prices[-1, j]) - amm.pool_value(prices[0, j])) / amm.V0
            for j in range(n_p)
        ])

    # ── Analytical LVR  ─────────────────────────────────────────────────
    # LVR_T = ∫ℓ(σ,P_t)dt  ≈ Σ_t ℓ(σ,P_t)·Δt
    if use_vec:
        analytical_lvr = _analytical_lvr_vec(amm, sigma_daily, prices, dt_days) / amm.V0
    else:
        lvr_increments = np.array([
            [amm.lvr_rate(sigma_daily, prices[t, j]) * dt_days
             for t in range(n_steps)]
            for j in range(n_p)
        ])                        # (n_paths, n_steps)
        analytical_lvr = lvr_increments.sum(axis=1) / amm.V0   # (n_paths,)

    # ── Fee revenue  ─────────────────────────────────────────────────────
    fee_revenue = batch_fee_revenue(
        amm=amm,
        prices=prices,
        lambda_per_day=lambda_per_day,
        mean_trade_frac=mean_trade_frac,
        tvl=amm.V0,
        dt_minutes=dt_minutes,
        seed=seed,
    ) / amm.V0               # (n_paths,) normalised

    # ── Rebalancing P&L at each frequency  ───────────────────────────────
    rebal_pnl    = {}
    hedged_pnl   = {}

    for freq in rebal_freqs_min:
        steps_per_rebal = max(1, round(freq / dt_minutes))
        if use_vec:
            rb_pnl = _rebalancing_pnl_vec(amm, prices, steps_per_rebal) / amm.V0
        else:
            rb_pnl = _rebalancing_pnl_scalar(
                amm, prices, dt_minutes, freq, n_p, n_steps) / amm.V0
        rebal_pnl[freq]  = rb_pnl
        # Delta-hedged LP P&L = (pool_pnl + fees) - rebal_pnl = fees - LVR
        # [Milionis et al., eq. 14]:  hedged = LP P&L - R_T = FEE - LVR
        # pool_pnl here is V(P_T)-V(P_0) only; adding fee_revenue gives full LP P&L
        hedged_pnl[freq] = pool_pnl + fee_revenue - rebal_pnl[freq]

    return {
        "prices":          prices,
        "pool_pnl":        pool_pnl,
        "hedged_pnl":      hedged_pnl,
        "analytical_lvr":  analytical_lvr,
        "fee_revenue":     fee_revenue,
        "rebal_pnl":       rebal_pnl,
        "sigma_daily":     sigma_daily,
        "tvl":             amm.V0,
    }


# -----------------------------------------------------------------------
def _rebalancing_pnl_scalar(
    amm,
    prices: np.ndarray,
    dt_minutes: int,
    freq_minutes: int,
    n_paths: int,
    n_steps: int,
) -> np.ndarray:
    """
    Scalar fallback for AMMs without x_star_vec.

    Compute rebalancing strategy P&L for a given rebalancing frequency.

    The rebalancing strategy holds x*(P_{t_k}) of the risky asset between
    consecutive rebalancing events at times t_k, t_{k+1}.
    P&L contribution from step t to t+1:
        ΔRB_t = x^RB_t · (P_{t+1} − P_t)
    where x^RB_t is the risky-asset holding at the start of the interval.

    [Milionis et al., eq. 22]

    Returns
    -------
    rb_pnl : np.ndarray (n_paths,)  in USD
    """
    steps_per_rebal = max(1, round(freq_minutes / dt_minutes))
    rb_pnl = np.zeros(n_paths)

    for j in range(n_paths):
        pnl_j = 0.0
        for t in range(n_steps):
            last_rebal = (t // steps_per_rebal) * steps_per_rebal
            x_rb = amm.x_star(prices[last_rebal, j])
            pnl_j += x_rb * (prices[t + 1, j] - prices[t, j])
        rb_pnl[j] = pnl_j

    return rb_pnl


# -----------------------------------------------------------------------
def run_experiment_grid(
    amms: list,
    sigma_list: list,
    volume_scenarios: dict,
    T_days: int = 30,
    dt_minutes: int = 1,
    n_paths: int = 1000,
    seed: int = 42,
    rebal_freqs_min: tuple = (1, 5, 60, 240, 1440),
) -> dict:
    """
    Run simulations over a grid of (amm, sigma, volume_scenario).

    Parameters
    ----------
    amms             : list of AMM instances
    sigma_list       : list of daily volatilities
    volume_scenarios : dict of label → (lambda_per_day, mean_trade_frac)
    ...

    Returns
    -------
    results : nested dict  results[amm.name][sigma][vol_label] → run_simulation output
    """
    results = {}
    total = len(amms) * len(sigma_list) * len(volume_scenarios)
    done  = 0

    for amm in amms:
        results[amm.name] = {}
        for sigma in sigma_list:
            results[amm.name][sigma] = {}
            for vol_label, (lam, mfrac) in volume_scenarios.items():
                print(f"  [{done+1}/{total}] {amm.name}  σ={sigma*100:.0f}%  vol={vol_label}")
                res = run_simulation(
                    amm=amm,
                    sigma_daily=sigma,
                    T_days=T_days,
                    dt_minutes=dt_minutes,
                    n_paths=n_paths,
                    seed=seed,
                    lambda_per_day=lam,
                    mean_trade_frac=mfrac,
                    rebal_freqs_min=rebal_freqs_min,
                )
                results[amm.name][sigma][vol_label] = res
                done += 1

    return results