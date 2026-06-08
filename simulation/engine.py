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
"""

import numpy as np
from typing import Optional

from .price_path import simulate
from .noise_trader import batch_fee_revenue


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

    # ── Pool P&L ────────────────────────────────────────────────────────
    # V(P_T) - V(P_0)  normalised by V0
    pool_pnl = np.array([
        (amm.pool_value(prices[-1, j]) - amm.pool_value(prices[0, j])) / amm.V0
        for j in range(n_p)
    ])

    # ── Analytical LVR  ─────────────────────────────────────────────────
    # LVR_T = ∫ℓ(σ,P_t)dt  ≈ Σ_t ℓ(σ,P_t)·Δt
    dt_days = dt_minutes / (60.0 * 24.0)
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
        rb_pnl = _rebalancing_pnl(amm, prices, dt_minutes, freq, n_paths, n_steps)
        rebal_pnl[freq]  = rb_pnl / amm.V0
        hedged_pnl[freq] = pool_pnl - rebal_pnl[freq]

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
def _rebalancing_pnl(
    amm,
    prices: np.ndarray,
    dt_minutes: int,
    freq_minutes: int,
    n_paths: int,
    n_steps: int,
) -> np.ndarray:
    """
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
    steps_per_rebal = max(1, freq_minutes // dt_minutes)
    rb_pnl = np.zeros(n_paths)

    for j in range(n_paths):
        pnl_j = 0.0
        for t in range(n_steps):
            # Determine the last rebalancing step index
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