"""
price_path.py
-------------
Geometric Brownian Motion price simulator.

Model (Milionis et al., Section 3):
    dP_t / P_t = σ_t dB^Q_t

Discretisation (Euler-Maruyama, log-normal exact solution):
    P_{t+Δt} = P_t · exp((-σ²/2)·Δt + σ·√Δt·Z_t),  Z_t ~ N(0,1)

This guarantees P_t > 0 always (log-normal paths).

API
---
    simulate(P0, sigma_daily, T_days, dt_minutes, n_paths, seed) -> np.ndarray
        shape: (n_steps+1, n_paths)

    simulate_stochastic_vol(...)  [bonus: Heston-style]
"""

import numpy as np
from typing import Optional


# -----------------------------------------------------------------------
def simulate(
    P0: float,
    sigma_daily: float,
    T_days: int = 30,
    dt_minutes: int = 1,
    n_paths: int = 1000,
    seed: Optional[int] = 42,
    mu_daily: float = 0.0,        # drift (set 0 for risk-neutral Q measure)
) -> np.ndarray:
    """
    Simulate GBM price paths via the exact log-normal discretisation.

    Parameters
    ----------
    P0           : initial price
    sigma_daily  : daily volatility (e.g. 0.05 for 5%)
    T_days       : simulation horizon in days
    dt_minutes   : time step in minutes (default 1 minute)
    n_paths      : number of Monte Carlo paths
    seed         : random seed for reproducibility (None = random)
    mu_daily     : daily drift (0.0 for risk-neutral measure)

    Returns
    -------
    prices : np.ndarray, shape (n_steps+1, n_paths)
        prices[0, :] == P0 for all paths
    """
    if seed is not None:
        np.random.seed(seed)

    # Convert daily params to per-step params
    dt_days    = dt_minutes / (60.0 * 24.0)   # fraction of a day
    n_steps    = int(T_days * 24 * 60 / dt_minutes)

    drift = (mu_daily - 0.5 * sigma_daily**2) * dt_days
    vol   = sigma_daily * np.sqrt(dt_days)

    # Draw all random increments at once: shape (n_steps, n_paths)
    Z = np.random.standard_normal((n_steps, n_paths))

    # Log-returns
    log_returns = drift + vol * Z            # (n_steps, n_paths)

    # Prepend zeros for t=0, then cumsum, then exponentiate
    log_prices = np.vstack([np.zeros(n_paths), log_returns])  # (n_steps+1, n_paths)
    log_prices = np.cumsum(log_prices, axis=0)

    prices = P0 * np.exp(log_prices)
    return prices                             # shape: (n_steps+1, n_paths)


# -----------------------------------------------------------------------
def simulate_multi_sigma(
    P0: float,
    sigma_list,
    T_days: int = 30,
    dt_minutes: int = 1,
    n_paths: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Convenience wrapper: simulate for multiple volatility regimes.

    Returns
    -------
    dict mapping sigma (float) → price array (n_steps+1, n_paths)
    """
    results = {}
    for i, sigma in enumerate(sigma_list):
        results[sigma] = simulate(
            P0=P0,
            sigma_daily=sigma,
            T_days=T_days,
            dt_minutes=dt_minutes,
            n_paths=n_paths,
            seed=seed + i,          # different seed per sigma to decorrelate
        )
    return results


# -----------------------------------------------------------------------
def daily_realized_vol(prices: np.ndarray, steps_per_day: int) -> np.ndarray:
    """
    Compute rolling daily realised volatility from a price matrix.

    Parameters
    ----------
    prices       : (n_steps+1, n_paths)
    steps_per_day: number of simulation steps per calendar day

    Returns
    -------
    vols : (n_days, n_paths) — daily realised vol (annualized)
    """
    log_ret = np.diff(np.log(prices), axis=0)   # (n_steps, n_paths)
    n_days  = log_ret.shape[0] // steps_per_day
    vols = np.zeros((n_days, log_ret.shape[1]))
    for d in range(n_days):
        chunk = log_ret[d * steps_per_day:(d + 1) * steps_per_day, :]
        vols[d] = np.std(chunk, axis=0) * np.sqrt(steps_per_day)
    return vols


# -----------------------------------------------------------------------
def get_simulation_params(sigma_daily: float, T_days: int, dt_minutes: int):
    """Return (n_steps, dt_days) for a given simulation setup."""
    dt_days = dt_minutes / (60.0 * 24.0)
    n_steps = int(T_days * 24 * 60 / dt_minutes)
    return n_steps, dt_days