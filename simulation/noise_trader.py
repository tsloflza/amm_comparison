"""
noise_trader.py
---------------
Poisson noise-trader arrival process.

At each time step, noise traders arrive independently with probability
    p = λ · Δt
where λ is the arrival rate (trades per day) and Δt is the step size in days.

Each trade size is drawn from a log-normal distribution parameterised by
the mean trade size (as a fraction of TVL) and a shape parameter.

This module generates trade schedules and accumulates fee revenue for each AMM.

Usage
-----
    schedule = generate_trade_schedule(prices, lambda_per_day, mean_trade_frac,
                                       dt_minutes, tvl, seed)
    fee_series = accumulate_fees(amm, schedule)
"""

import numpy as np
from typing import Optional


# -----------------------------------------------------------------------
def generate_trade_schedule(
    prices: np.ndarray,          # (n_steps+1, 1) or (n_steps+1,) — single path
    lambda_per_day: float,       # average trades per day
    mean_trade_frac: float,      # mean trade size as fraction of TVL
    tvl: float,                  # pool TVL in USD
    dt_minutes: float = 1.0,
    seed: Optional[int] = None,
) -> dict:
    """
    Generate a noise-trader schedule for a single price path.

    Parameters
    ----------
    prices         : 1-D price array of length n_steps+1
    lambda_per_day : Poisson arrival rate
    mean_trade_frac: mean |delta_x| / x*(P0)
    tvl            : pool TVL (to scale trade size)
    dt_minutes     : step size in minutes
    seed           : random seed

    Returns
    -------
    dict with keys:
        'arrival_mask' : bool array (n_steps,) — True where a trade occurs
        'trade_sizes'  : float array (n_steps,) — |delta_x| for each step
                         (zero if no trade)
        'directions'   : int array (n_steps,)  — +1 buy X, -1 sell X
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    prices_1d = prices.ravel()
    n_steps   = len(prices_1d) - 1
    dt_days   = dt_minutes / (60.0 * 24.0)

    # Poisson arrivals: Bernoulli approx for small dt
    p_arrive  = min(lambda_per_day * dt_days, 1.0)
    arrivals  = rng.random(n_steps) < p_arrive

    # Log-normal trade sizes
    # Mean = mean_trade_frac * x*(P0) ≈ mean_trade_frac * tvl / (2 * P0)
    # We parameterise in units of X
    mean_x    = mean_trade_frac * tvl / prices_1d[0]  # approximate x*(P0)
    sigma_ln  = 0.8   # log-normal shape (gives realistic fat tail)
    mu_ln     = np.log(mean_x) - 0.5 * sigma_ln**2

    raw_sizes = rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=n_steps)
    trade_sizes = np.where(arrivals, raw_sizes, 0.0)

    # Random buy/sell direction
    directions = rng.choice([-1, 1], size=n_steps)

    return {
        "arrival_mask": arrivals,
        "trade_sizes":  trade_sizes,
        "directions":   directions,
        "n_trades":     int(arrivals.sum()),
    }


# -----------------------------------------------------------------------
def accumulate_fees(
    amm,
    schedule: dict,
) -> np.ndarray:
    """
    Given a trade schedule and an AMM, compute cumulative fee revenue over time.

    Fee per trade:
        fee = |delta_x| * P_t * fee_tier   (fee on input, in USD)

    Returns
    -------
    cum_fees : np.ndarray (n_steps,) — cumulative fee in USD
    """
    arrival_mask = schedule["arrival_mask"]
    trade_sizes  = schedule["trade_sizes"]   # in X
    n_steps = len(arrival_mask)

    fees_per_step = np.zeros(n_steps)

    # For fee accumulation we only need trade size * P * fee_tier
    # We approximate P at each step from the simulated price path
    # (passed indirectly; caller should pass prices separately if needed)
    for t in range(n_steps):
        if arrival_mask[t] and trade_sizes[t] > 0:
            # Approximate fee in USD: fee_tier * |dx| * fee_tier already
            # built into get_amount_out; here we just tally it separately
            fee_usd = trade_sizes[t] * amm.P0 * amm.fee_tier
            fees_per_step[t] = fee_usd

    return np.cumsum(fees_per_step)


# -----------------------------------------------------------------------
def accumulate_fees_with_prices(
    amm,
    schedule: dict,
    prices: np.ndarray,
) -> np.ndarray:
    """
    Accumulate fees using actual simulated prices at each step.

    Returns cumulative fee array, shape (n_steps,).
    """
    arrival_mask = schedule["arrival_mask"]
    trade_sizes  = schedule["trade_sizes"]
    n_steps = len(arrival_mask)
    prices_1d = prices.ravel()[:n_steps]

    fees_per_step = np.zeros(n_steps)
    for t in range(n_steps):
        if arrival_mask[t] and trade_sizes[t] > 0:
            fee_usd = trade_sizes[t] * prices_1d[t] * amm.fee_tier
            fees_per_step[t] = fee_usd

    return np.cumsum(fees_per_step)


# -----------------------------------------------------------------------
# Vectorised version: batch over multiple paths
# -----------------------------------------------------------------------
def batch_fee_revenue(
    amm,
    prices: np.ndarray,           # (n_steps+1, n_paths)
    lambda_per_day: float,
    mean_trade_frac: float,
    tvl: float,
    dt_minutes: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """
    Compute total fee revenue over T for each of n_paths price paths.

    Returns
    -------
    total_fees : np.ndarray (n_paths,) — total fee in USD over simulation period
    """
    n_steps, n_paths = prices.shape[0] - 1, prices.shape[1]
    dt_days   = dt_minutes / (60.0 * 24.0)
    p_arrive  = min(lambda_per_day * dt_days, 1.0)

    rng = np.random.default_rng(seed)
    arrivals   = rng.random((n_steps, n_paths)) < p_arrive

    sigma_ln   = 0.8
    mean_x_mat = mean_trade_frac * tvl / prices[0, :]   # (n_paths,) broadcast
    # Draw sizes: (n_steps, n_paths)
    mu_ln_mat = np.log(mean_x_mat) - 0.5 * sigma_ln**2
    # We use a common shape, then scale
    raw_sizes  = rng.lognormal(mean=0.0, sigma=sigma_ln, size=(n_steps, n_paths))
    trade_sizes = raw_sizes * np.exp(mu_ln_mat)[np.newaxis, :]

    # fee_usd per step = trade_size * P * fee_tier
    fee_matrix = np.where(arrivals, trade_sizes * prices[:-1, :] * amm.fee_tier, 0.0)
    total_fees = fee_matrix.sum(axis=0)    # (n_paths,)
    return total_fees