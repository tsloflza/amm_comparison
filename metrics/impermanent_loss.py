"""
metrics/impermanent_loss.py
---------------------------
Compute impermanent loss (Loss-Versus-Holding, LVH) curves.

LVH(P) = R_HODL(P) - V(P)   [Milionis et al., Section 8]

where R_HODL holds the LP's initial basket at the new price and V(P) is
the mark-to-market pool value.  Always >= 0.

Public API
----------
    compute_il_curve(amm, price_ratios)           ->  np.ndarray  (% of V0)
    compare_il(amms, price_ratios)                ->  dict
    compute_il_over_paths(amm, prices)            ->  np.ndarray  (n_paths,)
    lvh_lvr_ratio(amm, prices, sigma)             ->  float
"""

import numpy as np


def compute_il_curve(amm, price_ratios: np.ndarray) -> np.ndarray:
    """
    IL curve as a function of price ratio r = P/P0.

    Parameters
    ----------
    amm          : any BaseAMM subclass
    price_ratios : 1-D array of r values, e.g. np.linspace(0.1, 5.0, 200)

    Returns
    -------
    il_pct : np.ndarray  — IL as % of initial position value (V0)
    """
    prices = price_ratios * amm.P0
    return np.array([amm.impermanent_loss_pct(P) for P in prices])


def compare_il(amms: list, price_ratios: np.ndarray) -> dict:
    """
    Compute IL curves for multiple AMMs.

    Returns
    -------
    dict  amm.name → np.ndarray of IL %
    """
    return {amm.name: compute_il_curve(amm, price_ratios)
            for amm in amms}


def compute_il_over_paths(amm, prices: np.ndarray) -> np.ndarray:
    """
    Compute final IL for each Monte Carlo price path.

    Parameters
    ----------
    prices : (n_steps+1, n_paths) price array

    Returns
    -------
    il_terminal : (n_paths,) — final IL as fraction of V0
    """
    final_prices = prices[-1, :]               # (n_paths,)
    return np.array([amm.impermanent_loss(P) for P in final_prices])


def lvh_lvr_ratio(amm, prices: np.ndarray, sigma: float, dt_minutes: float = 1.0) -> float:
    """
    Compute mean LVH / mean LVR across Monte Carlo paths.

    A ratio > 1 means IL overstates LVR (noisy due to market risk).
    A ratio near 1 means IL is a good proxy for LVR (happens only when prices
    don't drift, i.e. risk-neutral measure is exact).

    Parameters
    ----------
    prices     : (n_steps+1, n_paths)
    sigma      : daily volatility
    dt_minutes : simulation step size in minutes

    Returns
    -------
    ratio : float
    """
    dt_days = dt_minutes / (60.0 * 24.0)
    n_steps = prices.shape[0] - 1
    n_paths = prices.shape[1]

    # LVH: final IL across paths
    lvh_vals = compute_il_over_paths(amm, prices) * amm.V0   # back to USD

    # Analytical LVR per path
    lvr_vals = np.zeros(n_paths)
    for j in range(n_paths):
        for t in range(n_steps):
            lvr_vals[j] += amm.lvr_rate(sigma, prices[t, j]) * dt_days

    mean_lvh = np.mean(lvh_vals)
    mean_lvr = np.mean(lvr_vals)
    return mean_lvh / mean_lvr if mean_lvr > 0 else np.nan