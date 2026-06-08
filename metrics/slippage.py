"""
metrics/slippage.py
-------------------
Compute slippage (price impact) curves for any AMM.

Slippage is defined as the absolute percentage deviation of the average
execution price from the current spot price:

    slippage(Δx) = |P_eff(Δx) - P0| / P0 × 100%

where P_eff = Δy / |Δx|  is the average fill price.

Public API
----------
    compute_slippage_curve(amm, trade_fracs)  ->  np.ndarray
    compare_slippage(amms, trade_fracs)        ->  dict
"""

import numpy as np


def compute_slippage_curve(amm, trade_fracs: np.ndarray) -> np.ndarray:
    """
    Compute slippage (%) for a range of trade sizes.

    Parameters
    ----------
    amm          : any BaseAMM subclass
    trade_fracs  : 1-D array of trade sizes as *fraction of TVL*
                   e.g. np.logspace(-4, -1, 60)  →  0.01% … 10% of TVL

    Returns
    -------
    slippage_pct : np.ndarray, same shape as trade_fracs
    """
    # Convert TVL fractions → absolute X units (using x*(P0))
    x0 = amm.x_star(amm.P0)
    tvl = amm.V0

    slippage = np.zeros(len(trade_fracs))
    for i, frac in enumerate(trade_fracs):
        delta_x = frac * tvl / amm.P0   # approximate: frac * TVL / P0 ≈ frac * x0
        slippage[i] = amm.slippage_pct(delta_x)

    return slippage


def compare_slippage(amms: list, trade_fracs: np.ndarray) -> dict:
    """
    Compute slippage curves for multiple AMMs.

    Returns
    -------
    dict  amm.name → np.ndarray of slippage %
    """
    return {amm.name: compute_slippage_curve(amm, trade_fracs)
            for amm in amms}


def effective_spread(amm, trade_frac: float = 0.001) -> float:
    """
    Effective bid-ask spread proxy: 2 × slippage at `trade_frac` of TVL.
    Comparable to the half-spread quoted by a limit-order-book market maker.
    """
    delta_x = trade_frac * amm.V0 / amm.P0
    return 2.0 * amm.slippage_pct(delta_x)