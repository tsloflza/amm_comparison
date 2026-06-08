"""
metrics/pnl.py
--------------
LP P&L decomposition and performance statistics.

Following Milionis et al. eq. (13):

    LP P&L = Market Risk  +  Fees  −  LVR
           = Rebalancing P&L  +  (Fees − LVR)
           = Rebalancing P&L  +  Hedged P&L

This module:
  - Computes summary statistics (mean, std, Sharpe) for each P&L series
  - Produces the comparison table matching Milionis et al. Table 1
  - Computes annualised metrics from simulation results

Public API
----------
    pnl_summary(pnl_array, T_days)       ->  dict
    summary_table(sim_result, T_days)    ->  dict of label → stats
    capital_efficiency(amm_v3, amm_v2)   ->  float
"""

import numpy as np


# -----------------------------------------------------------------------
# Single P&L array statistics
# -----------------------------------------------------------------------
def pnl_summary(pnl_array: np.ndarray, T_days: int) -> dict:
    """
    Compute annualised return, std, and Sharpe for a 1-D array of
    total P&L returns (each entry is the total fractional return for one
    simulated path over T_days).

    Parameters
    ----------
    pnl_array : (n_paths,) fractional returns  (e.g. 0.05 = +5% of TVL)
    T_days    : simulation horizon used to annualise

    Returns
    -------
    dict with keys: mean_annual, std_annual, sharpe, median_annual
    """
    scale = 365.0 / T_days

    mean_ann   = float(np.mean(pnl_array)) * scale * 100.0   # %
    std_ann    = float(np.std(pnl_array))  * scale * 100.0   # %
    median_ann = float(np.median(pnl_array)) * scale * 100.0

    sharpe = mean_ann / std_ann if std_ann > 0 else np.nan

    return {
        "mean_annual_pct":   mean_ann,
        "std_annual_pct":    std_ann,
        "median_annual_pct": median_ann,
        "sharpe":            sharpe,
    }


# -----------------------------------------------------------------------
# Full summary table from run_simulation output
# -----------------------------------------------------------------------
def summary_table(sim_result: dict, T_days: int = 30) -> dict:
    """
    Build a summary statistics table from a run_simulation() result dict.

    Mirrors the structure of Milionis et al. Table 1:
        pool_pnl | hedged_1min | hedged_5min | hedged_1H | hedged_4H | hedged_1D | fees_minus_lvr

    Returns
    -------
    dict  label → pnl_summary dict
    """
    table = {}

    # Raw pool P&L
    table["pool_pnl"] = pnl_summary(sim_result["pool_pnl"], T_days)

    # Hedged P&L at each rebalancing frequency
    freq_labels = {1: "hedged_1min", 5: "hedged_5min",
                   60: "hedged_1H", 240: "hedged_4H", 1440: "hedged_1D"}
    for freq, label in freq_labels.items():
        if freq in sim_result["hedged_pnl"]:
            table[label] = pnl_summary(sim_result["hedged_pnl"][freq], T_days)

    # Fees minus LVR (analytical prediction)
    fees_minus_lvr = sim_result["fee_revenue"] - sim_result["analytical_lvr"]
    table["fees_minus_lvr"] = pnl_summary(fees_minus_lvr, T_days)

    return table


def print_summary_table(table: dict, amm_name: str = ""):
    """Pretty-print the summary table to stdout."""
    header = f"{'Label':<20} {'Mean APR%':>10} {'Std APR%':>10} {'Sharpe':>8}"
    print(f"\n{'═'*50}")
    print(f"  {amm_name}")
    print('═'*50)
    print(header)
    print('─'*50)
    for label, stats in table.items():
        print(f"  {label:<18} {stats['mean_annual_pct']:>10.2f} "
              f"{stats['std_annual_pct']:>10.2f} {stats['sharpe']:>8.2f}")
    print('═'*50)


# -----------------------------------------------------------------------
# Capital efficiency (V3 vs V2)
# -----------------------------------------------------------------------
def capital_efficiency(amm_v3, amm_v2) -> float:
    """
    Fee income per dollar of TVL for V3 relative to V2,
    assuming the same total trading volume through both pools.

    For a concentrated position [Pa, Pb] the efficiency gain is ≈ √(Pb/Pa).
    This function computes it numerically from the demand-curve slopes.

    Returns
    -------
    ratio : float — e.g. 4.0 means 4× more fee income per dollar
    """
    from math import sqrt
    # At P0, marginal liquidity ratio = x*'_v3 / x*'_v2
    ml_v3 = amm_v3.marginal_liquidity(amm_v3.P0) * amm_v3.V0
    ml_v2 = amm_v2.marginal_liquidity(amm_v2.P0) * amm_v2.V0
    if ml_v2 == 0:
        return np.nan
    return ml_v3 / ml_v2


def range_utilization(amm_v3, prices: np.ndarray) -> float:
    """
    Fraction of simulation steps where the V3 position is in-range.

    Parameters
    ----------
    prices : (n_steps+1, n_paths) or 1-D array
    """
    p_flat = prices.ravel()
    return float(np.mean((p_flat >= amm_v3.Pa) & (p_flat < amm_v3.Pb)))


# -----------------------------------------------------------------------
# Aggregate across AMMs / scenarios
# -----------------------------------------------------------------------
def collect_lvr_by_sigma(sim_results_by_sigma: dict, T_days: int = 30) -> dict:
    """
    Extract mean±std of analytical LVR across sigma values.

    Parameters
    ----------
    sim_results_by_sigma : dict  sigma → run_simulation output

    Returns
    -------
    dict with 'sigma', 'mean_lvr_annual', 'std_lvr_annual'
    """
    sigmas, means, stds = [], [], []
    for sigma, res in sorted(sim_results_by_sigma.items()):
        lvr = res["analytical_lvr"]
        scale = 365.0 / T_days * 100.0
        sigmas.append(sigma)
        means.append(float(np.mean(lvr)) * scale)
        stds.append(float(np.std(lvr))  * scale)
    return {"sigma": sigmas, "mean_lvr_annual": means, "std_lvr_annual": stds}