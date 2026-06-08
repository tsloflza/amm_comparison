#!/usr/bin/env python3
"""
run_all.py
==========
Master script that runs every experiment from the proposal and saves all figures.

Usage
-----
    # Full run (1 000 MC paths, 30-day horizon at 1-min steps) — ~5–10 min
    python run_all.py

    # Quick smoke-test (50 paths, 7-day horizon at 5-min steps) — ~20 sec
    python run_all.py --fast

    # Run only static experiments (no MC simulation) — ~5 sec
    python run_all.py --static-only

Output
------
All figures are written to  amm_comparison/results/  as PNG files.
A summary table is printed to stdout for every AMM × sigma combination.

Experiment map
--------------
E1  Slippage curves      → fig1, fig2, fig2b
E2  IL curves            → fig3, fig3b, fig3c
E3  LVR vs sigma (MC)    → fig4
E4  LVR/TVL heatmap      → fig5
E5  Hedged LP P&L        → fig6, fig7
E6  V3 range study       → fig8
E7  Curve A study        → fig9
E8  Break-even volume    → fig10
[B] Radar chart (bonus)  → fig11
"""

import sys
import argparse
import time
import numpy as np

# ── AMMs ──────────────────────────────────────────────────────────────────────
from amm import UniswapV2AMM, UniswapV3AMM, CurveStableSwapAMM, BalancerWeightedAMM

# ── Simulation ────────────────────────────────────────────────────────────────
from simulation.engine import run_simulation

# ── Plots ─────────────────────────────────────────────────────────────────────
from plots.plot_slippage import plot_bonding_curves, plot_slippage_curves, plot_slippage_all
from plots.plot_il       import plot_il_curves, plot_il_subpanels
from plots.plot_lvr      import (plot_lvr_by_sigma, plot_lvr_heatmap,
                                  plot_v3_range_study, plot_curve_A_study,
                                  plot_breakeven_volume)
from plots.plot_pnl      import plot_hedged_pnl, plot_sharpe_vs_rebal, plot_radar

# ── Metrics (for console output) ─────────────────────────────────────────────
from metrics.pnl import summary_table, print_summary_table


# =============================================================================
# Configuration
# =============================================================================
P0  = 1.0
TVL = 1_000_000

SIGMA_LIST   = [0.01, 0.03, 0.05, 0.10, 0.20]
SIGMA_MEDIUM = 0.05    # baseline for static experiments

VOLUME_SCENARIOS = {
    "low":    (50,   0.001),
    "medium": (500,  0.001),
    "high":   (5000, 0.001),
}

REBAL_FREQS = (1, 5, 60, 240, 1440)   # minutes


# =============================================================================
# AMM factory helpers
# =============================================================================
def make_amms(tvl=TVL, p0=P0):
    """Instantiate all AMMs used across experiments."""

    # Uniswap V2 (full-range)
    v2 = UniswapV2AMM(p0, tvl, fee_tier=0.003)

    # Uniswap V3 (four range widths)
    v3_full   = UniswapV3AMM(p0, tvl, 0.003, p0 * 0.01,  p0 * 100,  "full")
    v3_wide   = UniswapV3AMM(p0, tvl, 0.003, p0 * 0.50,  p0 * 2.0,  "wide")
    v3_medium = UniswapV3AMM(p0, tvl, 0.003, p0 * 0.75,  p0 * 1.33, "medium")
    v3_narrow = UniswapV3AMM(p0, tvl, 0.003, p0 * 0.90,  p0 * 1.11, "narrow")

    # Curve StableSwap (five A values)
    cu_1    = CurveStableSwapAMM(p0, tvl, 0.0004, A=1)
    cu_10   = CurveStableSwapAMM(p0, tvl, 0.0004, A=10)
    cu_100  = CurveStableSwapAMM(p0, tvl, 0.0004, A=100)
    cu_1000 = CurveStableSwapAMM(p0, tvl, 0.0004, A=1000)

    # Balancer (three weight configs)
    ba_20_80 = BalancerWeightedAMM(p0, tvl, 0.003, theta=0.20)
    ba_50_50 = BalancerWeightedAMM(p0, tvl, 0.003, theta=0.50)
    ba_80_20 = BalancerWeightedAMM(p0, tvl, 0.003, theta=0.80)

    return {
        "v2":       v2,
        "v3_full":  v3_full,
        "v3_wide":  v3_wide,
        "v3_medium":v3_medium,
        "v3_narrow":v3_narrow,
        "cu_1":     cu_1,
        "cu_10":    cu_10,
        "cu_100":   cu_100,
        "cu_1000":  cu_1000,
        "ba_20":    ba_20_80,
        "ba_50":    ba_50_50,
        "ba_80":    ba_80_20,
    }


# =============================================================================
# E1 + E2: Static experiments (no MC) ─ slippage and IL
# =============================================================================
def run_static_experiments(amms: dict):
    print("\n" + "="*60)
    print("E1 / E2  Static: Slippage & Impermanent Loss")
    print("="*60)

    # ── Fig 1: Bonding curves ────────────────────────────────────────────────
    representative = [amms["v2"], amms["v3_medium"],
                      amms["cu_100"], amms["ba_50"], amms["ba_80"]]
    plot_bonding_curves(representative)

    # ── Fig 2: Slippage — grouped panels ────────────────────────────────────
    amm_groups = {
        "V3 Ranges": [amms["v2"], amms["v3_wide"],
                      amms["v3_medium"], amms["v3_narrow"]],
        "Curve A":   [amms["cu_1"], amms["cu_10"],
                      amms["cu_100"], amms["cu_1000"]],
        "Balancer":  [amms["ba_20"], amms["ba_50"], amms["ba_80"]],
    }
    plot_slippage_curves(amm_groups)

    # ── Fig 2b: Single-panel representative comparison ───────────────────────
    plot_slippage_all([amms["v2"], amms["v3_medium"],
                       amms["cu_100"], amms["ba_50"], amms["ba_80"]])

    # ── Fig 3: IL curves ─────────────────────────────────────────────────────
    all_rep = [amms["v2"], amms["v3_medium"], amms["cu_100"],
               amms["ba_50"], amms["ba_80"]]
    plot_il_curves(all_rep, zoom=False)
    plot_il_curves([amms["cu_1"], amms["cu_10"],
                    amms["cu_100"], amms["cu_1000"]], zoom=True)

    # ── Fig 3c: IL sub-panels ────────────────────────────────────────────────
    il_groups = {
        "V3 Ranges":  [amms["v2"], amms["v3_wide"],
                       amms["v3_medium"], amms["v3_narrow"]],
        "Curve A":    [amms["cu_1"], amms["cu_10"],
                       amms["cu_100"], amms["cu_1000"]],
        "Balancer θ": [amms["ba_20"], amms["ba_50"], amms["ba_80"]],
    }
    plot_il_subpanels(il_groups)

    # ── Fig 8: V3 range study (static part) ─────────────────────────────────
    v3_list = [amms["v3_full"], amms["v3_wide"],
               amms["v3_medium"], amms["v3_narrow"]]
    plot_v3_range_study(v3_list, sigma=SIGMA_MEDIUM)

    # ── Fig 9: Curve A study ─────────────────────────────────────────────────
    curve_list = [amms["cu_1"], amms["cu_10"],
                  amms["cu_100"], amms["cu_1000"]]
    plot_curve_A_study(curve_list, sigma=0.02)

    # ── Fig 10: Break-even volume ────────────────────────────────────────────
    bev_amms = [amms["v2"], amms["v3_medium"],
                amms["cu_100"], amms["ba_50"], amms["ba_80"]]
    plot_breakeven_volume(bev_amms, sigma_list=SIGMA_LIST)

    print("  Static experiments done.\n")


# =============================================================================
# E3 + E4: LVR vs sigma (analytical, no MC required)
# =============================================================================
def run_lvr_static(amms: dict):
    print("="*60)
    print("E3 / E4  LVR vs Sigma (analytical)")
    print("="*60)

    rep_amms = [amms["v2"], amms["v3_medium"], amms["v3_narrow"],
                amms["cu_100"], amms["ba_50"], amms["ba_80"]]
    plot_lvr_by_sigma(rep_amms, SIGMA_LIST)

    # Heatmap: all AMMs × all sigma
    from collections import OrderedDict
    hmap = OrderedDict([
        ("V2 full",     amms["v2"]),
        ("V3 wide",     amms["v3_wide"]),
        ("V3 medium",   amms["v3_medium"]),
        ("V3 narrow",   amms["v3_narrow"]),
        ("Curve A=1",   amms["cu_1"]),
        ("Curve A=10",  amms["cu_10"]),
        ("Curve A=100", amms["cu_100"]),
        ("Curve A=1000",amms["cu_1000"]),
        ("Bal 20/80",   amms["ba_20"]),
        ("Bal 50/50",   amms["ba_50"]),
        ("Bal 80/20",   amms["ba_80"]),
    ])
    plot_lvr_heatmap(hmap, SIGMA_LIST)

    print("  LVR static experiments done.\n")


# =============================================================================
# E5: Monte Carlo — hedged P&L and Sharpe
# =============================================================================
def run_mc_experiments(amms: dict, fast: bool = False):
    print("="*60)
    print("E5  Monte Carlo: Hedged P&L & Sharpe Ratios")
    print("="*60)

    n_paths    = 50  if fast else 500
    T_days     = 7   if fast else 30
    dt_minutes = 5   if fast else 1

    print(f"  n_paths={n_paths}  T_days={T_days}  dt_minutes={dt_minutes}")

    # Representative AMMs for the MC run
    mc_amms = {
        "UniswapV2":         amms["v2"],
        "UniswapV3[medium]": amms["v3_medium"],
        "Curve(A=100)":      amms["cu_100"],
        "Balancer(80/20)":   amms["ba_80"],
    }

    sim_results_medium_vol = {}

    for name, amm in mc_amms.items():
        print(f"  Simulating {name} …")
        t0 = time.time()
        res = run_simulation(
            amm=amm,
            sigma_daily=SIGMA_MEDIUM,
            T_days=T_days,
            dt_minutes=dt_minutes,
            n_paths=n_paths,
            seed=42,
            lambda_per_day=500,
            mean_trade_frac=0.001,
            rebal_freqs_min=REBAL_FREQS,
        )
        print(f"    done in {time.time()-t0:.1f}s")
        sim_results_medium_vol[name] = res

        # Print summary table
        table = summary_table(res, T_days)
        print_summary_table(table, amm_name=name)

        # Fig 6: hedged P&L bar chart + stats table
        plot_hedged_pnl(res, amm_name=name, T_days=T_days)

    # Fig 7: Sharpe vs rebalancing frequency across AMMs
    plot_sharpe_vs_rebal(sim_results_medium_vol, T_days=T_days)

    print("  MC experiments done.\n")
    return sim_results_medium_vol


# =============================================================================
# Bonus: Radar chart
# =============================================================================
def run_bonus(amms: dict):
    print("="*60)
    print("[Bonus]  Radar Chart")
    print("="*60)
    radar_amms = [amms["v2"], amms["v3_medium"],
                  amms["cu_100"], amms["ba_50"], amms["ba_80"]]
    plot_radar(radar_amms, sigma=SIGMA_MEDIUM)
    print("  Radar done.\n")


# =============================================================================
# Entry point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Run all AMM comparison experiments.")
    parser.add_argument("--fast",        action="store_true",
                        help="Quick run: 50 paths, 7 days, 5-min steps.")
    parser.add_argument("--static-only", action="store_true",
                        help="Skip MC simulation; only static plots.")
    args = parser.parse_args()

    print("\n" + "█"*60)
    print("  AMM Comparison Study — run_all.py")
    print("  P0=%.1f  TVL=$%s  σ range: %s" % (
        P0, f"{TVL:,.0f}", [f"{s*100:.0f}%" for s in SIGMA_LIST]))
    print("█"*60)

    t_start = time.time()

    # Build all AMM instances once
    amms = make_amms()
    print(f"\n  Built {len(amms)} AMM instances.")

    # ── Static experiments (fast, no MC) ─────────────────────────────────────
    run_static_experiments(amms)
    run_lvr_static(amms)

    # ── Monte Carlo ──────────────────────────────────────────────────────────
    if not args.static_only:
        run_mc_experiments(amms, fast=args.fast)
    else:
        print("  [--static-only] Skipping MC simulation.\n")

    # ── Bonus ────────────────────────────────────────────────────────────────
    run_bonus(amms)

    elapsed = time.time() - t_start
    print("="*60)
    print(f"  All done in {elapsed:.1f}s")
    print("  Figures saved to:  amm_comparison/results/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()