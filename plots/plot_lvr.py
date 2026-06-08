"""
plots/plot_lvr.py
-----------------
Figure 4: Mean daily LVR by AMM and volatility regime (bar chart + error bars).
Figure 5: LVR/TVL heatmap as function of (σ, AMM parameter).
Figure 8: V3 LVR/TVL and capital efficiency vs range width.
Figure 9: Curve — slippage, IL, LVR vs amplification A.
Figure 10: Break-even volume curve.

Outputs
-------
results/fig4_lvr_by_amm_sigma.png
results/fig5_lvr_heatmap.png
results/fig8_v3_range_study.png
results/fig9_curve_A_study.png
results/fig10_breakeven_volume.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

from metrics.lvr import (daily_lvr_analytical, lvr_vs_sigma,
                          break_even_volume, analytical_lvr_path)
from metrics.pnl import collect_lvr_by_sigma

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SIGMA_LABELS = {0.01: "1%", 0.03: "3%", 0.05: "5%", 0.10: "10%", 0.20: "20%"}

AMM_COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759",
              "#b07aa1", "#76b7b2", "#ff9da7", "#edc948"]


# ── Figure 4: Bar chart LVR vs sigma ─────────────────────────────────────────
def plot_lvr_by_sigma(
    amms: list,
    sigma_list: list,
    sim_results: dict = None,    # optional: nested dict amm.name→sigma→run result
    T_days: int = 30,
    save: bool = True,
):
    """
    Grouped bar chart: mean daily LVR (% of TVL) per AMM for each σ regime.
    If sim_results provided, also draw std error bars from MC paths.
    """
    n_amms   = len(amms)
    n_sigmas = len(sigma_list)

    # Build data matrix
    means = np.zeros((n_amms, n_sigmas))
    stds  = np.zeros((n_amms, n_sigmas))

    for i, amm in enumerate(amms):
        for j, sigma in enumerate(sigma_list):
            if sim_results and amm.name in sim_results and sigma in sim_results[amm.name]:
                # Use MC distribution
                lvr_paths = list(sim_results[amm.name][sigma].values())[0]["analytical_lvr"]
                scale = 365.0 / T_days * 100.0
                means[i, j] = np.mean(lvr_paths) * scale
                stds[i, j]  = np.std(lvr_paths)  * scale
            else:
                # Use analytical formula (constant σ, evaluated at P0)
                means[i, j] = daily_lvr_analytical(amm, sigma) * 365 * 100  # annualised %

    fig, ax = plt.subplots(figsize=(10, 5))
    x     = np.arange(n_sigmas)
    width = 0.8 / n_amms

    for i, amm in enumerate(amms):
        offset = (i - n_amms / 2 + 0.5) * width
        bars = ax.bar(x + offset, means[i], width * 0.9,
                      label=amm.name, color=AMM_COLORS[i % len(AMM_COLORS)],
                      zorder=3)
        if stds[i].any():
            ax.errorbar(x + offset, means[i], yerr=stds[i],
                        fmt="none", color="black", capsize=3, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels([SIGMA_LABELS.get(s, f"{s*100:.0f}%") for s in sigma_list])
    ax.set_xlabel("Daily volatility σ", fontsize=11)
    ax.set_ylabel("Annual LVR  (% of TVL)", fontsize=11)
    ax.set_title("Fig 4 – Annual LVR by AMM and Volatility Regime", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "fig4_lvr_by_amm_sigma.png")
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved {path}")
    return fig


# ── Figure 5: Heatmap LVR/TVL (σ × AMM parameter) ───────────────────────────
def plot_lvr_heatmap(amms_param: dict, sigma_list: list, save: bool = True):
    """
    Heatmap of annual LVR/TVL as a function of (σ, AMM parameter).

    Parameters
    ----------
    amms_param : OrderedDict  param_label → AMM instance
                 e.g. {"V2 full": v2, "V3 wide": v3w, "V3 med": v3m, ...}
    sigma_list : list of daily σ values
    """
    labels  = list(amms_param.keys())
    amms    = list(amms_param.values())
    n_amms  = len(amms)
    n_sigma = len(sigma_list)

    Z = np.zeros((n_amms, n_sigma))
    for i, amm in enumerate(amms):
        for j, sigma in enumerate(sigma_list):
            Z[i, j] = daily_lvr_analytical(amm, sigma) * 365 * 100  # annual %

    fig, ax = plt.subplots(figsize=(max(6, n_sigma * 1.2), max(4, n_amms * 0.7)))
    im = ax.imshow(Z, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Annual LVR  (% of TVL)", fontsize=9)

    ax.set_xticks(range(n_sigma))
    ax.set_xticklabels([SIGMA_LABELS.get(s, f"{s*100:.0f}%") for s in sigma_list], fontsize=9)
    ax.set_yticks(range(n_amms))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Daily σ", fontsize=10)
    ax.set_title("Fig 5 – Annual LVR/TVL Heatmap", fontsize=11, fontweight="bold")

    # Annotate cells
    for i in range(n_amms):
        for j in range(n_sigma):
            ax.text(j, i, f"{Z[i,j]:.1f}", ha="center", va="center",
                    fontsize=7, color="black" if Z[i,j] < Z.max() * 0.6 else "white")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "fig5_lvr_heatmap.png")
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved {path}")
    return fig


# ── Figure 8: V3 range study ──────────────────────────────────────────────────
def plot_v3_range_study(v3_amms: list, sigma: float = 0.05, save: bool = True):
    """
    Two-panel plot: left = LVR/TVL/day vs range width, right = range utilization
    (requires a price array) or capital efficiency proxy.

    Parameters
    ----------
    v3_amms : list of UniswapV3AMM with increasing range widths
    sigma   : volatility for LVR calculation
    """
    widths  = [amm.Pb / amm.Pa for amm in v3_amms]
    lvr_day = [daily_lvr_analytical(amm, sigma) * 100 for amm in v3_amms]

    # Capital efficiency vs a full-range V2 baseline
    from amm.v2_amm import UniswapV2AMM
    v2_ref  = UniswapV2AMM(v3_amms[0].P0, v3_amms[0].V0, v3_amms[0].fee_tier)
    from metrics.pnl import capital_efficiency as cap_eff
    cap_effs = [cap_eff(amm, v2_ref) for amm in v3_amms]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(widths, lvr_day, "o-", color="#4e79a7", linewidth=2, markersize=6)
    for amm, x, y in zip(v3_amms, widths, lvr_day):
        ax1.annotate(amm.range_label, (x, y), textcoords="offset points",
                     xytext=(4, 4), fontsize=7)
    ax1.set_xlabel("Range width  r = Pₕ / Pₗ", fontsize=10)
    ax1.set_ylabel("Daily LVR / TVL  (%)", fontsize=10)
    ax1.set_title("V3 LVR/TVL vs Range Width", fontsize=10, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2.plot(widths, cap_effs, "s-", color="#f28e2b", linewidth=2, markersize=6)
    ax2.set_xlabel("Range width  r = Pₕ / Pₗ", fontsize=10)
    ax2.set_ylabel("Capital Efficiency vs V2", fontsize=10)
    ax2.set_title("V3 Capital Efficiency vs Range Width", fontsize=10, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(1.0, color="grey", linestyle="--", label="V2 baseline")
    ax2.legend(fontsize=8)

    fig.suptitle("Fig 8 – Uniswap V3 Range Analysis  (σ=%d%%)" % (sigma * 100),
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "fig8_v3_range_study.png")
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved {path}")
    return fig


# ── Figure 9: Curve A study ───────────────────────────────────────────────────
def plot_curve_A_study(curve_amms: list, sigma: float = 0.02, save: bool = True):
    """
    Three-panel: slippage at 1% TVL, daily LVR/TVL, IL at r=1.02
    as functions of amplification coefficient A.
    """
    import numpy as np
    from metrics.slippage import compute_slippage_curve
    from metrics.lvr import daily_lvr_analytical

    A_vals      = [amm.A for amm in curve_amms]
    slip_1pct   = [compute_slippage_curve(amm, np.array([0.01]))[0] for amm in curve_amms]
    lvr_day_pct = [daily_lvr_analytical(amm, sigma) * 100 for amm in curve_amms]
    il_102      = [amm.impermanent_loss_pct(amm.P0 * 1.02) for amm in curve_amms]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    kw = dict(marker="o", linewidth=2, markersize=6, color="#4e79a7")

    axes[0].semilogx(A_vals, slip_1pct, **kw)
    axes[0].set_xlabel("Amplification A", fontsize=10)
    axes[0].set_ylabel("Slippage at 1% TVL  (%)", fontsize=10)
    axes[0].set_title("Slippage", fontsize=10, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogx(A_vals, lvr_day_pct, **kw)
    axes[1].set_xlabel("Amplification A", fontsize=10)
    axes[1].set_ylabel("Daily LVR / TVL  (%)", fontsize=10)
    axes[1].set_title("LVR / TVL", fontsize=10, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogx(A_vals, il_102, **kw)
    axes[2].set_xlabel("Amplification A", fontsize=10)
    axes[2].set_ylabel("IL at r = 1.02  (%)", fontsize=10)
    axes[2].set_title("Impermanent Loss (r=1.02)", fontsize=10, fontweight="bold")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Fig 9 – Curve StableSwap: Effect of Amplification A  (σ=%d%%)" % (sigma * 100),
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "fig9_curve_A_study.png")
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved {path}")
    return fig


# ── Figure 10: Break-even volume curve ───────────────────────────────────────
def plot_breakeven_volume(amms: list, sigma_list: list, save: bool = True):
    """
    For each AMM, plot the break-even daily volume (as % of TVL) vs sigma.
    Above the line: LPs are profitable. Below: LPs lose money.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    sigmas_pct = [s * 100 for s in sigma_list]

    for i, amm in enumerate(amms):
        bev = [break_even_volume(amm, s) * 100 for s in sigma_list]
        ax.plot(sigmas_pct, bev, label=amm.name,
                color=AMM_COLORS[i % len(AMM_COLORS)], linewidth=2.0, marker="o", markersize=4)

    ax.set_xlabel("Daily volatility σ  (%)", fontsize=11)
    ax.set_ylabel("Break-even daily volume  (% of TVL)", fontsize=11)
    ax.set_title("Fig 10 – Break-even Trading Volume", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "fig10_breakeven_volume.png")
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved {path}")
    return fig