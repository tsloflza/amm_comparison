"""
plots/plot_slippage.py
----------------------
Figure 2: Slippage (price impact) curves across all AMM types.
Figure 1 (bonus): Bonding curves for visual AMM intuition.

Outputs
-------
results/fig1_bonding_curves.png
results/fig2_slippage_curves.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

from metrics.slippage import compute_slippage_curve, effective_spread

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── colour / style palette  ──────────────────────────────────────────────────
COLORS = {
    "UniswapV2":         "#e15759",
    "UniswapV3[wide]":   "#f28e2b",
    "UniswapV3[medium]": "#ff9da7",
    "UniswapV3[narrow]": "#b07aa1",
    "Curve(A=1)":        "#bab0ac",
    "Curve(A=10)":       "#76b7b2",
    "Curve(A=100)":      "#4e79a7",
    "Curve(A=1000)":     "#59a14f",
    "Balancer(20/80)":   "#edc948",
    "Balancer(50/50)":   "#e15759",   # same as V2 (they overlap at 50/50)
    "Balancer(80/20)":   "#f1ce63",
}

LINESTYLES = {
    "UniswapV2":         "-",
    "UniswapV3[wide]":   "--",
    "UniswapV3[medium]": "-.",
    "UniswapV3[narrow]": ":",
}


def _color(name):
    for k, v in COLORS.items():
        if k in name:
            return v
    return "grey"


def _ls(name):
    for k, v in LINESTYLES.items():
        if k in name:
            return v
    return "-"


# ── Figure 1: Bonding Curves ─────────────────────────────────────────────────
def plot_bonding_curves(amms: list, save: bool = True):
    """
    Plot reserve space (x, y) for each AMM around P0.
    Provides intuition for how each invariant behaves.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    P0 = amms[0].P0
    x0 = amms[0].x_star(P0)

    # price range 0.5×P0 … 2×P0
    prices = np.linspace(P0 * 0.3, P0 * 2.5, 300)

    for amm in amms:
        xs = np.array([amm.x_star(P) for P in prices])
        ys = np.array([amm.y_star(P) for P in prices])
        # normalise by initial reserves for a universal chart
        ax.plot(xs / amm.x_star(P0), ys / amm.y_star(P0),
                label=amm.name, color=_color(amm.name), linewidth=1.8)

    ax.set_xlabel("Risky asset reserves  (normalised)", fontsize=11)
    ax.set_ylabel("Numeraire reserves  (normalised)", fontsize=11)
    ax.set_title("Fig 1 – Bonding Curves", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.set_xlim(0, 3.5)
    ax.set_ylim(0, 3.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "fig1_bonding_curves.png")
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved {path}")
    return fig


# ── Figure 2: Slippage Curves ────────────────────────────────────────────────
def plot_slippage_curves(amm_groups: dict, save: bool = True):
    """
    Parameters
    ----------
    amm_groups : dict of group_label → list of AMMs
                 e.g. {"V3 ranges": [v3_wide, v3_med, v3_narrow],
                        "Curve A":  [cu_1, cu_10, cu_100, cu_1000],
                        "Balancer": [ba_20, ba_50, ba_80]}
                 Plus a "baseline" key containing [v2].
    """
    trade_fracs = np.logspace(-4, -0.7, 80)   # 0.01% … 20% of TVL

    n_groups = len(amm_groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 4.5), sharey=True)
    if n_groups == 1:
        axes = [axes]

    for ax, (group_label, amms) in zip(axes, amm_groups.items()):
        for amm in amms:
            slip = compute_slippage_curve(amm, trade_fracs)
            ax.loglog(trade_fracs * 100, slip,
                      label=amm.name, color=_color(amm.name),
                      linestyle=_ls(amm.name), linewidth=1.8)

        ax.set_xlabel("Trade size  (% of TVL)", fontsize=10)
        ax.set_title(group_label, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, framealpha=0.9)
        ax.grid(True, which="both", alpha=0.25)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2g%%"))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2g%%"))

    axes[0].set_ylabel("Price impact  (%)", fontsize=10)
    fig.suptitle("Fig 2 – Slippage vs Trade Size", fontsize=13, fontweight="bold", y=1.01)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "fig2_slippage_curves.png")
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved {path}")
    return fig


# ── Figure 2b: Single panel comparing all representative AMMs ───────────────
def plot_slippage_all(amms: list, save: bool = True):
    """
    Single panel: one representative per AMM family.
    Clear for presentations.
    """
    trade_fracs = np.logspace(-4, -0.7, 80)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for amm in amms:
        slip = compute_slippage_curve(amm, trade_fracs)
        ax.loglog(trade_fracs * 100, slip,
                  label=amm.name, color=_color(amm.name),
                  linestyle=_ls(amm.name), linewidth=2.0)

    ax.set_xlabel("Trade size  (% of TVL)", fontsize=11)
    ax.set_ylabel("Price impact  (%)", fontsize=11)
    ax.set_title("Fig 2b – Slippage Comparison (representative AMMs)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.25)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "fig2b_slippage_all.png")
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved {path}")
    return fig