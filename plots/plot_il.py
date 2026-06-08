"""
plots/plot_il.py
----------------
Figure 3: Impermanent Loss (LVH) curves across all AMM types.

Outputs
-------
results/fig3_il_curves.png
results/fig3b_il_zoom.png   (zoom near peg, for stablecoin comparison)
"""

import numpy as np
import matplotlib.pyplot as plt
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _color(name):
    palette = {
        "UniswapV2":         "#e15759",
        "UniswapV3[wide]":   "#f28e2b",
        "UniswapV3[medium]": "#ff9da7",
        "UniswapV3[narrow]": "#b07aa1",
        "Curve(A=1)":        "#bab0ac",
        "Curve(A=10)":       "#76b7b2",
        "Curve(A=100)":      "#4e79a7",
        "Curve(A=1000)":     "#59a14f",
        "Balancer(20/80)":   "#edc948",
        "Balancer(50/50)":   "#e15759",
        "Balancer(80/20)":   "#f1ce63",
    }
    for k, v in palette.items():
        if k in name:
            return v
    return "grey"


def _ls(name):
    if "V3" in name or "Uniswap" in name and "[" in name:
        lsmap = {"wide": "--", "medium": "-.", "narrow": ":"}
        for k, v in lsmap.items():
            if k in name:
                return v
    return "-"


# ── Figure 3: IL vs price ratio ──────────────────────────────────────────────
def plot_il_curves(amms: list, save: bool = True, zoom: bool = False):
    """
    Plot IL% as a function of price ratio r = P/P0.

    Parameters
    ----------
    amms : list of AMM instances (all normalised to same P0 and TVL)
    zoom : if True, restrict x-axis to [0.8, 1.2] for stablecoin view
    """
    if zoom:
        price_ratios = np.linspace(0.80, 1.20, 300)
        title_suffix = " (zoom near peg)"
        fname = "fig3b_il_zoom.png"
    else:
        price_ratios = np.concatenate([
            np.linspace(0.1, 0.9, 100),
            np.linspace(0.9, 1.1, 100),
            np.linspace(1.1, 5.0, 100),
        ])
        title_suffix = ""
        fname = "fig3_il_curves.png"

    fig, ax = plt.subplots(figsize=(8, 5))

    for amm in amms:
        il_vals = np.array([amm.impermanent_loss_pct(r * amm.P0) for r in price_ratios])
        ax.plot(price_ratios, il_vals,
                label=amm.name, color=_color(amm.name),
                linestyle=_ls(amm.name), linewidth=1.8)

    ax.axvline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5, label="P = P₀")
    ax.axhline(0.0, color="black", linestyle="-", linewidth=0.5)

    ax.set_xlabel("Price ratio  r = P / P₀", fontsize=11)
    ax.set_ylabel("Impermanent Loss  (% of V₀)", fontsize=11)
    ax.set_title(f"Fig 3 – Impermanent Loss vs Price Ratio{title_suffix}",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.5)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, fname)
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved {path}")
    return fig


# ── Figure 3c: IL by AMM family in sub-panels ────────────────────────────────
def plot_il_subpanels(amm_groups: dict, save: bool = True):
    """
    Sub-panel version: separate panel per AMM family, sweeping the key
    intra-family parameter (range width, A, or weight θ).
    """
    price_ratios = np.linspace(0.2, 3.0, 300)

    n = len(amm_groups)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (group, amms) in zip(axes, amm_groups.items()):
        for amm in amms:
            il_vals = np.array([amm.impermanent_loss_pct(r * amm.P0) for r in price_ratios])
            ax.plot(price_ratios, il_vals,
                    label=amm.name, color=_color(amm.name),
                    linestyle=_ls(amm.name), linewidth=1.8)
        ax.axvline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
        ax.set_xlabel("r = P / P₀", fontsize=10)
        ax.set_title(group, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, framealpha=0.9)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(bottom=-0.5)

    axes[0].set_ylabel("IL  (% of V₀)", fontsize=10)
    fig.suptitle("Fig 3c – IL by AMM Family", fontsize=12, fontweight="bold", y=1.01)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "fig3c_il_subpanels.png")
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved {path}")
    return fig