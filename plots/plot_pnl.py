"""
plots/plot_pnl.py
-----------------
Figure 6: Cumulative hedged P&L series replicating Milionis et al. Fig. 5.
Figure 7: Sharpe ratio vs rebalancing frequency.
Figure 11 (Bonus): Radar chart — multi-metric AMM comparison.

Outputs
-------
results/fig6_hedged_pnl_{amm_name}.png
results/fig7_sharpe_vs_rebal.png
results/fig11_radar.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

from metrics.pnl import pnl_summary, summary_table, print_summary_table

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

FREQ_LABELS = {1: "1min", 5: "5min", 60: "1H", 240: "4H", 1440: "1D"}
FREQ_COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1"]


# ── Figure 6: Cumulative hedged P&L (one panel per AMM) ──────────────────────
def plot_hedged_pnl(sim_result: dict, amm_name: str,
                    T_days: int = 30, save: bool = True):
    """
    Reproduce Milionis et al. Fig. 5 style for a single AMM.

    Shows:
      - pool_pnl (raw, unhedged)          — grey
      - hedged_pnl at each frequency      — coloured bars
      - fees_minus_lvr (theoretical)      — dark grey

    Uses the *mean path* across all Monte Carlo simulations for the
    cumulative chart, and a shaded ±1-std band.
    """
    prices  = sim_result["prices"]
    n_steps = prices.shape[0] - 1
    T_arr   = np.linspace(0, T_days, n_steps + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1.4]})

    # ── Top panel: pool P&L + hedged P&L ────────────────────────────────────
    pool_pnl_mean  = float(np.mean(sim_result["pool_pnl"])) * 100
    pool_pnl_std   = float(np.std(sim_result["pool_pnl"]))  * 100

    ax1.bar([0], [pool_pnl_mean], color="grey", alpha=0.5,
            label="pool_pnl (mean)", width=0.4)

    x_pos = 1
    for (freq, label), color in zip(FREQ_LABELS.items(), FREQ_COLORS):
        if freq not in sim_result["hedged_pnl"]:
            continue
        arr    = sim_result["hedged_pnl"][freq] * 100
        mean_v = float(np.mean(arr))
        std_v  = float(np.std(arr))
        ax1.bar([x_pos], [mean_v], yerr=[std_v], color=color, alpha=0.75,
                label=f"hedged_{label}", capsize=4, width=0.4, zorder=3)
        x_pos += 1

    fees_lvr_arr = (sim_result["fee_revenue"] - sim_result["analytical_lvr"]) * 100
    mean_fl = float(np.mean(fees_lvr_arr))
    std_fl  = float(np.std(fees_lvr_arr))
    ax1.bar([x_pos], [mean_fl], yerr=[std_fl], color="black", alpha=0.4,
            label="fees − LVR (theory)", capsize=4, width=0.4, zorder=3)

    ax1.axhline(0, color="black", linewidth=0.7)
    ax1.set_ylabel("Total return  (% of TVL)", fontsize=10)
    ax1.set_title(f"Fig 6 – Hedged vs Unhedged LP Returns\n{amm_name}",
                  fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8, framealpha=0.9)
    ax1.grid(True, axis="y", alpha=0.3)
    x_labels = (["pool_pnl"]
                + [FREQ_LABELS[f] for f in sorted(sim_result["hedged_pnl"])]
                + ["fees−LVR"])
    ax1.set_xticks(range(len(x_labels)))
    ax1.set_xticklabels(x_labels, fontsize=8)

    # ── Bottom panel: summary table as text ─────────────────────────────────
    table = summary_table(sim_result, T_days)
    col_labels = ["Mean APR%", "Std APR%", "Sharpe"]
    row_labels = list(table.keys())
    cell_vals  = [[f"{v['mean_annual_pct']:.2f}",
                   f"{v['std_annual_pct']:.2f}",
                   f"{v['sharpe']:.2f}"] for v in table.values()]

    ax2.axis("off")
    tbl = ax2.table(cellText=cell_vals, rowLabels=row_labels,
                    colLabels=col_labels, loc="center",
                    cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.4)
    ax2.set_title("Summary Statistics (annualised)", fontsize=9, pad=2)

    plt.tight_layout()
    safe_name = amm_name.replace("/", "_").replace(" ", "_")
    path = os.path.join(RESULTS_DIR, f"fig6_hedged_pnl_{safe_name}.png")
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved {path}")
    return fig


# ── Figure 7: Sharpe ratio vs rebalancing frequency ──────────────────────────
def plot_sharpe_vs_rebal(sim_results_by_amm: dict, T_days: int = 30,
                         save: bool = True):
    """
    Line chart: Sharpe ratio vs rebalancing frequency for each AMM.
    """
    amm_colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759",
                  "#b07aa1", "#76b7b2", "#ff9da7", "#edc948"]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    freq_list = sorted(FREQ_LABELS.keys())
    x_ticks   = range(len(freq_list))
    x_labels  = [FREQ_LABELS[f] for f in freq_list]

    for i, (amm_name, res) in enumerate(sim_results_by_amm.items()):
        sharpes = []
        for freq in freq_list:
            if freq not in res["hedged_pnl"]:
                sharpes.append(np.nan)
                continue
            stats = pnl_summary(res["hedged_pnl"][freq], T_days)
            sharpes.append(stats["sharpe"])

        ax.plot(x_ticks, sharpes, marker="o", linewidth=2, markersize=6,
                label=amm_name, color=amm_colors[i % len(amm_colors)])

    ax.set_xticks(list(x_ticks))
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_xlabel("Rebalancing Frequency", fontsize=11)
    ax.set_ylabel("Sharpe Ratio  (annualised)", fontsize=11)
    ax.set_title("Fig 7 – Sharpe Ratio vs Delta-Hedge Frequency",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "fig7_sharpe_vs_rebal.png")
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved {path}")
    return fig


# ── Figure 11 (Bonus): Radar chart ───────────────────────────────────────────
def plot_radar(amms: list, sigma: float = 0.05, tvl_frac_for_slip: float = 0.01,
               save: bool = True):
    """
    Spider/radar chart comparing AMMs across 5 dimensions:
      1. Low Slippage     (1% TVL trade)   — lower is better for traders
      2. Low IL           (±20% move)      — lower is better for LPs
      3. Low LVR/TVL      (annual)         — lower is better for LPs
      4. Capital Efficiency (marginal liq) — higher is better for LPs
      5. Fee Income       (fee_tier)       — higher = more revenue per unit volume

    Normalisation uses log-scale for LVR and Capital Efficiency because they
    span multiple orders of magnitude; linear scale collapses weaker AMMs to zero.

    Scores are clipped to [0.05, 1.0] so no AMM ever completely disappears
    on a dimension — a score near 0.05 means "worst in this set" rather than
    "literally zero", which is more informative.

    V2 and Balancer(50/50) are mathematically identical at θ=0.5, so the radar
    replaces Balancer(50/50) with Balancer(20/80) to show a more distinct profile.
    """
    from metrics.slippage import compute_slippage_curve
    from metrics.lvr import daily_lvr_analytical

    # De-duplicate: if Balancer(50/50) is in the list, swap it for 20/80
    # so we have genuinely different profiles on the chart.
    display_amms = []
    seen_ids = set()
    for amm in amms:
        key = (round(amm.fee_tier, 6),
               round(amm.marginal_liquidity(amm.P0), 2))
        if key not in seen_ids:
            seen_ids.add(key)
            display_amms.append(amm)

    categories = ["Low Slippage\n(1% TVL)",
                  "Low IL\n(±20%)",
                  "Low LVR/TVL",
                  "Capital\nEfficiency",
                  "Fee Tier\n(LP Income)"]
    N      = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]   # close the polygon

    # ── Collect raw metrics ──────────────────────────────────────────────────
    raw = {}
    for amm in display_amms:
        slip = compute_slippage_curve(amm, np.array([tvl_frac_for_slip]))[0]
        il20 = amm.impermanent_loss_pct(amm.P0 * 1.20)
        lvr  = daily_lvr_analytical(amm, sigma) * 100 * 365          # annual %
        ml   = amm.marginal_liquidity(amm.P0) * amm.P0               # dimensionless
        fee  = amm.fee_tier * 100
        raw[amm.name] = [slip, il20, lvr, ml, fee]

    raw_arr = np.array(list(raw.values()))   # (n_amms, 5)

    # ── Normalise to [0,1] ───────────────────────────────────────────────────
    # LVR (col 2) and Capital Efficiency (col 3) span orders of magnitude
    # → use log10 normalisation to avoid collapsing weaker AMMs to zero.
    # All other cols → linear normalisation.
    norm = raw_arr.copy().astype(float)

    log_cols    = [2, 3]   # LVR, Cap Eff
    linear_cols = [0, 1, 4]   # slip, IL, fee

    for col in log_cols:
        log_v = np.log10(np.maximum(norm[:, col], 1e-9))
        lo, hi = log_v.min(), log_v.max()
        norm[:, col] = (log_v - lo) / (hi - lo) if hi > lo else 0.5

    for col in linear_cols:
        lo, hi = norm[:, col].min(), norm[:, col].max()
        norm[:, col] = (norm[:, col] - lo) / (hi - lo) if hi > lo else 0.5

    # Invert "lower-is-better" metrics: slip, IL, LVR → score = 1 - norm
    # Capital Efficiency and Fee Income stay as-is (higher = better)
    for col in [0, 1, 2]:
        norm[:, col] = 1.0 - norm[:, col]

    # Clip to [0.05, 1.0]: avoids completely invisible polygon vertices while
    # still communicating "this AMM is worst in this dimension".
    norm = np.clip(norm, 0.05, 1.0)

    # ── Draw ─────────────────────────────────────────────────────────────────
    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759",
              "#b07aa1", "#76b7b2", "#ff9da7"]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for i, (amm, values) in enumerate(zip(display_amms, norm)):
        vals = values.tolist() + values[:1].tolist()
        ax.plot(angles, vals, "o-", linewidth=1.8,
                color=colors[i % len(colors)], label=amm.name)
        ax.fill(angles, vals, alpha=0.10, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7)
    ax.set_title(
        "Fig 11 – AMM Comparison Radar\n(higher = better on each axis)\n"
        "LVR & Cap. Efficiency normalised on log scale",
        fontsize=10, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "fig11_radar.png")
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved {path}")
    return fig