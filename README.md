# Research Proposal: Comparative Analysis of AMM Mechanisms
## Slippage, Impermanent Loss, and LP Returns across Uniswap V2/V3, Curve, and Balancer

---

## 1. Background and Motivation

Automated Market Makers (AMMs) have become the dominant infrastructure for decentralized exchange on blockchains. However, different AMM designs encode fundamentally different tradeoffs between capital efficiency, price slippage, and liquidity provider (LP) profitability. Milionis et al. (2024) establish a unified analytical framework — the **Loss-Versus-Rebalancing (LVR)** framework — showing that LP losses from price slippage equal:

$$\ell(\sigma, P) = \frac{\sigma^2 P^2}{2} \left| x^{*\prime}(P) \right|$$

where $\sigma$ is asset price volatility, $P$ is the current price, and $|x^{*\prime}(P)|$ is the **marginal liquidity** (the slope of the AMM's demand curve). This formula is elegant in that it applies to *any* locally-smooth AMM — CFMM, concentrated liquidity, or otherwise — and isolates the sole driver of LP adverse-selection losses: how aggressively the pool trades in response to price movements.

This project directly operationalizes the LVR framework by comparing three major AMM families across controlled simulations, measuring slippage, impermanent loss, and hedged/unhedged LP returns side by side.

---

## 2. Research Questions

1. How do slippage curves differ across Uniswap V2, Uniswap V3 (concentrated range), Curve StableSwap, and Balancer weighted pools as a function of trade size?
2. How do instantaneous and cumulative LVR compare across AMM types under equivalent liquidity depth, and how does this depend on asset volatility?
3. How does the fee-minus-LVR tradeoff (i.e., net hedged LP P&L) behave across AMM types under varying trading volume regimes?
4. **[Bonus]** How do Curve V2 (two-asset crypto pool with internal oracle) and Uniswap V4 hook-based dynamic fees affect these metrics?

---

## 3. AMMs Under Study

### 3.1 Core AMMs (Required)

| AMM | Invariant / Mechanism | Key Parameter(s) |
|-----|----------------------|-----------------|
| **Uniswap V2** | $x \cdot y = k$ (CPMM) | Fee tier $f = 0.3\%$ |
| **Uniswap V3** | Concentrated CPMM over range $[P_a, P_b)$ | Range width $r = P_b / P_a$; fee tier $f \in \lbrace 0.05\%, 0.3\%, 1\% \rbrace$ |
| **Curve StableSwap** | Hybrid: $A \cdot n^n \sum x_i + D = A \cdot n^n \cdot D + D^{n+1} / (n^n \prod x_i)$ | Amplification coefficient $A$ |
| **Balancer Weighted** | $\prod x_i^{w_i} = k$ (Geometric mean) | Weight vector $\mathbf{w}$; fee tier |

### 3.2 Bonus AMMs

| AMM | Key Feature |
|-----|------------|
| **Curve V2 (Crypto Pool)** | Internal EMA price oracle; dynamic amplification that re-centers the invariant around spot price |
| **DODO (PMM)** | Proactive Market Maker with external oracle integration; explicit separation of base/quote inventory |

---

## 4. Mathematical Models

### 4.1 Uniswap V2 (Full-Range CPMM)

Invariant: $\sqrt{xy} = L$. The pool value function and LVR follow directly from Milionis et al. Example 3:

$$V(P) = 2L\sqrt{P}, \quad \ell(\sigma, P) = \frac{L\sigma^2}{4\sqrt{P}}, \quad \frac{\ell(\sigma, P)}{V(P)} = \frac{\sigma^2}{8}$$

Marginal liquidity: $|x^{*\prime}(P)| = \frac{L}{2P^{3/2}}$

Slippage for a trade $\Delta x$: $\Delta y = L\sqrt{P} - \frac{L^2}{L/\sqrt{P} + \Delta x}$, giving effective price $P_{\text{eff}} = \Delta y / \Delta x$.

### 4.2 Uniswap V3 (Concentrated Liquidity)

For a single range order $[P_a, P_b)$ with liquidity $L$ (from Milionis et al. Example 4):

$$x^{*}(P) = L\left(\frac{1}{\sqrt{P}} - \frac{1}{\sqrt{P_b}}\right), \quad y^{*}(P) = L\left(\sqrt{P} - \sqrt{P_a}\right)$$

$$V(P) = L\left(2\sqrt{P} - \frac{P}{\sqrt{P_b}} - \sqrt{P_a}\right), \quad \ell(\sigma, P) = \frac{L\sigma^2}{4\sqrt{P}}$$

Key insight: instantaneous LVR equals V2's, but $V(P)$ is smaller — so **LVR per dollar of TVL diverges** as the range narrows. For aggregate V3 pools (Example 5), only in-range orders contribute:

$$\ell(\sigma, P) = \frac{\sigma^2}{4\sqrt{P}} \cdot \bar{L}(P)$$

where $\bar{L}(P)$ is total in-range liquidity at price $P$.

### 4.3 Curve StableSwap

The Curve invariant for $n = 2$ assets balances a constant-sum and constant-product component:

$$A \cdot 4(x + y) + D = A \cdot 4D + \frac{D^3}{4xy}$$

where $D$ is total invariant (≈ total value at peg) and $A$ is the amplification coefficient. As $A \to 0$ this approaches CPMM; as $A \to \infty$ it approaches the constant-sum market maker ($x + y = D$). The marginal liquidity and slippage are computed numerically from the implicit price function $P(x, D, A)$ derived from the first-order condition $\partial_x / \partial_y = -P$.

Pool value function: $V(P) = $ (numerically solved from invariant) — no closed form, but concavity is guaranteed. LVR is computed numerically via $\ell = \frac{\sigma^2 P^2}{2}|x^{*\prime}(P)|$ where $x^{*\prime}(P)$ is obtained by differentiating the invariant implicitly.

### 4.4 Balancer Weighted Pools

From Milionis et al. Example 2, the weighted geometric mean invariant $f(x,y) = x^\theta y^{1-\theta}$ gives:

$$V(P) = L \cdot \frac{\theta^\theta (1-\theta)^{1-\theta}}{1} \cdot P^\theta, \quad \frac{\ell(\sigma, P)}{V(P)} = \frac{\sigma^2}{2}\theta(1-\theta)$$

This is the only CFMM family where **LVR as a fraction of pool value is constant** across all price levels. We will study weights $\theta \in \lbrace0.2, 0.5, 0.8\rbrace$ (corresponding to 20/80, 50/50, and 80/20 pools). The 50/50 pool matches V2; asymmetric pools reduce LVR by concentrating exposure.

---

## 5. Simulation Design

### 5.1 Price Path Generation

All simulations use **Geometric Brownian Motion (GBM)** as in Milionis et al.:

$$dP_t = \sigma P_t \, dB_t$$

with a zero risk-free rate (risk-neutral measure $Q$). We simulate using the Euler-Maruyama discretization:

$$P_{t+\Delta t} = P_t \cdot \exp\left(\left(-\frac{\sigma^2}{2}\right)\Delta t + \sigma\sqrt{\Delta t}\, Z_t\right), \quad Z_t \sim \mathcal{N}(0,1)$$

**Simulation parameters:**

| Parameter | Values |
|-----------|--------|
| Initial price $P_0$ | 1.0 (normalized) |
| Time horizon $T$ | 30 days |
| Time step $\Delta t$ | 1 minute (43,200 steps) |
| Monte Carlo paths | 1,000 per scenario |
| Volatility $\sigma$ (daily) | 1%, 3%, 5%, 10%, 20% |
| Seed | Fixed for reproducibility |

**Volatility regime mapping:**

| Regime | $\sigma$ (daily) | Representative Asset |
|--------|-----------------|---------------------|
| Very Low | 1% | Stablecoin-stablecoin (e.g., USDC/DAI) |
| Low | 3% | Large-cap / stablecoin (e.g., ETH/USDC stable range) |
| Medium | 5% | ETH/USDC (baseline, matching Milionis et al. empirical range) |
| High | 10% | Mid-cap altcoin |
| Very High | 20% | Small-cap / memecoin |

### 5.2 Liquidity Scenarios

All AMMs are normalized to **$1,000,000 USD Total Value Locked (TVL)** at $P_0 = 1.0$ so that cross-AMM comparisons are meaningful. Liquidity parameters are chosen accordingly:

| AMM | Liquidity Normalization |
|-----|------------------------|
| Uniswap V2 | $L = \sqrt{x \cdot y}$ s.t. $V(1.0) = \$1\text{M}$ → $L = 500{,}000$ |
| Uniswap V3 | $L$ chosen per range s.t. $V(P_0) = \$1\text{M}$ |
| Curve | $D = \$1\text{M}$; $x = y = \$500\text{K}$ at peg |
| Balancer | Total initial value $= \$1\text{M}$; split per weight $\theta$ |

**Liquidity depth scenarios (cross-sectional, no price path):**

| Scenario | TVL | Purpose |
|----------|-----|---------|
| Thin | $100K | Stressed pool |
| Base | $1M | Standard comparison |
| Deep | $10M | Institutional-depth pool |

### 5.3 Uniswap V3 Range Scenarios

We vary the concentration range to study the LVR-per-TVL amplification effect:

| Range Label | $[P_a, P_b]$ | Width $r = P_b/P_a$ | Notes |
|-------------|-------------|---------------------|-------|
| Full range | $[0, \infty)$ | $\infty$ | Equivalent to V2 |
| Wide | $[0.5, 2.0]$ | 4× | Covers most ETH/USDC price action |
| Medium | $[0.75, 1.33]$ | 1.78× | Moderate concentration |
| Narrow | $[0.9, 1.11]$ | 1.23× | High concentration, range-order-like |

### 5.4 Trading Volume Scenarios

We simulate noise trader volume as a Poisson process. At each minute, a trade of size $\Delta x$ arrives with probability $\lambda \Delta t$, where $\Delta x \sim \text{Lognormal}(\mu_{\text{trade}}, \sigma_{\text{trade}})$.

| Scenario | $\lambda$ (trades/day) | Mean trade size | Fee revenue rate (approx.) |
|----------|----------------------|-----------------|---------------------------|
| Low volume | 50 | 0.1% of TVL | ~1 bp/day |
| Medium volume | 500 | 0.1% of TVL | ~10 bp/day |
| High volume | 5,000 | 0.1% of TVL | ~100 bp/day |

For slippage analysis (static, no price path), we sweep trade sizes from 0.01% to 20% of pool TVL.

### 5.5 Curve Amplification Scenarios

| $A$ value | Character | Representative Use Case |
|-----------|-----------|------------------------|
| 1 | Near-CPMM | Exotic / volatile stablecoins |
| 10 | Transitional | Conservative stablecoin pool |
| 100 | Standard Curve | USDC/DAI/USDT (typical) |
| 1,000 | Hyper-stable | fTokens, tightly pegged pairs |
| 10,000 | Near constant-sum | Perfect 1:1 pegs |

---

## 6. Metrics

### 6.1 Slippage

**Price Impact** for a trade of size $\Delta x$:

$$\text{Slippage}(\Delta x) = \frac{P_{\text{eff}}(\Delta x) - P_0}{P_0} \times 100\%$$

where $P_{\text{eff}} = \Delta y / \Delta x$ is the average execution price received.

We plot slippage curves as a function of $\Delta x / \text{TVL}$ (trade size relative to pool size) for all AMMs under fixed TVL.

### 6.2 Impermanent Loss (Loss-Versus-Holding, LVH)

$$\text{LVH}_t = R^{\text{HODL}}_t - V_t = (P_t x^{*}(P_0) + y^{*}(P_0)) - V(P_t)$$

Expressed as a percentage of initial position value $V_0$:

$$\text{IL\%}(P_t) = \frac{\text{LVH}_t}{V_0} \times 100\%$$

For Uniswap V2: $\text{IL\%}(r) = \frac{2\sqrt{r}}{1+r} - 1$ where $r = P_t / P_0$. We will derive and plot analogous closed-form or numerical expressions for all AMM types.

### 6.3 Loss-Versus-Rebalancing (LVR)

The core metric from Milionis et al.:

$$\text{LVR}_T = \int_0^T \ell(\sigma_t, P_t)\, dt$$

**Analytical (per-period) LVR:**

| AMM | Instantaneous LVR formula |
|-----|--------------------------|
| V2 (full range) | $\ell = \frac{\sigma^2}{8} V(P_t)$ |
| V3 (range $[P_a, P_b]$) | $\ell = \frac{\sigma^2}{4\sqrt{P}} \bar{L}(P)$ (zero outside range) |
| Balancer ($\theta$) | $\ell = \frac{\sigma^2}{2}\theta(1-\theta) V(P_t)$ |
| Curve | $\ell = \frac{\sigma^2 P^2}{2}|x^{*\prime}(P)|$ (numerical) |

**Empirical (delta-hedging) LVR** from Milionis et al. Section 6:

$$\text{LVR}_T^{\text{empirical}} = \sum_t x^*_t (P_{t+1} - P_t) - (V_T - V_0)$$

which equals the difference between the rebalancing strategy's P&L and the pool's P&L. This validates the analytical formula under finite-step simulation.

### 6.4 LP P&L Decomposition

Following Milionis et al. equation (13):

$$\underbrace{\int_0^T x^*(P_t)\, dP_t}_{\text{Market Risk}} + \underbrace{\text{FEE}_T - \text{LVR}_T}_{\text{Net Alpha (Hedged P\&L)}}$$

We report:
- **Raw (unhedged) LP P&L**: total pool value change + fees collected
- **Market risk component**: rebalancing strategy P&L (= directional ETH exposure)
- **Hedged LP P&L** (= fees minus LVR): the economically meaningful metric after removing market risk
- **Sharpe Ratio of hedged P&L** at various rebalancing frequencies (1min, 5min, 1H, 4H, 1D), following Table 1 of Milionis et al.

### 6.5 Capital Efficiency

$$\text{Capital Efficiency} = \frac{\text{Fee Revenue per Day}}{\text{TVL}} \times 100\%$$

compared across AMMs at equivalent trade volume. For concentrated liquidity, this is amplified by the range tightness factor $\sqrt{P_b/P_a}$.

### 6.6 [Bonus] Additional Metrics

| Metric | Definition | Motivation |
|--------|-----------|------------|
| **LVR per dollar of TVL** | $\ell(\sigma, P) / V(P)$ | Normalizes for pool size; V2 = $\sigma^2/8$ constant |
| **Break-even volume** | Min trade volume s.t. $\text{FEE}_T = \text{LVR}_T$ | Design guidance: when is the pool profitable for LPs? |
| **LVH / LVR ratio** | $\text{LVH}_T / \text{LVR}_T$ | Measures "noisiness" of impermanent loss vs. LVR |
| **Reversion probability** | Prob. $P_T = P_0$ within 30 days | Context for LVH reversion argument |
| **Effective spread** | $2 \times \text{Slippage}(1\%\text{ trade})$ | Comparison with LOB bid-ask spread |
| **Range utilization** (V3) | Fraction of time price is in range | Relates to effective capital deployment |

---

## 7. Code Architecture

The project extends the reference codebase (`atiselsts/uniswap-lp-articles-code`) with the following modules:

```
amm_comparison/
├── amm/
│   ├── v2_amm.py          # Extended from v2_math.py; adds LVR, IL, slippage
│   ├── v3_amm.py          # Extended from v3_math.py; adds LVR, hedged P&L
│   ├── curve_amm.py       # NEW: StableSwap invariant, numerical price/LVR
│   ├── balancer_amm.py    # NEW: Weighted geometric mean; closed-form LVR
│   └── base_amm.py        # Abstract base class (shared interface)
├── simulation/
│   ├── price_path.py      # GBM simulation (Euler-Maruyama, vectorized)
│   ├── noise_trader.py    # Poisson trade arrival process
│   └── engine.py          # Main simulation loop
├── metrics/
│   ├── slippage.py        # Price impact computation
│   ├── impermanent_loss.py # LVH computation
│   ├── lvr.py             # Analytical + empirical LVR
│   └── pnl.py             # LP P&L decomposition, Sharpe
├── plots/
│   ├── plot_slippage.py
│   ├── plot_il.py
│   ├── plot_lvr.py
│   └── plot_pnl.py
└── run_all.py             # Master script: runs all experiments
```

**Dependencies:** `numpy`, `scipy`, `matplotlib`, `pandas` (matching the reference repo's `requirements.txt` plus `scipy` for Curve's numerical solve).

---

## 8. Experiment Matrix

| Experiment | AMMs | Primary Variable | Fixed Parameters | Output |
|------------|------|-----------------|-----------------|--------|
| **E1: Slippage Curves** | V2, V3 (4 ranges), Curve (5 A values), Balancer (3 weights) | Trade size / TVL ∈ [0.01%, 20%] | TVL = $1M, $P_0 = 1$ | Slippage vs. trade size (log scale) |
| **E2: IL vs. Price Ratio** | V2, V3 (4 ranges), Curve (5 A values), Balancer (3 weights) | Price ratio $r = P_T/P_0 \in [0.1, 10]$ | Static (no path) | IL% vs. price ratio |
| **E3: LVR vs. Volatility** | V2, V3 (medium range), Curve ($A=100$), Balancer (50/50) | $\sigma \in \lbrace 1\%, 3\%, 5\%, 10\%, 20\% \rbrace$ | TVL = $1M, $T$ = 30d, 1,000 paths | Mean $\pm$ std LVR per day |
| **E4: LVR per TVL** | All AMMs | $\sigma$ and range/weight/A | TVL = $1M | $\ell/V$ as function of $\sigma^2$ |
| **E5: Hedged LP P&L** | V2, V3 (medium range), Curve ($A=100$) | Trade volume (low/med/high) | $\sigma = 5\%$, 1,000 paths | Mean hedged P&L, Sharpe by rebalancing freq. |
| **E6: V3 Range Study** | V3 | Range width $r$ | $\sigma = 5\%$, TVL = $1M | LVR/TVL, capital efficiency, range utilization |
| **E7: Curve A Study** | Curve | $A \in \lbrace1, 10, 100, 1000, 10000\rbrace$ | $\sigma = 2\%$ (stablecoin), TVL = $1M | Slippage, LVR/TVL, IL |
| **E8: Break-even Volume** | All AMMs | Trade volume | $\sigma \in \lbrace 1\%, 5\%, 10\% \rbrace$ | Fee revenue vs. LVR; break-even curve |

---

## 9. Visualization Plan

| Figure | Type | Description |
|--------|------|-------------|
| Fig. 1 | Line chart | Bonding curves $(x, y)$ for all AMMs at $L = const$ |
| Fig. 2 | Log-log line chart | Slippage vs. trade size/TVL (E1) |
| Fig. 3 | Line chart | IL% vs. price ratio (E2) |
| Fig. 4 | Bar chart + error bars | Mean daily LVR by AMM and volatility regime (E3) |
| Fig. 5 | Heatmap | LVR/TVL as function of ($\sigma$, AMM parameter) (E4) |
| Fig. 6 | Multi-line cumulative P&L | Hedged LP P&L (pool, hedged at 1min/5min/1H/4H/1D, fees-minus-LVR) — replicating style of Milionis et al. Fig. 5 but across AMMs |
| Fig. 7 | Scatter | Sharpe ratio vs. rebalancing frequency, by AMM (E5) |
| Fig. 8 | Line chart | V3 LVR/TVL and capital efficiency vs. range width (E6) |
| Fig. 9 | Multi-panel | Curve: slippage, IL, LVR vs. amplification A (E7) |
| Fig. 10 | Line chart | Break-even volume curve: fee revenue vs. LVR (E8) |
| Fig. 11 [Bonus] | Radar chart | Multi-metric AMM comparison spider plot |

---

## 10. Expected Results and Hypotheses

**Slippage:** Curve (high $A$) will have the lowest slippage for small-to-medium trades near peg, outperforming all other AMMs. Uniswap V3 with a narrow range will match Curve's efficiency while the price stays in range, but suffer from zero liquidity outside the range. V2 will have intermediate slippage. Balancer 80/20 will have higher slippage than 50/50 due to the asymmetric inventory.

**Impermanent Loss:** Curve (high $A$) will show dramatically lower IL than CPMM-based AMMs for small price deviations near peg, but IL blows up rapidly if the peg breaks (price moves outside the near-constant-sum region). V3 LPs will show higher IL per dollar when out of range (since the pool is no longer providing liquidity). Balancer 80/20 will show lower IL than 50/50 because the LP holds more of the appreciating asset.

**LVR:** Following the analytical formulas, LVR/TVL for V2 equals exactly $\sigma^2/8$ per unit time — a constant that does not depend on price level. Balancer 80/20 ($\theta = 0.8$) will have LVR/TVL = $\frac{\sigma^2}{2}(0.8)(0.2) = 0.08\sigma^2$, lower than V2's $0.125\sigma^2$. V3 concentrated positions will show dramatically higher LVR/TVL than V2, amplified by the reciprocal of the range fraction. Curve will show very low LVR for stablecoin pairs (low $\sigma$, low $|x^{*\prime}|$ near peg) but potentially higher LVR than CPMM during depeg events.

**Hedged LP P&L:** In all cases, the dominant driver of variation in raw LP returns is market risk (directional asset exposure), consistent with Milionis et al. Table 1's finding that unhedged Sharpe ≈ -0.15 while hedged Sharpe ≈ 1.75–23 depending on rebalancing frequency. Curve stablecoin pools should show positive hedged P&L (fees >> LVR) most of the time, making them the most LP-friendly for low-volatility pairs. V3 narrow-range positions will be the most LP-profitable when in range but catastrophic when out of range.

---

## 11. Timeline

| Week | Tasks |
|------|-------|
| 1 | Mathematical model derivation for all 4 AMMs; implement `curve_amm.py` and `balancer_amm.py` |
| 2 | Price path simulator; E1 (slippage) and E2 (IL) experiments; Figs. 1–3 |
| 3 | LVR engine (analytical + empirical delta-hedge); E3, E4, E6, E7; Figs. 4–5, 8–9 |
| 4 | Noise trader simulation; E5, E8; Figs. 6–7, 10; polish, bonus metrics |

---

## 12. References

- Milionis, J., Moallemi, C. C., Roughgarden, T., & Zhang, A. L. (2024). Automated Market Making and Loss-Versus-Rebalancing. *arXiv:2208.06046v5*.
- Adams, H. et al. (2020). *Uniswap v2 Core*. Uniswap.
- Adams, H. et al. (2021). *Uniswap v3 Core*. Uniswap.
- Egorov, M. (2019). *StableSwap — Efficient Mechanism for Stablecoin Liquidity*. Curve Finance.
- Martinelli, F., & Mushegian, N. (2019). *A Non-Custodial Portfolio Manager, Liquidity Provider, and Price Sensor*. Balancer Labs.
- Elsts, A. (2023). Conceptualizing Uniswap v3 LP Profit and Loss. Medium / `atiselsts/uniswap-lp-articles-code`.
- Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. *Journal of Political Economy*, 81(3), 637–654.
- Carr, P., & Madan, D. (2001). Towards a Theory of Volatility Trading. *Handbooks in Mathematical Finance*.