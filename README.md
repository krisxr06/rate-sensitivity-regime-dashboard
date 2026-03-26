# Rate Sensitivity & Regime Dashboard

## Core Question

> *How does bond duration risk behave across different yield curve regimes?*

## Why This Project Exists

Duration risk is not static. The same 10-year bond carries different practical
risk depending on whether the yield curve is inverted, re-steepening, flat, or
normal. This dashboard quantifies that relationship empirically — using
historical FRED data to show how yield volatility, directional bias, and
approximate price sensitivity have varied across regime states.

## Data Sources

All data from the Federal Reserve Economic Data (FRED):

| Series | Description | Frequency |
|--------|-------------|-----------|
| DGS2   | 2-Year Treasury Constant Maturity | Daily → monthly |
| DGS10  | 10-Year Treasury Constant Maturity | Daily → monthly |
| DGS30  | 30-Year Treasury Constant Maturity | Daily → monthly |
| FEDFUNDS | Effective Federal Funds Rate | Monthly |

All series resampled to month-end frequency. Short gaps (≤ 3 months) are
forward-filled. DGS30 was discontinued Feb 2002 – Feb 2006; that gap is
preserved as NaN and excluded from 30Y statistics.

## Methodology

### Regime Classification
See *Regime Definitions* below. Same thresholds as companion project
`yield-curve-inflation-dashboard`.

### Yield Change Statistics
Monthly first differences of each yield series (ppts). Average and standard
deviation computed per regime, converted to basis points for display.

### DV01 Proxy — Important Disclaimer

> **This is a simplified educational approximation. It is not suitable for
> risk management, trading, or investment decisions.**

Formula:
```
modified_duration ≈ maturity × 0.85   (DURATION_FACTOR)
dv01_proxy        = −modified_duration × (monthly_yield_change / 100)
```

Result: approximate fractional price change per month (×100 for %).
The 0.85 factor is a rough proxy for the relationship between maturity and
modified duration across a range of coupon levels. Real-world DV01 requires:
- Actual coupon and settlement terms
- Day-count conventions
- Full yield-to-price mapping (especially non-trivial for large moves)
- Convexity adjustment for large yield changes

The 30Y proxy (modified duration ≈ 25.5) will show the largest price
sensitivity, which is directionally correct but numerically approximate.

### Directional Bias Analysis
For each regime: percentage of months where 10Y yields rose vs fell, and
average magnitude in each direction. The point is that regimes differ not
just in volatility magnitude but in directional tendency.

### Rolling 12-Month Adverse Moves
Rolling 12-month cumulative yield change computed for each maturity. The 90th
percentile of adverse (upward) moves is shown per regime — representing a
historically significant but not extreme rate stress within that regime.

## Regime Definitions

Yield curve regimes are based on the 10Y–2Y spread, applied in strict
priority order:

1. **Re-steepening** (highest priority) — ALL must hold:
   - Spread was negative at any point in prior 6 months
   - Spread has risen > 0.25 ppts over last 3 months
   - Current spread is between −0.25 and +0.75 ppts
2. **Inverted** — spread < 0
3. **Flat** — spread 0 to 0.50 ppts
4. **Normal** — spread > 0.50 ppts

## Dashboard Walkthrough

| Section | Content |
|---------|---------|
| **1 — Current Snapshot** | 2Y, 10Y, 30Y yields; Fed Funds; spread; current regime; 2-sentence context |
| **2 — Yield Volatility** | Grouped bar: avg change & σ by regime for 2Y and 10Y |
| **3 — Duration Sensitivity** | DV01 heatmap + directional bias table + adverse moves chart |
| **4 — Risk Summary** | 4 data-driven bullets: highest volatility, most exposed maturity, directional bias, current regime |
| **5 — Regime → Portfolio Playbook** | Heuristic interpretation layer: volatility label, directional bias, regime character, duration risk, carry environment, convexity value, and positioning context bullets — all derived rule-based from historical regime statistics |

## Key Takeaways

- **Inverted regimes** have historically exhibited the highest 2Y yield volatility
  as policy rate expectations shift.
- **30Y duration** amplifies adverse moves significantly relative to 2Y in all
  regimes — the DV01 heatmap makes this explicit.
- **Re-steepening** is rare (~4% of months) but has historically coincided with
  transition dynamics that compress term premium.
- **Directional bias** varies by regime: not all yield changes in a given regime
  are adverse — the % of months rising vs falling matters as much as magnitude.
- **The Playbook section** translates current regime into descriptive risk labels (duration, carry, convexity)
  and positioning context derived from historical regime statistics — not forecasts.

## How to Run

```bash
cd rate-sensitivity-regime-dashboard
pip install -r requirements.txt
streamlit run app.py
```

### Optional: FRED API Key

```bash
cp .env.example .env
# Edit .env: FRED_API_KEY=your_key_here
```

Free API key: <https://fred.stlouisfed.org/docs/api/api_key.html>

Data is cached locally in `data/processed/` as parquet files on first run
(refreshed every 24 hours).

## Disclaimer

This is an educational portfolio project. It is not investment advice.
DV01 values are simplified proxies using a modified duration approximation.
All regime logic is rule-based and for analytical illustration only.
Historical patterns are not predictive of future outcomes.
