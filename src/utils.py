"""
Utility functions: extract latest values, compute regime statistics,
generate directional bias table, and produce all dashboard text.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .regimes import YC_REGIMES


# ── Latest values ──────────────────────────────────────────────────────────────

def _is_nan(v) -> bool:
    try:
        return pd.isna(v)
    except (TypeError, ValueError):
        return False


def _fmt(val, suffix: str = "", decimals: int = 2, signed: bool = False) -> str:
    if _is_nan(val):
        return "N/A"
    sign = "+" if signed and float(val) >= 0 else ""
    return f"{sign}{float(val):.{decimals}f}{suffix}"


def latest_values(df: pd.DataFrame) -> dict:
    """Extract most recent non-null value for each key metric."""
    def _last(col):
        if col not in df.columns:
            return float("nan")
        s = df[col].dropna()
        return s.iloc[-1] if not s.empty else float("nan")

    try:
        date = df[["DGS2", "DGS10"]].dropna(how="all").index[-1]
    except (IndexError, KeyError):
        date = df.index[-1] if not df.empty else pd.Timestamp.now()

    regime = _last("yc_regime")

    return {
        "date":      date,
        "dgs2":      _last("DGS2"),
        "dgs10":     _last("DGS10"),
        "dgs30":     _last("DGS30"),
        "fedfunds":  _last("FEDFUNDS"),
        "spread":    _last("spread_10y2y"),
        "yc_regime": str(regime) if not _is_nan(regime) else "Unknown",
    }


# ── Regime yield statistics ────────────────────────────────────────────────────

def compute_regime_yield_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by regime with columns:
        avg_2y, std_2y, avg_10y, std_10y  (all in bps = ppts × 100)
        n_2y, n_10y                        (observation counts)
    """
    valid = df[df["yc_regime"].isin(YC_REGIMES)].copy()
    rows  = []
    for regime in YC_REGIMES:
        sub = valid[valid["yc_regime"] == regime]
        rows.append({
            "regime":  regime,
            "avg_2y":  sub["DGS2_chg"].mean()  * 100,
            "std_2y":  sub["DGS2_chg"].std()   * 100,
            "avg_10y": sub["DGS10_chg"].mean() * 100,
            "std_10y": sub["DGS10_chg"].std()  * 100,
            "n_2y":    sub["DGS2_chg"].dropna().shape[0],
            "n_10y":   sub["DGS10_chg"].dropna().shape[0],
        })
    return pd.DataFrame(rows).set_index("regime")


# ── Directional bias table ─────────────────────────────────────────────────────

def directional_bias_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each regime, compute directional rate statistics for the 10Y yield:
        pct_rose          : % of months where 10Y yield increased
        pct_fell          : % of months where 10Y yield fell
        avg_rise_bps      : avg monthly increase when rising (bps)
        avg_fall_bps      : avg monthly decline when falling (bps, absolute)

    Rows with < 5 observations are marked as low-confidence.
    """
    valid = df[df["yc_regime"].isin(YC_REGIMES)].dropna(subset=["DGS10_chg"]).copy()
    rows  = []
    for regime in YC_REGIMES:
        sub    = valid[valid["yc_regime"] == regime]["DGS10_chg"]
        n      = len(sub)
        rising = sub[sub > 0]
        falling = sub[sub < 0]

        if n < 5:
            rows.append({
                "Regime": regime, "n": n,
                "% Months Rates Rose": "—", "% Months Rates Fell": "—",
                "Avg Rise (bps)": "—", "Avg Fall (bps)": "—",
            })
        else:
            rows.append({
                "Regime":               regime,
                "n":                    n,
                "% Months Rates Rose":  f"{len(rising)/n*100:.0f}%",
                "% Months Rates Fell":  f"{len(falling)/n*100:.0f}%",
                "Avg Rise (bps)":       f"{rising.mean()*100:+.1f}" if not rising.empty else "—",
                "Avg Fall (bps)":       f"{falling.mean()*100:.1f}" if not falling.empty else "—",
            })
    return pd.DataFrame(rows).set_index("Regime")


# ── Drawdown stats (feeds Section 4 bullets) ──────────────────────────────────

def compute_adverse_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (regime, maturity), compute the 90th-percentile rolling 12m
    cumulative yield increase (bps). Used to identify the most adverse
    historical environment per maturity bucket.
    """
    valid = df[df["yc_regime"].isin(YC_REGIMES)].copy()
    mats  = [("DGS2", "2Y"), ("DGS10", "10Y"), ("DGS30", "30Y")]
    rows  = []
    for regime in YC_REGIMES:
        sub = valid[valid["yc_regime"] == regime]
        row = {"regime": regime}
        for sid, label in mats:
            col     = f"{sid}_roll12"
            adverse = sub[col].dropna()
            adverse = adverse[adverse > 0]
            row[label] = adverse.quantile(0.90) * 100 if not adverse.empty else 0.0
        rows.append(row)
    return pd.DataFrame(rows).set_index("regime")


# ── Section 1: Snapshot summary ───────────────────────────────────────────────

def snapshot_summary(vals: dict) -> str:
    """2–3 sentence observational summary of the current rate environment."""
    yc     = vals.get("yc_regime", "Unknown")
    spread = vals.get("spread", float("nan"))
    dgs2   = vals.get("dgs2",   float("nan"))
    dgs10  = vals.get("dgs10",  float("nan"))
    ff     = vals.get("fedfunds", float("nan"))

    spread_s = _fmt(spread, " ppts", signed=True)
    ff_s     = _fmt(ff,     "%")

    yc_map = {
        "Inverted":
            f"Yield curve: **inverted** ({spread_s}) — short rates exceed long rates.",
        "Re-steepening":
            f"Yield curve: **re-steepening** ({spread_s}) following a period of inversion.",
        "Flat":
            f"Yield curve: **flat** ({spread_s}) — limited term premium between short and long rates.",
        "Normal":
            f"Current regime: **Normal** ({spread_s} spread).",
    }
    yc_sent = yc_map.get(yc, f"Yield curve regime: **{yc}** ({spread_s}).")

    ctx_map = {
        "Inverted":
            "Historically associated with elevated short-end yields relative to long rates and compressed term premium.",
        "Re-steepening":
            "Historically a transitional environment — spread widening after inversion has preceded varied rate outcomes across prior cycles.",
        "Flat":
            "Historically associated with contained rate volatility; directional bias has depended on the stage of the policy cycle.",
        "Normal":
            "Historically associated with lower rate volatility than inversion or re-steepening phases, though regime alone does not determine duration outcomes.",
    }
    ctx_sent = ctx_map.get(yc, "Historical context for this regime is limited.")

    disclaimer = (
        f"Fed Funds: **{ff_s}**. "
        "*This framework is descriptive rather than predictive — intended to "
        "contextualize rate environments, not forecast them.*"
    )

    return f"{yc_sent}  \n{ctx_sent}  \n{disclaimer}"


# ── Section 4: Regime Duration Risk bullets ────────────────────────────────────

def section4_bullets(df: pd.DataFrame, vals: dict) -> str:
    """
    Generate 4 sharp, data-driven bullet points for the summary section.
    """
    stats    = compute_regime_yield_stats(df)
    adverse  = compute_adverse_stats(df)
    bias_raw = df[df["yc_regime"].isin(YC_REGIMES)].dropna(subset=["DGS10_chg"])

    # ── Bullet 1: highest volatility regime ──────────────────────────────────
    if not stats.empty and "std_10y" in stats.columns:
        vol_regime = stats["std_10y"].idxmax()
        vol_val    = stats.loc[vol_regime, "std_10y"]
        avg_val    = stats.loc[vol_regime, "avg_10y"]
        b1 = (
            f"- **Highest 10Y volatility regime:** {vol_regime} "
            f"(σ = {vol_val:.0f} bps/month). Average monthly 10Y change: {avg_val:+.0f} bps."
        )
    else:
        b1 = "- **Highest volatility regime:** insufficient data"

    # ── Bullet 2: most exposed maturity ──────────────────────────────────────
    if not adverse.empty:
        # Find (regime, maturity) with highest 90th-pct adverse move
        worst_val    = 0.0
        worst_regime = "N/A"
        worst_mat    = "N/A"
        for mat in ["2Y", "10Y", "30Y"]:
            if mat in adverse.columns:
                idx = adverse[mat].idxmax()
                if adverse.loc[idx, mat] > worst_val:
                    worst_val    = adverse.loc[idx, mat]
                    worst_regime = idx
                    worst_mat    = mat
        b2 = (
            f"- **Largest adverse 12M rate move:** {worst_mat} during {worst_regime} regimes "
            f"(~{worst_val:.0f} bps, 90th percentile)."
        )
    else:
        b2 = "- **Most exposed maturity:** insufficient data"

    # ── Bullet 3: directional bias for most historically notable regime ────────
    # Use Inverted if it has data; fall back to highest-vol regime
    bias_regime = "Inverted" if "Inverted" in stats.index and stats.loc["Inverted", "n_10y"] >= 5 else vol_regime
    sub_bias = bias_raw[bias_raw["yc_regime"] == bias_regime]["DGS10_chg"]
    if not sub_bias.empty:
        pct_rose = (sub_bias > 0).mean() * 100
        pct_fell = (sub_bias < 0).mean() * 100
        b3 = (
            f"- **Directional bias:** in {bias_regime} regimes, 10Y yields rose "
            f"{pct_rose:.0f}% of months vs fell {pct_fell:.0f}%"
        )
    else:
        b3 = f"- **Directional bias:** insufficient data for {bias_regime}"

    # ── Bullet 4: current regime characterisation ─────────────────────────────
    current = vals.get("yc_regime", "Unknown")
    char_map = {
        "Inverted":
            "historically associated with elevated rate volatility and compressed term spread",
        "Re-steepening":
            "historically associated with transitional rate dynamics following inversion",
        "Flat":
            "historically associated with contained volatility relative to inversion periods",
        "Normal":
            "historically associated with lower rate volatility than inversion or re-steepening phases",
        "Unknown":
            "insufficient history for characterisation",
    }
    b4 = f"- **Current regime:** {current} — {char_map.get(current, 'see historical data')}"

    return "  \n".join([b1, b2, b3, b4])
