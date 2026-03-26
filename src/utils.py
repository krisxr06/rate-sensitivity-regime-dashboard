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


# ── Section 5: Regime → Portfolio Playbook ────────────────────────────────────

def get_portfolio_playbook(df: pd.DataFrame, vals: dict) -> dict:
    """
    Descriptive heuristic layer translating the current yield curve regime
    into historically informed portfolio risk labels and positioning context.

    This is NOT a trading signal or investment recommendation.
    All outputs are rule-based interpretations of historical regime statistics.

    Returns a dict with:
        current_regime, volatility_label, directional_bias, regime_character,
        duration_risk, carry_environment, convexity_value, bullets
    """
    current_regime = vals.get("yc_regime", "Unknown")

    # ── Volatility label ──────────────────────────────────────────────────────
    # Derived from the current regime's 10Y monthly std dev (bps).
    # Thresholds: Low < 25 bps | Moderate 25–45 bps | High > 45 bps
    stats = compute_regime_yield_stats(df)
    vol_label = "N/A"
    if current_regime in stats.index:
        std_10y = stats.loc[current_regime, "std_10y"]
        if std_10y < 25:
            vol_label = "Low"
        elif std_10y <= 45:
            vol_label = "Moderate"
        else:
            vol_label = "High"

    # ── Directional bias label ────────────────────────────────────────────────
    # % of months the 10Y yield rose within the current regime.
    # Rose ≥ 55% → "Tended to Rise" | Fell ≥ 55% → "Tended to Fall" | else Mixed
    bias_label = "Mixed"
    valid = df[df["yc_regime"] == current_regime].dropna(subset=["DGS10_chg"])
    if len(valid) >= 5:
        pct_rose = (valid["DGS10_chg"] > 0).mean() * 100
        pct_fell = (valid["DGS10_chg"] < 0).mean() * 100
        if pct_rose >= 55:
            bias_label = "Tended to Rise"
        elif pct_fell >= 55:
            bias_label = "Tended to Fall"

    # ── Regime character ──────────────────────────────────────────────────────
    # Descriptive structural label — not a forecast.
    character_map = {
        "Re-steepening": "Transitional",
        "Flat":          "Transitional",
        "Inverted":      "Stressed",
        "Normal":        "Stable",
    }
    regime_character = character_map.get(current_regime, "Unknown")

    # ── Portfolio interpretation (fixed heuristic mapping) ────────────────────
    # Descriptive labels derived from regime structure. Not predictions.
    # "Low to Moderate" → Low; "Moderate to High" → High (simplified for display).
    portfolio_map = {
        "Normal":        {"duration_risk": "Moderate", "carry": "Supportive", "convexity": "Low"},
        "Flat":          {"duration_risk": "Moderate", "carry": "Neutral",    "convexity": "Increasing"},
        "Inverted":      {"duration_risk": "High",     "carry": "Weak",       "convexity": "High"},
        "Re-steepening": {"duration_risk": "High",     "carry": "Weak",       "convexity": "High"},
    }
    interp = portfolio_map.get(current_regime, {
        "duration_risk": "N/A", "carry": "N/A", "convexity": "N/A",
    })

    # ── Positioning implication bullets ──────────────────────────────────────
    # Observational, regime-specific context. Not trade recommendations.
    bullets_map = {
        "Normal": [
            "Historically associated with lower rate volatility than inversion or transition phases.",
            "Duration exposure has generally been easier to hold than in re-steepening or inverted regimes.",
            "Optionality has typically been less central than in more volatile transition states.",
        ],
        "Flat": [
            "Historically a more neutral carry environment with less directional clarity.",
            "Positioning has typically required more balance between carry and rate risk.",
            "Optionality can become more relevant if the curve begins to transition.",
            "Flat regimes are typically transition phases rather than stable states.",
        ],
        "Inverted": [
            "Historically associated with tighter carry conditions and elevated policy uncertainty.",
            "Duration outcomes have been less one-directional than the curve shape alone might suggest.",
            "More cautious positioning has typically been favored relative to stable normal-curve periods.",
        ],
        "Re-steepening": [
            "Historically the most volatile transition phase in the sample.",
            "Duration exposure has been harder to manage as rates reprice unevenly.",
            "Optionality has typically become more valuable during these transition periods.",
        ],
    }
    bullets = bullets_map.get(current_regime, [
        "Insufficient regime history for positioning context.",
    ])

    return {
        "current_regime":    current_regime,
        "volatility_label":  vol_label,
        "directional_bias":  bias_label,
        "regime_character":  regime_character,
        "duration_risk":     interp["duration_risk"],
        "carry_environment": interp["carry"],
        "convexity_value":   interp["convexity"],
        "bullets":           bullets,
    }
