"""
Yield curve regime classification.

Regimes applied in strict priority order (highest to lowest):
    1. Re-steepening  — post-inversion spread widening
    2. Inverted       — spread < 0
    3. Flat           — spread 0 to 0.50
    4. Normal         — spread > 0.50

Documented assumption: when spread_3m_chg is NaN (first 3 months), the
Re-steepening condition cannot be evaluated and defaults to the level-based
classification below it in priority order.
"""

import pandas as pd

YC_REGIMES = ["Inverted", "Re-steepening", "Flat", "Normal"]

YC_COLORS = {
    "Inverted":      "rgba(204,  51,  51, 0.22)",
    "Re-steepening": "rgba(230, 150,  30, 0.22)",
    "Flat":          "rgba(210, 195,  50, 0.22)",
    "Normal":        "rgba( 50, 180,  80, 0.22)",
    "Unknown":       "rgba(150, 150, 150, 0.10)",
}


def classify_yc_regimes(df: pd.DataFrame) -> pd.Series:
    """
    Vectorised yield curve regime classification.

    Re-steepening conditions (ALL must hold):
        • Spread was negative at any point in prior 6 months
          (shift(1) excludes current month; rolling(6) looks 6 months back)
        • Spread has increased > 0.25 ppts over last 3 months
        • Current spread is between −0.25 and +0.75 ppts

    Rules applied lowest→highest priority so the last assignment wins.
    """
    spread    = df["spread_10y2y"]
    spread_3m = df["spread_3m_chg"]

    had_neg_prior_6m = spread.shift(1).rolling(6, min_periods=1).min() < 0

    cond_re_steep = (
        had_neg_prior_6m
        & (spread_3m > 0.25)
        & (spread >= -0.25)
        & (spread <= 0.75)
    )

    regimes = pd.Series("Unknown", index=df.index, dtype=object)
    regimes[spread > 0.50]                    = "Normal"
    regimes[(spread >= 0) & (spread <= 0.50)] = "Flat"
    regimes[spread < 0]                       = "Inverted"
    regimes[cond_re_steep]                    = "Re-steepening"
    regimes[spread.isna()]                    = "Unknown"

    return regimes.rename("yc_regime")


def add_regimes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["yc_regime"] = classify_yc_regimes(df)
    return df
