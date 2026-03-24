"""
Transformation pipeline for rate sensitivity analysis.

Derived columns added
---------------------
spread_10y2y      : DGS10 − DGS2 (ppts)
spread_3m_chg     : 3-month change in spread
DGS{n}_chg        : month-over-month yield change (ppts) for n ∈ {2, 10, 30}
DGS{n}_dv01       : simplified DV01 proxy (see note below)
DGS{n}_roll12     : rolling 12-month cumulative yield change

DV01 Proxy Methodology
-----------------------
  modified_duration ≈ maturity × 0.85   (DURATION_FACTOR)
  dv01_proxy        = −modified_duration × (monthly_yield_change / 100)

Units: fractional price change per month (multiply by 100 for %).
This is a first-order approximation for educational illustration only.
Real-world DV01 requires coupon, settlement, day-count conventions,
and a full yield-to-price mapping. See README Methodology section.
"""

import pandas as pd

# Simplified modified duration proxy: actual duration < maturity for coupon bonds.
# 0.85 is a rough approximation across a range of coupon levels and maturities.
# Documented assumption — do not treat as precise.
DURATION_FACTOR = 0.85

YIELD_SERIES = ["DGS2", "DGS10", "DGS30"]
MATURITIES   = {"DGS2": 2, "DGS10": 10, "DGS30": 30}


def resample_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample to month-end (ME) frequency using the last available value.
    Short gaps (≤ 3 months) are forward-filled.
    DGS30 gap (Feb 2002 – Feb 2006) exceeds this limit and is preserved as NaN.
    """
    monthly = df.resample("ME").last()
    monthly = monthly.ffill(limit=3)
    return monthly


def compute_spread(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["spread_10y2y"] = df["DGS10"] - df["DGS2"]
    return df


def compute_3m_changes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["spread_3m_chg"] = df["spread_10y2y"].diff(3)
    return df


def compute_monthly_yield_changes(df: pd.DataFrame) -> pd.DataFrame:
    """Month-over-month first difference for each yield series (ppts)."""
    df = df.copy()
    for sid in YIELD_SERIES:
        if sid in df.columns:
            df[f"{sid}_chg"] = df[sid].diff(1)
    return df


def compute_dv01_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simplified DV01 proxy for each maturity.

    dv01_proxy = −modified_duration × (monthly_yield_change / 100)

    A negative value means rates rose → bond price fell.
    A positive value means rates fell → bond price rose.

    Result is in fractional terms; multiply by 100 for % price change.
    """
    df = df.copy()
    for sid, maturity in MATURITIES.items():
        chg_col = f"{sid}_chg"
        if chg_col in df.columns:
            mod_dur = maturity * DURATION_FACTOR
            df[f"{sid}_dv01"] = -mod_dur * (df[chg_col] / 100)
    return df


def compute_rolling_12m_changes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling 12-month cumulative yield change for each maturity.
    Requires at least 6 non-null months in the window (min_periods=6).
    Used to identify adverse yield episodes by regime.
    """
    df = df.copy()
    for sid in YIELD_SERIES:
        chg_col = f"{sid}_chg"
        if chg_col in df.columns:
            df[f"{sid}_roll12"] = df[chg_col].rolling(12, min_periods=6).sum()
    return df


def apply_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full transformation pipeline in order."""
    df = resample_monthly(df)
    df = compute_spread(df)
    df = compute_3m_changes(df)
    df = compute_monthly_yield_changes(df)
    df = compute_dv01_proxies(df)
    df = compute_rolling_12m_changes(df)
    return df
