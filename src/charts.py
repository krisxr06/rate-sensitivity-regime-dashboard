"""
Chart creation module.

Public functions
----------------
yield_volatility_chart  : grouped bar — avg change & std dev by regime (2Y, 10Y)
dv01_heatmap            : heatmap — avg DV01 proxy % by regime × maturity
adverse_moves_chart     : bar — worst rolling 12m cumulative yield rise by regime
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .regimes import YC_REGIMES, YC_COLORS

_TEMPLATE    = "plotly_dark"
_FONT        = "Inter, 'Helvetica Neue', Arial, sans-serif"
_GRID        = "rgba(255,255,255,0.07)"
_ZERO        = "rgba(255,255,255,0.30)"

# Regime display colours (solid)
_REGIME_SOLID = {
    "Inverted":      "#dc3333",
    "Re-steepening": "#e6962a",
    "Flat":          "#d4c44a",
    "Normal":        "#46b855",
    "Unknown":       "#888888",
}

MATURITY_LABELS = {"DGS2": "2Y", "DGS10": "10Y", "DGS30": "30Y"}


# ── Section 2: Yield Volatility ────────────────────────────────────────────────

def yield_volatility_chart(df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart: average monthly yield change and standard deviation
    (in basis points) for 2Y and 10Y Treasuries, by yield curve regime.
    """
    valid = df[df["yc_regime"].isin(YC_REGIMES)].copy()

    rows = []
    for regime in YC_REGIMES:
        sub = valid[valid["yc_regime"] == regime]
        n   = sub[["DGS2_chg", "DGS10_chg"]].dropna(how="all").shape[0]
        rows.append({
            "regime":   regime,
            "avg_2y":   sub["DGS2_chg"].mean() * 100,
            "std_2y":   sub["DGS2_chg"].std()  * 100,
            "avg_10y":  sub["DGS10_chg"].mean() * 100,
            "std_10y":  sub["DGS10_chg"].std()  * 100,
            "n":        n,
        })
    stats = pd.DataFrame(rows).set_index("regime")

    fig = go.Figure()

    bar_defs = [
        ("avg_2y",  "2Y Avg Change",  "#5b9bd5", True),
        ("std_2y",  "2Y Volatility",  "#a3c4e8", False),
        ("avg_10y", "10Y Avg Change", "#f4a460", True),
        ("std_10y", "10Y Volatility", "#f7c898", False),
    ]

    for col, name, color, show_zero in bar_defs:
        fig.add_trace(go.Bar(
            name=name,
            x=stats.index,
            y=stats[col].round(1),
            marker_color=color,
            hovertemplate=f"<b>{name}</b><br>%{{x}}: %{{y:.1f}} bps<extra></extra>",
        ))

    fig.add_hline(y=0, line_dash="dot", line_color=_ZERO, line_width=1)

    # Sample count annotations
    for i, regime in enumerate(stats.index):
        fig.add_annotation(
            x=regime, y=stats.loc[regime, ["avg_2y","std_2y","avg_10y","std_10y"]].abs().max() + 4,
            text=f"n={int(stats.loc[regime,'n'])}",
            showarrow=False, font=dict(size=9, color="#888"), yref="y",
        )

    fig.update_layout(
        template=_TEMPLATE,
        barmode="group",
        title=dict(text="Monthly Yield Change & Volatility by Regime (bps)", font=dict(size=15)),
        xaxis=dict(title="Yield Curve Regime", showgrid=False),
        yaxis=dict(title="Basis Points", showgrid=True, gridcolor=_GRID),
        legend=dict(orientation="h", y=-0.18, x=0),
        font=dict(family=_FONT),
        margin=dict(l=60, r=20, t=55, b=80),
        height=420,
    )
    return fig


# ── Section 3: DV01 Heatmap ───────────────────────────────────────────────────

def dv01_heatmap(df: pd.DataFrame, current_regime: str) -> go.Figure:
    """
    Heatmap of average DV01 proxy impact (% price change) by
    yield curve regime × bond maturity.

    Negative = rates rose on average → adverse for bond holders (red).
    Positive = rates fell on average → favourable (green).

    Values are × 100 to express as approximate % price change.
    """
    valid = df[df["yc_regime"].isin(YC_REGIMES)].copy()

    maturities = ["2Y", "10Y", "30Y"]
    dv01_cols  = {"2Y": "DGS2_dv01", "10Y": "DGS10_dv01", "30Y": "DGS30_dv01"}

    z      = []
    text   = []
    annots = []

    for regime in YC_REGIMES:
        sub  = valid[valid["yc_regime"] == regime]
        row_z, row_t = [], []
        for mat in maturities:
            col  = dv01_cols[mat]
            vals = sub[col].dropna() * 100  # fractional → %
            avg  = vals.mean() if not vals.empty else float("nan")
            row_z.append(round(avg, 2) if not np.isnan(avg) else 0)
            row_t.append(f"{avg:+.2f}%" if not np.isnan(avg) else "N/A")
        z.append(row_z)
        text.append(row_t)

    # Highlight current regime
    if current_regime in YC_REGIMES:
        ci = YC_REGIMES.index(current_regime)
        for j in range(len(maturities)):
            annots.append(dict(
                x=maturities[j], y=current_regime,
                text="▶", showarrow=False,
                font=dict(size=10, color="white", family=_FONT),
                xref="x", yref="y",
            ))

    fig = go.Figure(go.Heatmap(
        z=z,
        x=maturities,
        y=YC_REGIMES,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=12, family=_FONT),
        colorscale="RdYlGn",
        zmid=0,
        colorbar=dict(
            title=dict(text="Avg % Price Chg", side="right"),
            thickness=14, len=0.85,
            tickformat=".1f",
        ),
        hovertemplate=(
            "Regime: <b>%{y}</b><br>"
            "Maturity: <b>%{x}</b><br>"
            "Avg DV01 impact: <b>%{text}</b><extra></extra>"
        ),
        xgap=3, ygap=3,
    ))

    fig.update_layout(
        template=_TEMPLATE,
        title=dict(
            text="Avg DV01 Proxy Impact (% price change/month) by Regime & Maturity",
            font=dict(size=15),
        ),
        xaxis=dict(title="Bond Maturity", tickfont=dict(size=13)),
        yaxis=dict(title="Yield Curve Regime", autorange="reversed", tickfont=dict(size=12)),
        font=dict(family=_FONT),
        margin=dict(l=140, r=80, t=60, b=60),
        height=360,
        annotations=annots,
    )
    return fig


# ── Section 3: Adverse Moves ──────────────────────────────────────────────────

def adverse_moves_chart(df: pd.DataFrame) -> go.Figure:
    """
    Bar chart of the 90th-percentile rolling 12-month cumulative yield INCREASE
    (in bps) for each maturity, grouped by yield curve regime.

    Shows which regimes have historically produced the largest adverse
    (upward) yield moves for each maturity bucket.
    Only positive rolling 12m changes are included in the percentile calculation
    (adverse direction only — the scenario where rates rise and prices fall).
    """
    valid = df[df["yc_regime"].isin(YC_REGIMES)].copy()

    maturities = [("DGS2", "2Y"), ("DGS10", "10Y"), ("DGS30", "30Y")]
    fig = go.Figure()

    colors = {"2Y": "#5b9bd5", "10Y": "#f4a460", "30Y": "#c084e8"}

    for sid, label in maturities:
        col  = f"{sid}_roll12"
        vals = []
        for regime in YC_REGIMES:
            sub = valid[valid["yc_regime"] == regime][col].dropna()
            # Adverse = positive cumulative change (rates rose)
            adverse = sub[sub > 0]
            p90 = adverse.quantile(0.90) * 100 if not adverse.empty else 0  # ppts → bps
            vals.append(round(p90, 1))

        fig.add_trace(go.Bar(
            name=f"{label} Bond",
            x=YC_REGIMES,
            y=vals,
            marker_color=colors[label],
            hovertemplate=f"<b>{label}</b><br>%{{x}}: %{{y:.0f}} bps<extra></extra>",
        ))

    fig.update_layout(
        template=_TEMPLATE,
        barmode="group",
        title=dict(
            text="90th-Pct Rolling 12M Cumulative Yield Increase by Regime (bps)",
            font=dict(size=15),
        ),
        xaxis=dict(title="Yield Curve Regime", showgrid=False),
        yaxis=dict(title="Basis Points", showgrid=True, gridcolor=_GRID),
        legend=dict(orientation="h", y=-0.18, x=0),
        font=dict(family=_FONT),
        margin=dict(l=60, r=20, t=55, b=80),
        height=380,
    )
    return fig
