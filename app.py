"""
Rate Sensitivity & Regime Dashboard
=====================================
Main Streamlit application.

Sections
--------
1. Current Snapshot         — yield levels, spread, regime, summary
2. Yield Volatility         — avg change & std dev by regime (2Y, 10Y)
3. Duration Sensitivity     — DV01 heatmap + directional bias table + adverse moves
4. Regime Duration Risk     — 4-bullet summary
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_all
from src.transforms  import apply_transforms
from src.regimes     import add_regimes
from src.charts      import yield_volatility_chart, dv01_heatmap, adverse_moves_chart
from src.utils       import (
    latest_values,
    snapshot_summary,
    directional_bias_table,
    section4_bullets,
    get_portfolio_playbook,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rate Sensitivity Across Yield Curve Regimes",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
[data-testid="metric-container"] {
    background: #1a1d2e;
    border: 1px solid #2a2f45;
    border-radius: 10px;
    padding: 14px 18px 10px;
}
.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 15px;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.badge-inverted   { background:#3a0e0e; color:#ff7070; border:1px solid #6b1f1f; }
.badge-resteep    { background:#362200; color:#ffb347; border:1px solid #6b4a00; }
.badge-flat       { background:#2e2900; color:#e8d44d; border:1px solid #5a5000; }
.badge-normal     { background:#0d2914; color:#5dde7c; border:1px solid #1e5c30; }
.badge-unknown    { background:#1e2030; color:#9da5b4; border:1px solid #3a3f55; }
.divider { border-top:1px solid #2a2f45; margin:28px 0 20px; }
.caption-text { font-size:13px; color:#9ca3b0; line-height:1.65; padding:6px 0 0; }
.disclaimer { font-size:12px; color:#7a8299; font-style:italic; padding:6px 0 2px; }
</style>
""", unsafe_allow_html=True)


# ── Data ───────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_data() -> pd.DataFrame:
    return add_regimes(apply_transforms(load_all()))


with st.spinner("Fetching FRED data…"):
    df = get_data()

vals       = latest_values(df)
current_yc = vals["yc_regime"]


# ── Header ─────────────────────────────────────────────────────────────────────
st.title("Rate Sensitivity Across Yield Curve Regimes")
st.markdown("##### Bond Duration Risk Across Yield Curve Regimes")
try:
    date_str = vals["date"].strftime("%B %Y")
except Exception:
    date_str = "latest available"
st.caption(f"Data through **{date_str}** · Source: FRED (DGS2, DGS10, DGS30, FEDFUNDS)")
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


def _fmt(val, suffix="", decimals=2, signed=False):
    try:
        if pd.isna(val):
            return "N/A"
        sign = "+" if signed and float(val) >= 0 else ""
        return f"{sign}{float(val):.{decimals}f}{suffix}"
    except Exception:
        return str(val)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Current Snapshot
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("1 — Current Snapshot")

c1, c2, c3, c4 = st.columns(4)
c1.metric("2Y Treasury",    _fmt(vals["dgs2"],     "%"))
c2.metric("10Y Treasury",   _fmt(vals["dgs10"],    "%"))
c3.metric("30Y Treasury",   _fmt(vals["dgs30"],    "%"))
c4.metric("Fed Funds Rate", _fmt(vals["fedfunds"], "%"))

st.write("")

c5, c6 = st.columns([1, 3])
c5.metric("10Y–2Y Spread", _fmt(vals["spread"], " ppts", signed=True))

_badge = {
    "Inverted":      "badge-inverted",
    "Re-steepening": "badge-resteep",
    "Flat":          "badge-flat",
    "Normal":        "badge-normal",
}.get(current_yc, "badge-unknown")

with c6:
    st.markdown("**Yield Curve Regime**")
    st.markdown(
        f"<span class='badge {_badge}'>{current_yc}</span>",
        unsafe_allow_html=True,
    )

st.write("")
st.info(snapshot_summary(vals), icon="💡")
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Yield Volatility by Regime
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("2 — Yield Volatility by Regime")
st.plotly_chart(yield_volatility_chart(df), use_container_width=True)
st.markdown(
    "<div class='caption-text'>"
    "Average monthly yield change and standard deviation (bps) for 2Y and 10Y Treasuries by regime. "
    "Volatility (σ) is always positive; avg change reflects directional bias within each regime."
    "</div>",
    unsafe_allow_html=True,
)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Duration Sensitivity by Regime
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("3 — Duration Sensitivity by Regime")

st.markdown(
    "<div class='disclaimer'>"
    "⚠ DV01 values are simplified proxies using modified duration approximation "
    "(modified duration ≈ maturity × 0.85). For educational illustration only — "
    "not suitable for risk management or trading decisions."
    "</div>",
    unsafe_allow_html=True,
)
st.write("")

st.plotly_chart(dv01_heatmap(df, current_yc), use_container_width=True)

st.markdown(
    "<div class='caption-text'>"
    "Average approximate price impact per month. Negative (red) = rates rose on average → adverse for holders. "
    "Positive (green) = rates fell → favourable. ▶ marks the current regime."
    "</div>",
    unsafe_allow_html=True,
)

st.write("")

# Directional bias table
st.markdown("#### Directional Bias — 10Y Yield")
st.markdown(
    "<div class='caption-text'>"
    "Regimes differ not just in volatility magnitude but in directional tendency. "
    "Below: % of months the 10Y yield rose vs fell, and average magnitude, by regime."
    "</div>",
    unsafe_allow_html=True,
)
st.write("")

bias_df = directional_bias_table(df)

def _highlight_current(row):
    if row.name == current_yc:
        return ["background-color: #1a3a5c; font-weight: bold"] * len(row)
    return [""] * len(row)

st.dataframe(
    bias_df.style.apply(_highlight_current, axis=1),
    use_container_width=True,
    hide_index=False,
)

st.write("")

# Adverse moves
st.markdown("#### Worst Rolling 12-Month Adverse Yield Moves by Regime")
st.plotly_chart(adverse_moves_chart(df), use_container_width=True)
st.markdown(
    "<div class='caption-text'>"
    "90th-percentile rolling 12-month cumulative yield increase (bps) across adverse episodes. "
    "Higher bars = larger historical rate rises experienced within that regime."
    "</div>",
    unsafe_allow_html=True,
)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Regime Duration Risk Summary
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("4 — Regime Duration Risk Summary")
st.markdown(section4_bullets(df, vals))
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Regime → Portfolio Playbook
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("5 — Regime → Portfolio Playbook")
st.markdown(
    "This section translates regime classification into historically observed portfolio risk characteristics."
)
st.markdown(
    "<div class='disclaimer'>"
    "⚠ Descriptive heuristic layer only — not investment advice or a trading signal. "
    "All labels are rule-based interpretations of historical regime statistics."
    "</div>",
    unsafe_allow_html=True,
)
st.write("")

playbook = get_portfolio_playbook(df, vals)

# Block A: Current regime
st.markdown(
    f"**Current Regime:** &nbsp;<span class='badge {_badge}'>{playbook['current_regime']}</span>",
    unsafe_allow_html=True,
)
st.write("")

# Block B: Historical Behavior
st.markdown("**Historical Behavior**")
b1, b2, b3 = st.columns(3)
b1.metric("Volatility",       playbook["volatility_label"])
b2.metric("Directional Bias", playbook["directional_bias"])
b3.metric("Regime Character", playbook["regime_character"])
st.write("")

# Block C: Portfolio Risk Interpretation
st.markdown("**Portfolio Risk Interpretation**")
p1, p2, p3 = st.columns(3)
p1.metric("Duration Risk",     playbook["duration_risk"])
p2.metric("Carry Environment", playbook["carry_environment"])
p3.metric("Convexity Value",   playbook["convexity_value"])
st.write("")

# Block D: Positioning Implications
st.markdown("**Positioning Implications**")
for bullet in playbook["bullets"]:
    st.markdown(f"- {bullet}")

st.markdown(
    "<div class='caption-text'>"
    "Labels derived from historical regime statistics. "
    "This framework is descriptive rather than predictive — intended to contextualize "
    "rate environments, not forecast them."
    "</div>",
    unsafe_allow_html=True,
)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.caption(
    "Data: Federal Reserve Economic Data (FRED) · "
    "Series: DGS2, DGS10, DGS30, FEDFUNDS · "
    "DV01 proxies are simplified approximations for educational purposes only · "
    "Built with Streamlit & Plotly"
)
