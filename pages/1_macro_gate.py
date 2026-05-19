"""Macro deployment gate — Layer 1."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from backtest.deployment_backtest import compute as bt_compute
from backtest.deployment_backtest import zone_summary
from signals.composite import LABELS, compute as compute_macro, load as load_macro, save as save_macro
from utils.data_fetch import utc_now_str
from utils.theme import AMBER, BG, CARD, GREEN, GRID, MUTED, RED, TEXT, card_style, pill, register_template, score_color, signal_bar, zone_color

load_dotenv()
register_template()

st.set_page_config(page_title="Macro Gate", page_icon="🟢", layout="wide")

st.markdown("## MACRO DEPLOYMENT GATE")
st.caption("Layer 1 — Should we be deploying capital, and how aggressively?")

cols = st.columns([1, 4])
with cols[0]:
    refresh = st.button("Refresh macro signals", use_container_width=True)
with cols[1]:
    st.markdown(f"<div style='color:{MUTED};padding-top:8px'>data refreshed via yfinance · displayed at {utc_now_str()}</div>", unsafe_allow_html=True)

if refresh:
    with st.spinner("Recomputing 6 macro signals..."):
        macro = compute_macro()
        save_macro(macro)
else:
    macro = load_macro()
    if macro is None:
        with st.spinner("First run — computing 6 macro signals..."):
            macro = compute_macro()
            save_macro(macro)

# ---- top row ----
top = st.columns(3)
zc = zone_color(macro.zone)

with top[0]:
    st.markdown(
        f"<div style='{card_style()}'>"
        f"<div style='color:{MUTED};font-size:11px;letter-spacing:0.1em;'>DEPLOYMENT SCORE</div>"
        f"<div style='font-size:72px;font-weight:700;color:{zc};line-height:1;margin-top:8px;'>{macro.score:.0f}</div>"
        f"<div style='color:{MUTED};margin-top:6px;'>/ 100</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

with top[1]:
    st.markdown(
        f"<div style='{card_style()}'>"
        f"<div style='color:{MUTED};font-size:11px;letter-spacing:0.1em;'>POSTURE</div>"
        f"<div style='margin-top:10px;'>{pill(macro.zone, zc)}</div>"
        f"<div style='font-size:22px;font-weight:700;margin-top:14px;color:{TEXT};'>Deploy at {macro.sizing_pct}% of normal sizing</div>"
        f"<div style='color:{MUTED};margin-top:6px;font-size:13px;'>Composite of 6 macro signals, weighted blend.</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

with top[2]:
    mix_html = [
        f"<div style='{card_style()}'>",
        f"<div style='color:{MUTED};font-size:11px;letter-spacing:0.1em;margin-bottom:10px;'>SIGNAL MIX</div>",
    ]
    for key, label in LABELS.items():
        s = macro.signals.get(key, {})
        v = float(s.get("score", 50.0))
        mix_html.append(
            f"<div style='display:flex;justify-content:space-between;align-items:center;margin:6px 0;'>"
            f"<span style='color:{TEXT};font-size:13px;'>{label}</span>"
            f"<span>{signal_bar(v, width_px=180)}</span>"
            f"</div>"
        )
    mix_html.append("</div>")
    st.markdown("".join(mix_html), unsafe_allow_html=True)

# ---- detail cards ----
st.markdown("### Signal detail")
detail_cols = st.columns(3)
for i, (key, label) in enumerate(LABELS.items()):
    sig = macro.signals.get(key, {})
    score = float(sig.get("score", 50.0))
    sub = sig.get("raw", {}).get("label", "")
    c = score_color(score)
    with detail_cols[i % 3]:
        st.markdown(
            f"<div style='{card_style()}margin-bottom:14px;'>"
            f"<div style='color:{MUTED};font-size:11px;letter-spacing:0.08em;'>{label.upper()}</div>"
            f"<div style='display:flex;align-items:baseline;gap:8px;margin-top:6px;'>"
            f"<span style='font-size:28px;font-weight:700;color:{c}'>{score:.0f}</span>"
            f"<span style='color:{MUTED};font-size:12px;'>/ 100</span></div>"
            f"<div style='color:{TEXT};margin-top:6px;font-size:13px;'>{sub}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

# ---- backtest ----
st.markdown("### SPY · ZONE OVERLAY")
st.caption("Backtest uses a 4-signal re-weighted composite (excludes breadth + crowding — too costly to recompute daily).")
with st.spinner("Loading historical composite..."):
    bt = bt_compute(period="3y")

if not bt.empty:
    bt_2y = bt.tail(504)
    fig = go.Figure()

    # zone bands behind SPY
    zone_to_color = {"FULL DEPLOY": GREEN, "REDUCED": AMBER, "DEFENSIVE": RED}
    z = bt_2y["zone_lag"].astype(str).fillna("")
    spans: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    if not z.empty:
        start = z.index[0]
        cur = z.iloc[0]
        for ts, val in z.items():
            if val != cur:
                spans.append((start, ts, cur))
                start = ts
                cur = val
        spans.append((start, z.index[-1], cur))
    for x0, x1, zone in spans:
        if zone not in zone_to_color:
            continue
        fig.add_vrect(
            x0=x0, x1=x1, fillcolor=zone_to_color[zone], opacity=0.10,
            line_width=0, layer="below",
        )

    fig.add_trace(go.Scatter(x=bt_2y.index, y=bt_2y["spy"], mode="lines", name="SPY", line=dict(color=TEXT, width=2)))
    fig.update_layout(height=380, margin=dict(l=30, r=20, t=10, b=30), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### COMPOSITE HISTORY")
    hist = go.Figure()
    hist.add_hline(y=70, line=dict(color=GREEN, dash="dot", width=1))
    hist.add_hline(y=40, line=dict(color=RED, dash="dot", width=1))
    hist.add_trace(go.Scatter(x=bt_2y.index, y=bt_2y["composite"], mode="lines", line=dict(color="#8b5cf6", width=2)))
    hist.update_layout(height=260, margin=dict(l=30, r=20, t=10, b=30), yaxis=dict(range=[0, 100]))
    st.plotly_chart(hist, use_container_width=True)

    st.markdown("### Zone vs. forward 1-day SPY return")
    summary = zone_summary(bt_2y)
    if not summary.empty:
        st.dataframe(summary, use_container_width=True)
else:
    st.warning("Could not load historical data for backtest.")
