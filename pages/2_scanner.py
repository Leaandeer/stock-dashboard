"""Stock scanner — Layer 2."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from plotly.subplots import make_subplots

from scanner.constants import FACTOR_COLS, FACTOR_DISPLAY
from scanner.ranker import load as load_scanner, run as run_scanner
from signals.composite import compute as compute_macro, load as load_macro, save as save_macro
from utils.data_fetch import fetch_history, utc_now_str
from utils.theme import AMBER, BG, CARD, GREEN, MUTED, PURPLE, RED, TEXT, card_style, pill, register_template, score_color, signal_bar, zone_color

load_dotenv()
register_template()

st.set_page_config(page_title="Scanner", page_icon="🟢", layout="wide")

st.markdown("## STOCK SCANNER")
st.caption("Layer 2 — Multi-factor ranking (6 factors, equal weight, sector-neutral)")

# ---- macro context banner (informational only — not a gate, Phase 2 Part C) ----
macro = load_macro()
if macro is None:
    with st.spinner("First run — computing macro context..."):
        macro = compute_macro()
        save_macro(macro)

zc = zone_color(macro.zone)
top = st.columns([2, 1])
with top[0]:
    st.markdown(
        f"<div style='{card_style()}'>"
        f"<div style='display:flex;gap:14px;align-items:center;'>"
        f"<div><div style='color:{MUTED};font-size:11px;letter-spacing:0.08em'>MACRO CONTEXT</div>"
        f"<div style='font-size:36px;font-weight:700;color:{zc};margin-top:2px'>{macro.score:.0f}</div></div>"
        f"<div>{pill(macro.zone, zc)}</div>"
        f"<div style='color:{MUTED};font-size:12px;max-width:240px'>"
        f"Informational only — the macro gate failed walk-forward validation, "
        f"so it no longer gates or resizes the scanner.</div>"
        f"<div style='margin-left:auto;color:{MUTED};font-size:12px;'>updated {macro.timestamp}</div>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
with top[1]:
    refresh = st.button("Re-run scanner", use_container_width=True)
    st.caption("First run fetches fundamentals — can take several minutes.")

if refresh or load_scanner() is None:
    with st.spinner("Scanning the universe — computing 6 factors..."):
        try:
            results = run_scanner(macro)
        except Exception as e:
            st.error(f"Scan halted: {e}")
            st.stop()
else:
    results = load_scanner()

threshold = results.get("threshold", 65)
candidates = results.get("candidates", [])
scored = results.get("scored_universe", 0)

st.markdown(
    f"<div style='color:{MUTED};font-size:13px;margin-top:8px;'>"
    f"Scanner found <b style='color:{TEXT}'>{len(candidates)}</b> candidates at or above composite "
    f"<b style='color:{TEXT}'>{threshold:.0f}</b>, out of <b style='color:{TEXT}'>{scored}</b> "
    f"fully-scored names. Macro context: {macro.zone} — informational only."
    f"</div>",
    unsafe_allow_html=True,
)

# ---- top candidates table ----
st.markdown("### TOP CANDIDATES")

WIDTHS = [0.4, 0.9, 0.8, 0.9, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2]

if not candidates:
    st.info(
        "No candidates cleared the threshold. With 6 factors and "
        "drop-not-impute NaN handling, the fully-scored pool can be small — "
        "see the count above."
    )
else:
    header_cols = st.columns(WIDTHS)
    headers = ["#", "TICKER", "PRICE", "COMPOSITE"] + [FACTOR_DISPLAY[f] for f in FACTOR_COLS]
    for c, h in zip(header_cols, headers):
        c.markdown(f"<div style='color:{MUTED};font-size:11px;letter-spacing:0.06em'>{h}</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1px;background:#1f2937;margin:4px 0 8px 0;'></div>", unsafe_allow_html=True)

    for c in candidates:
        cols = st.columns(WIDTHS)
        comp_color = score_color(c["composite"])
        cols[0].markdown(f"<div style='color:{MUTED}'>{c['rank']}</div>", unsafe_allow_html=True)
        cols[1].markdown(f"<b style='color:{TEXT}'>{c['ticker']}</b>", unsafe_allow_html=True)
        cols[2].markdown(f"<span style='color:{TEXT}'>${c['price']:.2f}</span>", unsafe_allow_html=True)
        cols[3].markdown(f"<span style='color:{comp_color};font-weight:700'>{c['composite']:.0f}</span>", unsafe_allow_html=True)
        for i, f in enumerate(FACTOR_COLS):
            cols[4 + i].markdown(signal_bar(c[f], width_px=95), unsafe_allow_html=True)

# ---- inspector ----
st.markdown("---")
st.markdown("### Inspect a candidate")

if candidates:
    options = [f"{c['ticker']} · {c['composite']:.0f}" for c in candidates]
    sel = st.selectbox("Pick a ticker", options=options, index=0)
    pick = candidates[options.index(sel)]
    info_cols = st.columns([1, 1, 1, 1, 2])
    info_cols[0].markdown(f"<div style='{card_style()}'><div style='color:{MUTED};font-size:11px'>TICKER</div><div style='font-size:24px;font-weight:700'>{pick['ticker']}</div></div>", unsafe_allow_html=True)
    info_cols[1].markdown(f"<div style='{card_style()}'><div style='color:{MUTED};font-size:11px'>PRICE</div><div style='font-size:24px;font-weight:700'>${pick['price']:.2f}</div></div>", unsafe_allow_html=True)
    cc = score_color(pick["composite"])
    info_cols[2].markdown(f"<div style='{card_style()}'><div style='color:{MUTED};font-size:11px'>COMPOSITE</div><div style='font-size:24px;font-weight:700;color:{cc}'>{pick['composite']:.0f}</div></div>", unsafe_allow_html=True)
    info_cols[3].markdown(f"<div style='{card_style()}'><div style='color:{MUTED};font-size:11px'>SECTOR</div><div style='font-size:14px;font-weight:700;margin-top:8px'>{pick.get('sector','Unknown')}</div></div>", unsafe_allow_html=True)

    df = fetch_history(pick["ticker"], period="1y")
    if not df.empty:
        df = df.tail(170)
        ema10 = df["Close"].ewm(span=10, adjust=False).mean()
        ema50 = df["Close"].ewm(span=50, adjust=False).mean()
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.75, 0.25])
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close", line=dict(color=TEXT, width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=ema10, mode="lines", name="EMA 10", line=dict(color=PURPLE, width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=ema50, mode="lines", name="EMA 50", line=dict(color="#f97316", width=1.5)), row=1, col=1)
        colors = [GREEN if (cl >= op) else RED for op, cl in zip(df["Open"], df["Close"])]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=colors, opacity=0.6), row=2, col=1)
        fig.update_layout(height=520, margin=dict(l=30, r=20, t=20, b=20), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
