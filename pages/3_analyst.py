"""Claude analyst — Layer 3."""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from analyst.blender import load as load_analyst, run as run_blender
from scanner.ranker import load as load_scanner
from signals.composite import load as load_macro
from utils.theme import GREEN, MUTED, RED, TEXT, card_style, pill, register_template, score_color, signal_bar, zone_color

load_dotenv()
register_template()

st.set_page_config(page_title="Analyst", page_icon="🟣", layout="wide")

st.markdown("## CLAUDE ANALYST")
st.caption("Layer 3 — Fundamental quality scoring")

macro = load_macro()
scanner = load_scanner()

top = st.columns([3, 1])
with top[0]:
    if macro:
        zc = zone_color(macro.zone)
        st.markdown(
            f"<div style='{card_style()}'>"
            f"<div style='display:flex;gap:14px;align-items:center;'>"
            f"<div><div style='color:{MUTED};font-size:11px;letter-spacing:0.08em'>MACRO</div>"
            f"<div style='font-size:28px;font-weight:700;color:{zc}'>{macro.score:.0f}</div></div>"
            f"<div>{pill(macro.zone, zc)}</div>"
            f"<div style='margin-left:auto;color:{MUTED};font-size:12px'>Model: {os.environ.get('ANTHROPIC_MODEL','claude-sonnet-4-5')}</div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )
    else:
        st.warning("Run the macro gate first.")

with top[1]:
    force = st.checkbox("Force refresh (bypass cache)", value=False)
    run = st.button("Run analyst", use_container_width=True)

if not os.environ.get("ANTHROPIC_API_KEY"):
    st.warning("ANTHROPIC_API_KEY is not set. Add it to .env locally or to Streamlit secrets when deployed.")

if scanner is None or not scanner.get("candidates"):
    st.info("Run the scanner first — no candidates to analyze.")
    st.stop()

if run:
    with st.spinner(f"Running Claude on {len(scanner['candidates'])} candidates..."):
        results = run_blender(scanner, force=force)
else:
    results = load_analyst() or {}

rows = results.get("rows", [])
if not rows:
    st.info("No analyst results yet — click 'Run analyst' to score the scanner output.")
    st.stop()

st.caption(f"Generated at {results.get('timestamp', '')} · {sum(1 for r in rows if r['cached'])} of {len(rows)} from cache")

# ---- blended table ----
st.markdown("### Blended ranks")
hdr = st.columns([0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0])
for c, h in zip(hdr, ["#", "TICKER", "PRICE", "QUANT (/10)", "CLAUDE (/10)", "BLENDED", "Δ RANK", "SUMMARY"]):
    c.markdown(f"<div style='color:{MUTED};font-size:11px;letter-spacing:0.08em'>{h}</div>", unsafe_allow_html=True)
st.markdown("<div style='height:1px;background:#1f2937;margin:4px 0 8px 0;'></div>", unsafe_allow_html=True)

for r in rows:
    delta = r["rank_delta"]
    glow = ""
    if delta >= 3:
        glow = f"border:1px solid {GREEN}55;background:linear-gradient(180deg, {GREEN}11, transparent);"
    elif delta <= -3:
        glow = f"border:1px solid {RED}55;background:linear-gradient(180deg, {RED}11, transparent);"
    cols = st.columns([0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0])
    arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "·")
    arrow_color = GREEN if delta > 0 else (RED if delta < 0 else MUTED)
    summary_html = (r["summary"] or "").replace("<", "&lt;")
    style = f"padding:8px;border-radius:8px;{glow}" if glow else "padding:8px;"
    cols[0].markdown(f"<div style='{style}'><span style='color:{MUTED}'>{r['new_rank']}</span></div>", unsafe_allow_html=True)
    cols[1].markdown(f"<div style='{style}'><b>{r['ticker']}</b></div>", unsafe_allow_html=True)
    cols[2].markdown(f"<div style='{style}'>${r['price']:.2f}</div>", unsafe_allow_html=True)
    cols[3].markdown(f"<div style='{style}'>{r['quant_score_10']:.1f}</div>", unsafe_allow_html=True)
    cols[4].markdown(f"<div style='{style}'>{r['claude_score']:.1f}</div>", unsafe_allow_html=True)
    cols[5].markdown(f"<div style='{style}'><b>{r['blended']:.2f}</b></div>", unsafe_allow_html=True)
    cols[6].markdown(f"<div style='{style}'><span style='color:{arrow_color};font-weight:700'>{arrow} {abs(delta)}</span></div>", unsafe_allow_html=True)
    cols[7].markdown(f"<div style='{style};color:{TEXT};font-size:12px'>{summary_html}</div>", unsafe_allow_html=True)

# ---- expanders ----
st.markdown("### Per-ticker detail")
for r in rows:
    with st.expander(f"{r['ticker']}  ·  blended {r['blended']:.2f}  ·  quant {r['quant_score_10']:.1f}  ·  claude {r['claude_score']:.1f}"):
        st.write(r["summary"])
        st.markdown("**Sub-scores**")
        ss = r["sub_scores"] or {}
        labels = {
            "earnings_quality": "Earnings Quality",
            "growth_trajectory": "Growth Trajectory",
            "balance_sheet_health": "Balance Sheet Health",
            "margin_trends": "Margin Trends",
            "red_flags": "Red Flags",
        }
        for k, label in labels.items():
            val = float(ss.get(k, 5)) * 10.0
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;align-items:center;margin:4px 0;'>"
                f"<span style='color:{TEXT};font-size:13px'>{label}</span>"
                f"<span>{signal_bar(val, width_px=200)}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.markdown("**Key observations**")
        for obs in r.get("key_observations", []):
            st.markdown(f"- {obs}")
        st.markdown("**Financials (4 quarters, most recent first)**")
        tbl = r.get("financial_table", {})
        if tbl:
            df = pd.DataFrame(tbl).T
            df.columns = [f"Q{-i}" if i > 0 else "Latest" for i in range(len(df.columns))]
            st.dataframe(df, use_container_width=True)
