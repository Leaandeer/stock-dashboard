"""Stock Dashboard — entry point with sidebar nav and a top-right deploy badge."""
from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from signals.composite import load as load_macro
from utils.theme import card_style, pill, register_template, zone_color

load_dotenv()
register_template()

st.set_page_config(
    page_title="Stock Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem;}
      [data-testid="stSidebar"] {background-color: #0b0e17;}
      [data-testid="stSidebarNav"] li div a span {font-family: ui-monospace, Menlo, monospace;}
      .deploy-badge {
        position: fixed; top: 14px; right: 24px; z-index: 9999;
        display: flex; gap: 10px; align-items: center;
        font-family: ui-monospace, Menlo, monospace; font-size: 12px;
      }
      .stPlotlyChart {background: transparent !important;}
    </style>
    """,
    unsafe_allow_html=True,
)


def deploy_badge() -> None:
    macro = load_macro()
    if macro is None:
        st.markdown(
            '<div class="deploy-badge"><span style="color:#6b7280">no macro state — run `python run_macro_gate.py`</span></div>',
            unsafe_allow_html=True,
        )
        return
    color = zone_color(macro.zone)
    st.markdown(
        f'<div class="deploy-badge">'
        f'<span style="color:#e6e8ee">Deploy</span>'
        f'{pill(f"{macro.zone} · {macro.score:.0f}", color)}'
        f'<span style="color:#6b7280">· {macro.timestamp}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


deploy_badge()

st.markdown("## Stock Analysis Dashboard")
st.markdown(
    "<span style='color:#6b7280'>Three layers — macro gate, quantitative scanner, Claude fundamental analyst.</span>",
    unsafe_allow_html=True,
)

cols = st.columns(3)
items = [
    ("MACRO GATE", "Layer 1", "Should we be deploying capital, and how aggressively?", "pages/1_macro_gate.py"),
    ("SCANNER", "Layer 2", "If yes, which S&P 500 names look best by the numbers?", "pages/2_scanner.py"),
    ("ANALYST", "Layer 3", "Do the fundamentals back up what the numbers say?", "pages/3_analyst.py"),
]
for col, (title, layer, desc, path) in zip(cols, items):
    with col:
        st.markdown(
            f"<div style='{card_style()}'>"
            f"<div style='color:#6b7280;font-size:11px;letter-spacing:0.1em;'>{layer}</div>"
            f"<div style='font-size:20px;font-weight:700;margin-top:4px;'>{title}</div>"
            f"<div style='color:#9ca3af;margin-top:8px;font-size:13px;'>{desc}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.markdown("---")
st.markdown(
    "Open a layer from the sidebar. Refresh data from the CLI: "
    "`python run_macro_gate.py`, then `python run_scanner.py`, then "
    "`python run_analysis.py --scan-and-analyze`."
)
