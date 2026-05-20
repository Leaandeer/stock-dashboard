"""Validation — is the composite actually predictive?"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from backtest.walk_forward import load as load_wf, walk_forward, zone_summary as wf_zone_summary
from tracking.performance import load_track_record, overall_stats, summary_by_bucket, update_returns
from utils.theme import AMBER, GREEN, MUTED, RED, TEXT, card_style, register_template

load_dotenv()
register_template()

st.set_page_config(page_title="Validation", page_icon="🔬", layout="wide")

st.markdown("## VALIDATION")
st.caption("Is the composite actually predictive — or are we tuning to noise?")

# =====================================================================
# Section A — live forward-performance track record (scanner composite)
# =====================================================================
st.markdown("### Forward performance · scanner picks")
st.caption(
    "Every scanner run snapshots its top 20 picks. Forward returns mature over "
    "1 / 5 / 20 trading days and are recomputed by `python run_performance.py`."
)

cols = st.columns([1, 1, 4])
with cols[0]:
    refresh_perf = st.button("Recompute returns", use_container_width=True)
with cols[1]:
    horizon = st.selectbox("Horizon", ["20d", "5d", "1d"], index=0)

if refresh_perf:
    with st.spinner("Fetching prices and recomputing forward returns..."):
        res = update_returns()
    st.success(f"{res['snapshots']} snapshots · {res['updated']} return rows refreshed")

tr = load_track_record()

if tr.empty:
    st.info(
        "No snapshots yet. Run the scanner on a few different trading days "
        "(`python run_scanner.py`), then `python run_performance.py` nightly. "
        "The track record grows from there — by design it starts empty."
    )
else:
    stats = overall_stats(tr, horizon=horizon)
    s = st.columns(4)
    s[0].markdown(
        f"<div style='{card_style()}'><div style='color:{MUTED};font-size:11px'>TOTAL PICKS</div>"
        f"<div style='font-size:30px;font-weight:700'>{stats.get('total_picks',0)}</div>"
        f"<div style='color:{MUTED};font-size:11px'>{stats.get('snapshot_days',0)} snapshot days</div></div>",
        unsafe_allow_html=True,
    )
    s[1].markdown(
        f"<div style='{card_style()}'><div style='color:{MUTED};font-size:11px'>MATURED ({horizon})</div>"
        f"<div style='font-size:30px;font-weight:700'>{stats.get('matured',0)}</div></div>",
        unsafe_allow_html=True,
    )
    avg_ex = stats.get("avg_excess_pct")
    ex_color = GREEN if (avg_ex or 0) > 0 else RED
    s[2].markdown(
        f"<div style='{card_style()}'><div style='color:{MUTED};font-size:11px'>AVG EXCESS vs SPY</div>"
        f"<div style='font-size:30px;font-weight:700;color:{ex_color}'>"
        f"{'n/a' if avg_ex is None else f'{avg_ex:+.2f}%'}</div></div>",
        unsafe_allow_html=True,
    )
    beat = stats.get("beat_spy_pct")
    s[3].markdown(
        f"<div style='{card_style()}'><div style='color:{MUTED};font-size:11px'>BEAT SPY</div>"
        f"<div style='font-size:30px;font-weight:700'>"
        f"{'n/a' if beat is None else f'{beat:.0f}%'}</div></div>",
        unsafe_allow_html=True,
    )

    bucket = summary_by_bucket(tr, horizon=horizon)
    if bucket.empty:
        st.warning(
            f"No picks have a matured {horizon} return yet — check back after "
            f"{horizon} trading days have elapsed since your first snapshot."
        )
    else:
        st.markdown("#### Forward return by composite-score bucket")
        st.caption(
            "If the composite is predictive, avg excess return rises monotonically "
            "across buckets. A flat or inverted pattern means the score is noise."
        )
        st.dataframe(bucket, use_container_width=True)

        ex_col = f"avg_{horizon}_excess_%"
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=[str(i) for i in bucket.index],
                y=bucket[ex_col],
                marker_color=[GREEN if v > 0 else RED for v in bucket[ex_col]],
            )
        )
        fig.add_hline(y=0, line=dict(color=MUTED, width=1))
        fig.update_layout(
            height=280, margin=dict(l=30, r=20, t=20, b=30),
            yaxis_title=f"avg {horizon} excess %", xaxis_title="composite bucket",
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Recent snapshots"):
        st.dataframe(tr.head(60), use_container_width=True)

st.markdown("---")

# =====================================================================
# Section B — walk-forward validation (macro composite)
# =====================================================================
st.markdown("### Walk-forward backtest · macro composite")
st.caption(
    "Weights are refit on a rolling 5-year window, then frozen and tested on the "
    "next 6 months. Out-of-sample only — the test windows span all available "
    "history (~2013-present: the 2018 vol spike, COVID, the 2022 bear and the "
    "2023-24 recovery)."
)

run_wf = st.button("Run / refresh walk-forward (≈10s)")
cached = load_wf()
if run_wf or cached is None:
    with st.spinner("Fetching 10y history and rolling the walk-forward windows..."):
        oos, segments = walk_forward()
else:
    oos, segments = cached

if oos is None or oos.empty:
    st.warning("Could not assemble enough history for a walk-forward run.")
else:
    st.markdown(
        f"<span style='color:{MUTED}'>{len(segments)} out-of-sample segments · "
        f"{len(oos)} OOS test days · {oos.index[0].date()} → {oos.index[-1].date()}</span>",
        unsafe_allow_html=True,
    )

    fitted = wf_zone_summary(oos, "zone_lag")
    equal = wf_zone_summary(oos, "zone_eq_lag")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Fitted weights — OOS zones")
        st.caption("Forward 1-day SPY return by zone, refit weights.")
        st.dataframe(fitted, use_container_width=True)
    with c2:
        st.markdown("#### Equal weights — OOS zones")
        st.caption("Same windows, naive 25/25/25/25 blend — the baseline to beat.")
        st.dataframe(equal, use_container_width=True)

    # verdict
    verdict = "Inconclusive."
    try:
        full = fitted.loc["FULL DEPLOY", "avg_fwd_1d_pct"]
        defn = fitted.loc["DEFENSIVE", "avg_fwd_1d_pct"]
        spread = full - defn
        if spread > 0.02:
            verdict = (
                f"Out-of-sample, FULL DEPLOY days beat DEFENSIVE days by "
                f"{spread:.3f}%/day in forward SPY return — the composite separates "
                f"regimes on data it was never fit on."
            )
        elif spread > 0:
            verdict = (
                f"Weak but positive: FULL DEPLOY edges DEFENSIVE by {spread:.3f}%/day "
                f"OOS. The signal is real but thin — don't over-trust it."
            )
        else:
            verdict = (
                f"FULL DEPLOY did NOT beat DEFENSIVE out-of-sample "
                f"({spread:.3f}%/day). The composite is likely overfit — treat the "
                f"zones as informational, not as a hard allocation rule."
            )
    except Exception:
        pass
    st.info(verdict)

    # OOS composite chart with zone bands
    st.markdown("#### Out-of-sample composite & SPY")
    zone_to_color = {"FULL DEPLOY": GREEN, "REDUCED": AMBER, "DEFENSIVE": RED}
    z = oos["zone_lag"].astype(str).fillna("")
    spans: list[tuple] = []
    if not z.empty:
        start_ts = z.index[0]
        cur = z.iloc[0]
        for ts, val in z.items():
            if val != cur:
                spans.append((start_ts, ts, cur))
                start_ts, cur = ts, val
        spans.append((start_ts, z.index[-1], cur))

    fig = go.Figure()
    for x0, x1, zone in spans:
        if zone in zone_to_color:
            fig.add_vrect(x0=x0, x1=x1, fillcolor=zone_to_color[zone], opacity=0.10, line_width=0, layer="below")
    spy_norm = oos["spy"] / oos["spy"].iloc[0] * 100.0
    fig.add_trace(go.Scatter(x=oos.index, y=spy_norm, mode="lines", name="SPY (norm)", line=dict(color=TEXT, width=2)))
    fig.add_trace(go.Scatter(x=oos.index, y=oos["composite"], mode="lines", name="Composite", line=dict(color="#8b5cf6", width=1.5), yaxis="y2"))
    fig.update_layout(
        height=380, margin=dict(l=30, r=40, t=20, b=30),
        yaxis=dict(title="SPY (indexed to 100)"),
        yaxis2=dict(title="composite", overlaying="y", side="right", range=[0, 100], showgrid=False),
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Refit weights over time")
    st.caption("How the regression re-weights the 4 signals each 6-month roll. Large swings = unstable fit.")
    seg_df = pd.DataFrame(
        [
            {
                "test window": f"{s['test_start']} → {s['test_end']}",
                "vix_level": s["weights"]["vix_level"],
                "term": s["weights"]["term"],
                "credit": s["weights"]["credit"],
                "put_call": s["weights"]["put_call"],
            }
            for s in segments
        ]
    )
    st.dataframe(seg_df, use_container_width=True)
