"""Color tokens, plotly template, signal-bar helper."""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

BG = "#0b0e17"
CARD = "#11151f"
TEXT = "#e6e8ee"
MUTED = "#6b7280"
GREEN = "#10b981"
AMBER = "#f59e0b"
RED = "#ef4444"
PURPLE = "#8b5cf6"
ORANGE = "#f97316"
GRID = "#1f2937"


def score_color(value: float) -> str:
    if value is None:
        return MUTED
    if value >= 70:
        return GREEN
    if value >= 40:
        return AMBER
    return RED


def zone_color(zone: str) -> str:
    return {"FULL DEPLOY": GREEN, "REDUCED": AMBER, "DEFENSIVE": RED}.get(zone, MUTED)


def register_template() -> None:
    tpl = go.layout.Template()
    tpl.layout = go.Layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family="ui-monospace, SFMono-Regular, Menlo, monospace", color=TEXT, size=12),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
        colorway=[GREEN, AMBER, RED, PURPLE, ORANGE, TEXT, MUTED],
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    pio.templates["stockdash"] = tpl
    pio.templates.default = "stockdash"


def signal_bar(value: float, width_px: int = 220) -> str:
    """Return an inline HTML horizontal gauge."""
    v = max(0.0, min(100.0, float(value)))
    color = score_color(v)
    return (
        f'<div style="display:inline-block;width:{width_px}px;height:10px;'
        f'background:{GRID};border-radius:6px;overflow:hidden;vertical-align:middle;">'
        f'<div style="width:{v:.1f}%;height:100%;background:{color};"></div></div>'
        f'<span style="margin-left:8px;color:{color};font-weight:600;">{v:.0f}</span>'
    )


def card_style(border: str | None = None) -> str:
    b = f"border:1px solid {border};" if border else "border:1px solid #1f2937;"
    return (
        f"background:{CARD};{b}border-radius:12px;padding:18px 20px;"
        "box-shadow:0 1px 0 rgba(255,255,255,0.02) inset;"
    )


def pill(text: str, color: str) -> str:
    return (
        f'<span style="display:inline-block;padding:4px 10px;border-radius:999px;'
        f'background:{color}22;color:{color};border:1px solid {color}55;'
        f'font-weight:600;font-size:12px;letter-spacing:0.04em;">{text}</span>'
    )
