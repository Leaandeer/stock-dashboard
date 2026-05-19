"""Signal 1 — VIX level percentile-ranked against 1y history."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from utils.data_fetch import fetch_history


@dataclass
class SignalResult:
    score: float
    raw: dict


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def compute() -> SignalResult:
    df = fetch_history("^VIX", period="2y")
    if df is None or df.empty:
        return SignalResult(score=50.0, raw={"label": "VIX unavailable"})
    close = df["Close"].dropna()
    current = float(close.iloc[-1])
    one_year = close.tail(252)
    pct = float((one_year <= current).mean() * 100.0)
    # low VIX = high score → invert the percentile
    base = 100.0 - pct
    if current < 15:
        base += 5
    if current > 30:
        base -= 10
    score = _clamp(base)
    return SignalResult(
        score=score,
        raw={
            "current": current,
            "percentile_1y": pct,
            "label": f"VIX {current:.1f} ({pct:.0f}th pct)",
        },
    )
