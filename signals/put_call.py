"""Signal 5 — sentiment proxy via 20-day VIX ROC."""
from __future__ import annotations

from dataclasses import dataclass

from utils.data_fetch import fetch_history


@dataclass
class SignalResult:
    score: float
    raw: dict


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def compute() -> SignalResult:
    df = fetch_history("^VIX", period="3mo")
    if df.empty:
        return SignalResult(score=50.0, raw={"label": "VIX unavailable"})
    close = df["Close"].dropna()
    if len(close) < 25:
        return SignalResult(score=50.0, raw={"label": "insufficient VIX history"})
    roc = (close.iloc[-1] / close.iloc[-21] - 1.0) * 100.0
    # ROC -30% → 100, ROC +50% → 0
    score = _clamp((50.0 - roc) / (50.0 - (-30.0)) * 100.0)
    return SignalResult(
        score=score,
        raw={"vix_roc_20d_pct": float(roc), "label": f"VIX 20d ROC {roc:+.1f}%"},
    )
