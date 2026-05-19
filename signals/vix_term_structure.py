"""Signal 2 — VIX term structure: ^VIX / ^VIX3M."""
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
    vix = fetch_history("^VIX", period="3mo")
    vix3m = fetch_history("^VIX3M", period="3mo")
    if vix.empty or vix3m.empty:
        return SignalResult(score=50.0, raw={"label": "term-structure data unavailable"})
    v = float(vix["Close"].dropna().iloc[-1])
    v3 = float(vix3m["Close"].dropna().iloc[-1])
    if v3 == 0:
        return SignalResult(score=50.0, raw={"label": "term-structure denominator zero"})
    ratio = v / v3
    # linear map 0.85 → 100, 1.15 → 0
    score = _clamp((1.15 - ratio) / (1.15 - 0.85) * 100.0)
    state = "contango" if ratio < 1.0 else "backwardation"
    return SignalResult(
        score=score,
        raw={
            "ratio": ratio,
            "vix": v,
            "vix3m": v3,
            "state": state,
            "label": f"VIX/VIX3M {ratio:.2f} ({state})",
        },
    )
