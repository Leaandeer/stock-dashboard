"""Signal 4 — HYG/TLT spread proxy, z-scored vs 1y."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.data_fetch import fetch_history


@dataclass
class SignalResult:
    score: float
    raw: dict


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def compute() -> SignalResult:
    hyg = fetch_history("HYG", period="2y")
    tlt = fetch_history("TLT", period="2y")
    if hyg.empty or tlt.empty:
        return SignalResult(score=50.0, raw={"label": "HYG/TLT unavailable"})
    h = hyg["Close"].dropna()
    t = tlt["Close"].dropna()
    df = h.to_frame("h").join(t.to_frame("t"), how="inner").dropna()
    if len(df) < 60:
        return SignalResult(score=50.0, raw={"label": "insufficient credit data"})
    ratio = (df["h"] / df["t"]).tail(252)
    z = (ratio.iloc[-1] - ratio.mean()) / (ratio.std() or 1.0)
    # tight spreads (high HYG/TLT, z high) → bullish → high score
    # plan says z=-2 → 100, z=+2 → 0 (spreads wide = high yields = low ratio?)
    # We interpret per plan: linear map z=-2 → 100, z=+2 → 0
    score = _clamp((2.0 - z) / 4.0 * 100.0)
    return SignalResult(
        score=score,
        raw={
            "hyg_tlt_ratio": float(ratio.iloc[-1]),
            "z_score": float(z),
            "label": f"HYG/TLT z={z:+.2f}",
        },
    )
