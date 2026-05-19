"""Signal 3 — % of S&P 500 above 200d SMA (expensive, daily cache)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from scanner.universe import sp500_tickers
from utils.data_fetch import fetch_many

CACHE = Path(__file__).resolve().parent.parent / "data" / "breadth_cache.json"


@dataclass
class SignalResult:
    score: float
    raw: dict


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def compute() -> SignalResult:
    if CACHE.exists():
        try:
            cached = json.loads(CACHE.read_text())
            if cached.get("date") == _today():
                pct = float(cached["pct_above_200"])
                score = _clamp((pct - 30) / (80 - 30) * 100.0)
                return SignalResult(
                    score=score,
                    raw={"pct_above_200": pct, "n": cached.get("n", 0), "label": f"{pct:.0f}% > 200d SMA (cached)"},
                )
        except Exception:
            pass
    tickers = sp500_tickers()
    if not tickers:
        return SignalResult(score=50.0, raw={"label": "universe unavailable"})
    data = fetch_many(tickers, period="1y", max_age_hours=18)
    above = 0
    counted = 0
    for t, df in data.items():
        if df is None or df.empty or "Close" not in df.columns:
            continue
        close = df["Close"].dropna()
        if len(close) < 200:
            continue
        sma200 = close.tail(220).rolling(200).mean().dropna()
        if sma200.empty:
            continue
        if float(close.iloc[-1]) > float(sma200.iloc[-1]):
            above += 1
        counted += 1
    if counted == 0:
        return SignalResult(score=50.0, raw={"label": "no breadth data"})
    pct = above / counted * 100.0
    try:
        CACHE.write_text(json.dumps({"date": _today(), "pct_above_200": pct, "n": counted}))
    except Exception:
        pass
    score = _clamp((pct - 30) / (80 - 30) * 100.0)
    return SignalResult(
        score=score,
        raw={"pct_above_200": pct, "n": counted, "label": f"{pct:.0f}% > 200d SMA"},
    )
