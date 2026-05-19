"""Signal 6 — momentum/value basket correlation (factor crowding)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scanner.universe import sp500_tickers
from utils.data_fetch import fetch_info, fetch_many

CACHE = Path(__file__).resolve().parent.parent / "data" / "crowding_cache.json"


@dataclass
class SignalResult:
    score: float
    raw: dict


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _basket_returns(prices: dict[str, pd.DataFrame], tickers: list[str]) -> pd.Series:
    rets = []
    for t in tickers:
        df = prices.get(t)
        if df is None or df.empty:
            continue
        c = df["Close"].dropna()
        if len(c) < 130:
            continue
        rets.append(c.pct_change().rename(t))
    if not rets:
        return pd.Series(dtype=float)
    return pd.concat(rets, axis=1).mean(axis=1)


def compute() -> SignalResult:
    if CACHE.exists():
        try:
            cached = json.loads(CACHE.read_text())
            if cached.get("date") == _today():
                corr = float(cached["correlation"])
                score = _clamp((corr - (-0.8)) / (0.3 - (-0.8)) * 100.0)
                return SignalResult(score=score, raw={"correlation": corr, "label": f"M/V corr {corr:+.2f} (cached)"})
        except Exception:
            pass

    tickers = sp500_tickers()
    if not tickers:
        return SignalResult(score=50.0, raw={"label": "universe unavailable"})
    prices = fetch_many(tickers, period="6mo", max_age_hours=18)

    # momentum basket: 3-month return
    mom = {}
    for t, df in prices.items():
        if df is None or df.empty:
            continue
        c = df["Close"].dropna()
        if len(c) < 65:
            continue
        mom[t] = float(c.iloc[-1] / c.iloc[-63] - 1.0)
    if len(mom) < 100:
        return SignalResult(score=50.0, raw={"label": "insufficient momentum data"})
    mom_sorted = sorted(mom.items(), key=lambda kv: kv[1], reverse=True)
    mom_top = [t for t, _ in mom_sorted[:50]]
    mom_bot = [t for t, _ in mom_sorted[-50:]]

    # value basket: inverse P/E — use fetch_info on top 200 momentum-ranked names only to save calls
    sample = [t for t, _ in mom_sorted[:300]]
    val_scores: dict[str, float] = {}
    for t in sample:
        info = fetch_info(t)
        pe = info.get("trailingPE") or info.get("forwardPE")
        if pe and pe > 0:
            val_scores[t] = 1.0 / pe
    if len(val_scores) < 60:
        # fall back to a neutral score if value data sparse
        return SignalResult(score=50.0, raw={"label": "value data sparse"})
    val_sorted = sorted(val_scores.items(), key=lambda kv: kv[1], reverse=True)
    val_top = [t for t, _ in val_sorted[:30]]
    val_bot = [t for t, _ in val_sorted[-30:]]

    mom_ret = _basket_returns(prices, mom_top) - _basket_returns(prices, mom_bot)
    val_ret = _basket_returns(prices, val_top) - _basket_returns(prices, val_bot)
    joined = pd.concat([mom_ret.rename("m"), val_ret.rename("v")], axis=1).dropna()
    if len(joined) < 60:
        return SignalResult(score=50.0, raw={"label": "insufficient overlap"})
    corr = float(joined.tail(60).corr().loc["m", "v"])
    try:
        CACHE.write_text(json.dumps({"date": _today(), "correlation": corr}))
    except Exception:
        pass
    score = _clamp((corr - (-0.8)) / (0.3 - (-0.8)) * 100.0)
    return SignalResult(score=score, raw={"correlation": corr, "label": f"M/V corr {corr:+.2f}"})
