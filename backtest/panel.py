"""Shared daily signal panel for backtests.

Builds a long-history DataFrame of the four time-series-derivable macro
signals (each 0-100) plus SPY close. Breadth and crowding are excluded — they
require a 500-ticker recomputation per day, which is intractable for a 10-year
daily backtest.
"""
from __future__ import annotations

import pandas as pd

from utils.data_fetch import fetch_history

SIGNALS = ["vix_level", "term", "credit", "put_call"]


def _clamp(s: pd.Series, lo: float = 0.0, hi: float = 100.0) -> pd.Series:
    return s.clip(lower=lo, upper=hi)


def _vix_level_score(vix_close: pd.Series) -> pd.Series:
    pct = vix_close.rolling(252).apply(lambda w: (w <= w.iloc[-1]).mean() * 100.0, raw=False)
    base = 100.0 - pct
    bonus = (vix_close < 15).astype(float) * 5.0
    penalty = (vix_close > 30).astype(float) * 10.0
    return _clamp(base + bonus - penalty)


def _term_structure_score(vix: pd.Series, vix3m: pd.Series) -> pd.Series:
    ratio = vix / vix3m
    return _clamp((1.15 - ratio) / (1.15 - 0.85) * 100.0)


def _credit_score(hyg: pd.Series, tlt: pd.Series) -> pd.Series:
    ratio = hyg / tlt
    mu = ratio.rolling(252).mean()
    sd = ratio.rolling(252).std()
    z = (ratio - mu) / sd
    return _clamp((2.0 - z) / 4.0 * 100.0)


def _put_call_score(vix: pd.Series) -> pd.Series:
    roc = (vix / vix.shift(20) - 1.0) * 100.0
    return _clamp((50.0 - roc) / (50.0 - (-30.0)) * 100.0)


def fetch_panel(period: str = "10y") -> pd.DataFrame:
    """Daily panel of the 4 signal scores + SPY. Warmup year is dropped."""
    vix = fetch_history("^VIX", period=period)
    vix3m = fetch_history("^VIX3M", period=period)
    hyg = fetch_history("HYG", period=period)
    tlt = fetch_history("TLT", period=period)
    spy = fetch_history("SPY", period=period)
    frames = [vix, vix3m, hyg, tlt, spy]
    if any(d is None or d.empty for d in frames):
        return pd.DataFrame()

    vix = vix["Close"].dropna()
    vix3m = vix3m["Close"].dropna()
    hyg = hyg["Close"].dropna()
    tlt = tlt["Close"].dropna()
    spy = spy["Close"].dropna()

    idx = vix.index
    for s in (vix3m, hyg, tlt, spy):
        idx = idx.intersection(s.index)
    if len(idx) < 300:
        return pd.DataFrame()
    vix = vix.reindex(idx)
    vix3m = vix3m.reindex(idx)
    hyg = hyg.reindex(idx)
    tlt = tlt.reindex(idx)
    spy = spy.reindex(idx)

    df = pd.DataFrame(
        {
            "vix_level": _vix_level_score(vix),
            "term": _term_structure_score(vix, vix3m),
            "credit": _credit_score(hyg, tlt),
            "put_call": _put_call_score(vix),
            "spy": spy,
        }
    )
    return df.dropna()
