"""Recompute composite daily over last 2y from time-series-derivable signals.

Breadth and crowding require per-day S&P 500 recomputation which is too
expensive for an interactive backtest; we re-weight the remaining four signals
proportionally and call out the simplification on the page.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from utils.data_fetch import fetch_history

CACHE = Path(__file__).resolve().parent.parent / "data" / "composite_history.parquet"


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


def compute(period: str = "3y") -> pd.DataFrame:
    """Daily composite + zone over the trailing period."""
    vix = fetch_history("^VIX", period=period)["Close"].dropna()
    vix3m = fetch_history("^VIX3M", period=period)["Close"].dropna()
    hyg = fetch_history("HYG", period=period)["Close"].dropna()
    tlt = fetch_history("TLT", period=period)["Close"].dropna()
    spy = fetch_history("SPY", period=period)["Close"].dropna()

    idx = vix.index.intersection(vix3m.index).intersection(hyg.index).intersection(tlt.index).intersection(spy.index)
    vix = vix.reindex(idx)
    vix3m = vix3m.reindex(idx)
    hyg = hyg.reindex(idx)
    tlt = tlt.reindex(idx)
    spy = spy.reindex(idx)

    s_vix = _vix_level_score(vix)
    s_term = _term_structure_score(vix, vix3m)
    s_credit = _credit_score(hyg, tlt)
    s_pc = _put_call_score(vix)

    # original weights: 0.25 / 0.20 / 0.20 / 0.15 / 0.10 / 0.10
    # for backtest we re-normalize over the 4 we can compute daily (0.25 / 0.20 / 0.15 / 0.10)
    w = {"vix": 0.25, "term": 0.20, "credit": 0.15, "pc": 0.10}
    wsum = sum(w.values())
    composite = (
        s_vix * w["vix"] + s_term * w["term"] + s_credit * w["credit"] + s_pc * w["pc"]
    ) / wsum

    df = pd.DataFrame(
        {
            "composite": composite,
            "vix_level": s_vix,
            "term": s_term,
            "credit": s_credit,
            "put_call": s_pc,
            "spy": spy,
        }
    ).dropna()
    df["zone"] = pd.cut(
        df["composite"],
        bins=[-0.1, 39.999, 69.999, 100.1],
        labels=["DEFENSIVE", "REDUCED", "FULL DEPLOY"],
    )
    # use yesterday's composite to classify today (no look-ahead)
    df["zone_lag"] = df["zone"].shift(1)
    df["spy_fwd_1d"] = df["spy"].pct_change().shift(-1)
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(CACHE)
    except Exception:
        pass
    return df


def zone_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.dropna(subset=["zone_lag", "spy_fwd_1d"]).groupby("zone_lag", observed=True)["spy_fwd_1d"]
    out = pd.DataFrame(
        {
            "days": g.count(),
            "avg_fwd_1d_pct": (g.mean() * 100).round(3),
            "hit_rate_pct": (g.apply(lambda s: (s > 0).mean()) * 100).round(1),
        }
    )
    return out
