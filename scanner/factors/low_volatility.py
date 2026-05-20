"""Low Volatility (Frazzini-Pedersen low-risk anomaly).

Inverse of 60-day realized daily-return standard deviation. Replaces 52-Week
High Proximity, which Part A showed had a negative forward spread (-16 bps).
"""
from __future__ import annotations

import pandas as pd

WINDOW = 60


def raw_low_volatility(prices: dict[str, pd.DataFrame]) -> pd.Series:
    """Raw signal = negative realized vol, so higher = lower vol = better."""
    out: dict[str, float] = {}
    for ticker, df in prices.items():
        if df is None or df.empty or "Close" not in df.columns:
            continue
        c = df["Close"].dropna()
        if len(c) < WINDOW + 5:
            continue
        rets = c.pct_change().dropna().tail(WINDOW)
        if len(rets) < WINDOW:
            continue
        sd = float(rets.std())
        if sd > 0:
            out[ticker] = -sd  # negate: less-negative (lower vol) ranks higher
    return pd.Series(out, name="low_volatility", dtype=float)
