"""12-1 Momentum (Jegadeesh-Titman).

Return from month t-12 to month t-1 — the last 11 months, excluding the most
recent month (which mean-reverts). 21 trading days ≈ 1 month, 252 ≈ 12 months.
"""
from __future__ import annotations

import pandas as pd

LOOKBACK = 252  # ~12 months
SKIP = 21       # ~1 month, excluded


def raw_momentum(prices: dict[str, pd.DataFrame]) -> pd.Series:
    """Raw 12-1 return per ticker. Higher = stronger momentum."""
    out: dict[str, float] = {}
    for ticker, df in prices.items():
        if df is None or df.empty or "Close" not in df.columns:
            continue
        c = df["Close"].dropna()
        if len(c) < LOOKBACK + 5:
            continue
        p_start = float(c.iloc[-LOOKBACK])
        p_end = float(c.iloc[-SKIP])
        if p_start > 0:
            out[ticker] = p_end / p_start - 1.0
    return pd.Series(out, name="momentum_12_1", dtype=float)
