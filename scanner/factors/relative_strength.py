"""Relative Strength vs SPY.

20-day stock return minus 20-day SPY return. Complements 12-1 momentum at a
shorter horizon. Kept unchanged from the original factor set (Part B.4).
"""
from __future__ import annotations

import pandas as pd

WINDOW = 20


def raw_rel_strength(prices: dict[str, pd.DataFrame], spy_close: pd.Series) -> pd.Series:
    """Raw 20d return spread vs SPY per ticker. Higher = outperforming."""
    if spy_close is None or spy_close.empty or len(spy_close.dropna()) < WINDOW + 1:
        return pd.Series(dtype=float, name="rel_strength")
    s = spy_close.dropna()
    spy_ret = float(s.iloc[-1] / s.iloc[-(WINDOW + 1)] - 1.0)

    out: dict[str, float] = {}
    for ticker, df in prices.items():
        if df is None or df.empty or "Close" not in df.columns:
            continue
        c = df["Close"].dropna()
        if len(c) < WINDOW + 1:
            continue
        stock_ret = float(c.iloc[-1] / c.iloc[-(WINDOW + 1)] - 1.0)
        out[ticker] = stock_ret - spy_ret
    return pd.Series(out, name="rel_strength", dtype=float)
