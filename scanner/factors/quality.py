"""Gross Profitability (Novy-Marx 2013, "The Other Side of Value").

gross_profit (trailing 4 quarters) / total_assets (most recent). The strongest
non-momentum anomaly. Higher = more profitable per dollar of assets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def raw_quality(fundamentals: pd.DataFrame) -> pd.Series:
    """Raw gross profitability per ticker. NaN where inputs are missing/invalid."""
    if fundamentals is None or fundamentals.empty:
        return pd.Series(dtype=float, name="quality")
    gp = fundamentals.get("gross_profit_ttm")
    ta = fundamentals.get("total_assets")
    if gp is None or ta is None:
        return pd.Series(np.nan, index=fundamentals.index, name="quality")
    q = gp / ta
    q[(ta <= 0) | ta.isna() | gp.isna()] = np.nan
    return q.rename("quality")
