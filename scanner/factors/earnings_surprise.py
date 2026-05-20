"""Earnings Surprise (Post-Earnings Announcement Drift).

Weighted EPS surprise over the last 2 reported quarters: 0.7 most recent,
0.3 prior. surprise% = (actual - estimate) / abs(estimate). A robust anomaly
across decades. Both quarters must be present — missing data is dropped, not
imputed (Part B.7).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

W_RECENT = 0.7
W_PRIOR = 0.3


def raw_earnings_surprise(fundamentals: pd.DataFrame) -> pd.Series:
    """Raw weighted EPS surprise per ticker. NaN if either quarter is missing."""
    if fundamentals is None or fundamentals.empty:
        return pd.Series(dtype=float, name="earnings_surprise")
    recent = fundamentals.get("surprise_recent")
    prior = fundamentals.get("surprise_prior")
    if recent is None or prior is None:
        return pd.Series(np.nan, index=fundamentals.index, name="earnings_surprise")
    combined = W_RECENT * recent + W_PRIOR * prior  # NaN propagates if either missing
    return combined.rename("earnings_surprise")
