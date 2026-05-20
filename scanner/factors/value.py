"""Value (EV/EBITDA).

enterprise_value / ebitda. Lower = cheaper = better, so the raw signal is the
inverted ratio. Tickers with non-positive EBITDA are excluded (NaN) — a
negative multiple is not interpretable as "cheap".
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def raw_value(fundamentals: pd.DataFrame) -> pd.Series:
    """Raw inverted EV/EBITDA per ticker. Higher = cheaper. NaN if EBITDA <= 0."""
    if fundamentals is None or fundamentals.empty:
        return pd.Series(dtype=float, name="value")
    ev = fundamentals.get("enterprise_value")
    ebitda = fundamentals.get("ebitda")
    if ev is None or ebitda is None:
        return pd.Series(np.nan, index=fundamentals.index, name="value")
    ratio = ev / ebitda
    inv = 1.0 / ratio
    bad = ebitda.isna() | ev.isna() | (ebitda <= 0) | (ev <= 0)
    inv[bad] = np.nan
    return inv.rename("value")
