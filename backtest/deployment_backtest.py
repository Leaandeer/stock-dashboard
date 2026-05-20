"""Single-pass zone overlay using the fixed composite weights.

This is the visual "where have we been" backtest for the macro gate page.
For out-of-sample validation of the weights, see walk_forward.py.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest.panel import fetch_panel

CACHE = Path(__file__).resolve().parent.parent / "data" / "composite_history.parquet"

# fixed weights for the 4 daily-computable signals (original plan weights,
# re-normalized after dropping breadth + crowding)
FIXED_WEIGHTS = {"vix_level": 0.25, "term": 0.20, "credit": 0.15, "put_call": 0.10}


def _zone(comp: pd.Series) -> pd.Series:
    return pd.cut(
        comp,
        bins=[-0.1, 39.999, 69.999, 100.1],
        labels=["DEFENSIVE", "REDUCED", "FULL DEPLOY"],
    )


def compute(period: str = "3y") -> pd.DataFrame:
    """Daily composite + zone over the trailing period (fixed weights)."""
    panel = fetch_panel(period)
    if panel.empty:
        return pd.DataFrame()
    wsum = sum(FIXED_WEIGHTS.values())
    composite = sum(panel[k] * w for k, w in FIXED_WEIGHTS.items()) / wsum
    df = panel.copy()
    df["composite"] = composite
    df["zone"] = _zone(composite)
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
    return pd.DataFrame(
        {
            "days": g.count(),
            "avg_fwd_1d_pct": (g.mean() * 100).round(3),
            "hit_rate_pct": (g.apply(lambda s: (s > 0).mean()) * 100).round(1),
        }
    )
