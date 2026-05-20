"""Walk-forward validation of the macro composite over ~10 years.

Single-pass backtests over one regime (a bull market) tell you almost
nothing — you can always fit weights that look good in-sample. This module
does the honest thing:

  1. Fit composite weights on a 5-year training window (regression of the 4
     signal scores onto forward 20-day SPY return; coefficients clipped to
     non-negative and normalized to sum to 1).
  2. Apply those frozen weights to the *next* 6 months — pure out-of-sample.
  3. Roll the windows forward 6 months and repeat.

The concatenated test windows form a continuous out-of-sample track record.
We also carry an equal-weight composite through the same windows, so you can
see whether refitting actually adds anything beyond a naive blend.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.panel import SIGNALS, fetch_panel

OOS_CACHE = Path(__file__).resolve().parent.parent / "data" / "walk_forward.parquet"
SEG_CACHE = Path(__file__).resolve().parent.parent / "data" / "walk_forward_segments.json"

EQUAL = {s: 1.0 / len(SIGNALS) for s in SIGNALS}


def _forward_return(spy: pd.Series, n: int) -> pd.Series:
    return spy.shift(-n) / spy - 1.0


def _zone(comp: pd.Series) -> pd.Series:
    return pd.cut(
        comp,
        bins=[-0.1, 39.999, 69.999, 100.1],
        labels=["DEFENSIVE", "REDUCED", "FULL DEPLOY"],
    )


def fit_weights(train: pd.DataFrame, horizon: int = 20) -> dict[str, float]:
    """Regress forward SPY return on the 4 signal scores; clip + normalize.

    Weights stay in raw 0-100 score space so the composite remains a 0-100
    weighted average and the 40/70 zone thresholds keep their meaning.
    """
    y = _forward_return(train["spy"], horizon)
    d = train[SIGNALS].join(y.rename("y")).dropna()
    if len(d) < 100:
        return dict(EQUAL)
    X = d[SIGNALS].values
    yv = d["y"].values
    design = np.column_stack([X, np.ones(len(X))])  # intercept
    coef, *_ = np.linalg.lstsq(design, yv, rcond=None)
    coef = np.clip(coef[: len(SIGNALS)], 0.0, None)
    if coef.sum() <= 1e-12:
        return dict(EQUAL)
    w = coef / coef.sum()
    return {s: float(x) for s, x in zip(SIGNALS, w)}


def _composite(df: pd.DataFrame, w: dict[str, float]) -> pd.Series:
    return sum(df[s] * w[s] for s in SIGNALS)


def walk_forward(
    panel: pd.DataFrame | None = None,
    train_days: int = 1260,
    test_days: int = 126,
    horizon: int = 20,
) -> tuple[pd.DataFrame, list[dict]]:
    """Return (oos_df, segments). oos_df is the concatenated OOS test windows."""
    if panel is None:
        # max history (~2008+ once VIX3M/HYG are available) so the rolling
        # 5y training window still leaves 2018 vol, COVID and the 2022 bear
        # inside the *out-of-sample* test set rather than the training set.
        panel = fetch_panel("max")
    if panel.empty or len(panel) < train_days + test_days:
        return pd.DataFrame(), []

    segments: list[dict] = []
    rows: list[pd.DataFrame] = []
    n = len(panel)
    start = 0
    while start + train_days + test_days <= n:
        train = panel.iloc[start : start + train_days]
        test = panel.iloc[start + train_days : start + train_days + test_days].copy()
        w = fit_weights(train, horizon)

        test["composite"] = _composite(test, w)
        test["composite_eq"] = _composite(test, EQUAL)
        test["zone"] = _zone(test["composite"])
        test["zone_eq"] = _zone(test["composite_eq"])
        test["zone_lag"] = test["zone"].shift(1)
        test["zone_eq_lag"] = test["zone_eq"].shift(1)
        test["spy_fwd_1d"] = test["spy"].pct_change().shift(-1)
        rows.append(test)

        segments.append(
            {
                "train_start": str(train.index[0].date()),
                "train_end": str(train.index[-1].date()),
                "test_start": str(test.index[0].date()),
                "test_end": str(test.index[-1].date()),
                "weights": {k: round(v, 3) for k, v in w.items()},
            }
        )
        start += test_days

    oos = pd.concat(rows) if rows else pd.DataFrame()
    try:
        OOS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        oos.to_parquet(OOS_CACHE)
        SEG_CACHE.write_text(json.dumps(segments, indent=2))
    except Exception:
        pass
    return oos, segments


def zone_summary(oos: pd.DataFrame, zone_col: str = "zone_lag") -> pd.DataFrame:
    """Out-of-sample forward 1-day SPY return grouped by zone."""
    if oos.empty or zone_col not in oos.columns:
        return pd.DataFrame()
    g = oos.dropna(subset=[zone_col, "spy_fwd_1d"]).groupby(zone_col, observed=True)["spy_fwd_1d"]
    return pd.DataFrame(
        {
            "days": g.count(),
            "avg_fwd_1d_pct": (g.mean() * 100).round(3),
            "hit_rate_pct": (g.apply(lambda s: (s > 0).mean()) * 100).round(1),
        }
    )


def load() -> tuple[pd.DataFrame, list[dict]] | None:
    if OOS_CACHE.exists() and SEG_CACHE.exists():
        try:
            return pd.read_parquet(OOS_CACHE), json.loads(SEG_CACHE.read_text())
        except Exception:
            return None
    return None


if __name__ == "__main__":
    oos_df, segs = walk_forward()
    if oos_df.empty:
        print("Not enough history for a walk-forward run.")
    else:
        print(f"{len(segs)} out-of-sample segments · {len(oos_df)} OOS test days")
        print(f"OOS window: {oos_df.index[0].date()} → {oos_df.index[-1].date()}\n")
        print("Fitted-weight OOS zones:")
        print(zone_summary(oos_df, "zone_lag"))
        print("\nEqual-weight OOS zones:")
        print(zone_summary(oos_df, "zone_eq_lag"))
