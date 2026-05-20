"""Part A — Scanner walk-forward validation.

Answers one question: do the Scanner's top picks actually outperform the
universe forward, or are the 5 factors noise?

No look-ahead — every factor at sample date t is derived only from price data
dated <= t (EWM / rolling / pct_change are all causal); forward returns use
only data dated > t. Percentile ranks are computed across the cross-section
that exists at t.

Two disclosed limitations:

  * Survivorship bias — the universe is the *current* S&P 500. True
    point-in-time membership needs a historical constituent feed we don't
    have, so names delisted over the window are absent. Real spreads are
    likely a touch worse than reported here.

  * Short Interest Decline is NOT backtestable — yfinance exposes only
    current short data, no history. It is held neutral (50) historically and
    flagged "not backtestable" in the per-factor table.
"""
from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scanner.universe import sp500_tickers
from utils.data_fetch import fetch_history, fetch_many

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PARQUET = DATA_DIR / "scanner_backtest.parquet"
META = DATA_DIR / "scanner_backtest_meta.json"

# factor scores that can be reconstructed point-in-time from price history
PRICE_FACTORS = ["momentum", "volume_surge", "rel_strength", "high_proximity"]
SHORT_NEUTRAL = 50.0  # short interest is not backtestable — held neutral
FACTOR_WEIGHT = 0.20  # equal weight, mirrors the live scanner/ranker.py
HORIZONS = {"5d": 5, "20d": 20, "60d": 60}
MIN_HISTORY = 320  # trading days required before a ticker can be scored
MIN_TICKERS_PER_DATE = 50


def _clamp(s: pd.Series) -> pd.Series:
    return s.clip(lower=0.0, upper=100.0)


def _ticker_raws(df: pd.DataFrame, spy_ret20: pd.Series) -> pd.DataFrame | None:
    """Per-ticker causal raw factor inputs over the full history."""
    if df is None or df.empty or "Close" not in df.columns:
        return None
    c = df["Close"].dropna()
    if len(c) < MIN_HISTORY:
        return None
    out = pd.DataFrame(index=c.index)
    out["close"] = c
    ema10 = c.ewm(span=10, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    out["ema_gap"] = (ema10 - ema50) / ema50
    out["ret_3m"] = c / c.shift(63) - 1.0
    cross = ema10 > ema50
    prev = cross.shift(5, fill_value=False)
    out["recent_cross"] = (cross & ~prev).astype(float)
    if "Volume" in df.columns:
        vol = df["Volume"].reindex(c.index)
        out["vol_ratio"] = vol.rolling(5).mean() / vol.rolling(20).mean()
    else:
        out["vol_ratio"] = np.nan
    stock_ret20 = c / c.shift(20) - 1.0
    out["rs_spread"] = stock_ret20 - spy_ret20.reindex(c.index).ffill()
    out["high_prox"] = c / c.rolling(252).max()
    return out


def _score_cross_section(xs: pd.DataFrame) -> pd.DataFrame:
    """Map the as-of raw values to 0-100 factor scores (mirrors factors.py)."""
    gap_rank = xs["ema_gap"].rank(pct=True) * 100.0
    ret_rank = xs["ret_3m"].rank(pct=True) * 100.0
    base = 0.5 * gap_rank + 0.5 * ret_rank
    recent = xs["recent_cross"].fillna(0.0) > 0.5
    mom = base.copy()
    mom[recent] = mom[recent] + 10.0
    weak = (~recent) & (gap_rank < 50)
    mom[weak] = mom[weak] - 5.0
    momentum = _clamp(mom)

    volume_surge = _clamp((xs["vol_ratio"] - 0.7) / (2.0 - 0.7) * 100.0)
    rel_strength = xs["rs_spread"].rank(pct=True) * 100.0
    high_proximity = _clamp((xs["high_prox"] - 0.5) / (0.95 - 0.5) * 100.0)

    scores = pd.DataFrame(
        {
            "momentum": momentum,
            "volume_surge": volume_surge,
            "rel_strength": rel_strength,
            "high_proximity": high_proximity,
        }
    )
    # drop, don't impute — a ticker missing any factor is out of the ranking
    scores = scores.dropna(how="any")
    scores["short_interest"] = SHORT_NEUTRAL
    scores["composite"] = (
        FACTOR_WEIGHT
        * (scores["momentum"] + scores["volume_surge"] + scores["rel_strength"] + scores["high_proximity"])
        + FACTOR_WEIGHT * SHORT_NEUTRAL
    )
    return scores


def _asof(wide: pd.DataFrame, t: pd.Timestamp) -> pd.Series:
    sub = wide.loc[:t]
    if sub.empty:
        return pd.Series(dtype=float)
    return sub.iloc[-1]


def _spotcheck_lookahead(raws: dict[str, pd.DataFrame], prices: dict[str, pd.DataFrame],
                         dates: list[pd.Timestamp], n: int = 4) -> list[dict]:
    """Recompute a raw factor from data truncated at t and confirm it matches.

    If any non-causal operation crept into _ticker_raws this fails loudly.
    """
    out: list[dict] = []
    tickers = list(raws.keys())
    if not tickers or not dates:
        return out
    rng = random.Random(42)
    for _ in range(n):
        tk = rng.choice(tickers)
        t = rng.choice(dates)
        c = prices[tk]["Close"].dropna()
        trunc = c.loc[:t]
        if len(trunc) < 70:
            continue
        ema10 = trunc.ewm(span=10, adjust=False).mean().iloc[-1]
        ema50 = trunc.ewm(span=50, adjust=False).mean().iloc[-1]
        gap_trunc = (ema10 - ema50) / ema50
        precomp = _asof(raws_field(raws, "ema_gap"), t).get(tk)
        if precomp is None or pd.isna(precomp):
            continue
        out.append(
            {
                "ticker": tk,
                "date": str(pd.Timestamp(t).date()),
                "gap_from_truncated_data": round(float(gap_trunc), 8),
                "gap_precomputed": round(float(precomp), 8),
                "match": bool(abs(gap_trunc - precomp) < 1e-9),
            }
        )
    return out


def raws_field(raws: dict[str, pd.DataFrame], field: str) -> pd.DataFrame:
    return pd.DataFrame({tk: r[field] for tk, r in raws.items()})


def _year_extremes(spread_by_date: pd.Series) -> tuple[str, str]:
    if spread_by_date.empty:
        return "n/a", "n/a"
    by_year = spread_by_date.groupby(spread_by_date.index.year).mean() * 10000.0
    if by_year.empty:
        return "n/a", "n/a"
    best = by_year.idxmax()
    worst = by_year.idxmin()
    return (
        f"{best} ({by_year[best]:+.0f} bps)",
        f"{worst} ({by_year[worst]:+.0f} bps)",
    )


@dataclass
class BacktestResult:
    meta: dict
    rows: pd.DataFrame


def run_backtest(years: int = 5, period: str = "8y", top_n: int = 10) -> BacktestResult:
    """Weekly walk-forward backtest of the current 5-factor scanner."""
    universe = sp500_tickers()
    if not universe:
        raise RuntimeError("empty universe")

    log.info("fetching %d tickers x %s ...", len(universe), period)
    prices = fetch_many(universe, period=period, max_age_hours=24 * 14)
    spy_df = fetch_history("SPY", period=period, max_age_hours=24 * 14)
    if spy_df is None or spy_df.empty:
        raise RuntimeError("SPY history unavailable")
    spy_close = spy_df["Close"].dropna()
    spy_ret20 = spy_close / spy_close.shift(20) - 1.0

    raws: dict[str, pd.DataFrame] = {}
    for tk in universe:
        r = _ticker_raws(prices.get(tk), spy_ret20)
        if r is not None:
            raws[tk] = r
    coverage = len(raws) / len(universe)
    if coverage < 0.60:
        raise RuntimeError(
            f"price data covered only {coverage:.0%} of the universe (<60%) — "
            "halting instead of producing a partial backtest"
        )

    wide = {
        f: raws_field(raws, f)
        for f in ["close", "ema_gap", "ret_3m", "recent_cross", "vol_ratio", "rs_spread", "high_prox"]
    }
    close_w = wide["close"]
    cidx = close_w.index

    last = cidx[-1]
    start = last - pd.DateOffset(years=years)
    sample_dates = list(pd.date_range(start=start, end=last, freq="W-FRI"))

    parquet_rows: list[dict] = []
    composite_by_date: list[dict] = []
    factor_decile: dict[str, list[tuple[pd.Timestamp, float]]] = {f: [] for f in PRICE_FACTORS}
    used_dates: list[pd.Timestamp] = []

    for t in sample_dates:
        pos = cidx.searchsorted(t, side="right") - 1
        if pos < 0:
            continue
        t_idx = cidx[pos]

        xs = pd.DataFrame(
            {
                "ema_gap": _asof(wide["ema_gap"], t),
                "ret_3m": _asof(wide["ret_3m"], t),
                "recent_cross": _asof(wide["recent_cross"], t),
                "vol_ratio": _asof(wide["vol_ratio"], t),
                "rs_spread": _asof(wide["rs_spread"], t),
                "high_prox": _asof(wide["high_prox"], t),
            }
        )
        scores = _score_cross_section(xs)
        if len(scores) < MIN_TICKERS_PER_DATE:
            continue

        base_px = close_w.iloc[pos]
        fwd: dict[str, pd.Series] = {}
        for label, n in HORIZONS.items():
            if pos + n < len(cidx):
                fwd[label] = close_w.iloc[pos + n] / base_px - 1.0
            else:
                fwd[label] = pd.Series(np.nan, index=base_px.index)

        valid = scores.index
        uni_median = {
            label: float(fwd[label].reindex(valid).median(skipna=True))
            for label in HORIZONS
        }
        if all(math.isnan(v) for v in uni_median.values()):
            continue
        used_dates.append(t_idx)

        top = scores.sort_values("composite", ascending=False).head(top_n)
        for rank, (ticker, row) in enumerate(top.iterrows(), start=1):
            parquet_rows.append(
                {
                    "date": t_idx,
                    "ticker": ticker,
                    "rank": rank,
                    "composite": round(float(row["composite"]), 2),
                    "momentum": round(float(row["momentum"]), 2),
                    "volume_surge": round(float(row["volume_surge"]), 2),
                    "rel_strength": round(float(row["rel_strength"]), 2),
                    "high_proximity": round(float(row["high_proximity"]), 2),
                    "short_interest": SHORT_NEUTRAL,
                    "fwd_5d": _safe(fwd["5d"].get(ticker)),
                    "fwd_20d": _safe(fwd["20d"].get(ticker)),
                    "fwd_60d": _safe(fwd["60d"].get(ticker)),
                    "universe_median_5d": uni_median["5d"],
                    "universe_median_20d": uni_median["20d"],
                    "universe_median_60d": uni_median["60d"],
                }
            )

        top_means = {
            label: float(fwd[label].reindex(top.index).mean(skipna=True))
            for label in HORIZONS
        }
        composite_by_date.append(
            {
                "date": str(t_idx.date()),
                "top_5d": top_means["5d"],
                "top_20d": top_means["20d"],
                "top_60d": top_means["60d"],
                "uni_5d": uni_median["5d"],
                "uni_20d": uni_median["20d"],
                "uni_60d": uni_median["60d"],
            }
        )

        # per-factor decile (A.2) — top 10% by each factor alone, 20d horizon
        decile_n = max(1, int(math.ceil(len(scores) * 0.10)))
        for f in PRICE_FACTORS:
            top_f = scores[f].sort_values(ascending=False).head(decile_n).index
            ret = float(fwd["20d"].reindex(top_f).mean(skipna=True))
            if not math.isnan(ret) and not math.isnan(uni_median["20d"]):
                factor_decile[f].append((t_idx, ret - uni_median["20d"]))

    if not parquet_rows:
        raise RuntimeError("backtest produced no rows — insufficient history")

    rows_df = pd.DataFrame(parquet_rows)
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        rows_df.to_parquet(PARQUET, index=False)
    except Exception as e:
        log.warning("parquet write failed: %s", e)

    # per-factor table
    per_factor: list[dict] = []
    for f in PRICE_FACTORS:
        series = pd.Series(
            {d: s for d, s in factor_decile[f]}
        ).sort_index()
        if series.empty:
            per_factor.append({"factor": f, "backtestable": True, "note": "no data"})
            continue
        mean_spread = float(series.mean())
        std_spread = float(series.std())
        best, worst = _year_extremes(series)
        per_factor.append(
            {
                "factor": f,
                "backtestable": True,
                "avg_20d_spread_bps": round(mean_spread * 10000.0, 1),
                "win_rate_pct": round(float((series > 0).mean()) * 100.0, 1),
                "sharpe_per_20d": round(mean_spread / std_spread, 2) if std_spread else None,
                "best_year": best,
                "worst_year": worst,
                "observations": int(len(series)),
            }
        )
    per_factor.append(
        {
            "factor": "short_interest",
            "backtestable": False,
            "note": "NOT backtestable — yfinance exposes no historical short data",
        }
    )

    # composite headline
    cbd = pd.DataFrame(composite_by_date)
    spread_20d = (cbd["top_20d"] - cbd["uni_20d"]).dropna()
    headline_bps = round(float(spread_20d.mean()) * 10000.0, 1) if not spread_20d.empty else None
    if headline_bps is None:
        verdict = "Inconclusive — no matured 20d windows."
    elif headline_bps >= 50:
        verdict = (
            f"Composite top-{top_n} beat the universe median by {headline_bps:.0f} bps "
            f"per 20 trading days out-of-sample — a real, if modest, edge."
        )
    elif headline_bps > 0:
        verdict = (
            f"Composite edge is only {headline_bps:.0f} bps per 20d (< 50 bps "
            f"threshold) — no meaningful edge. Treat the scanner as an "
            f"exploratory tool, not an allocation signal."
        )
    else:
        verdict = (
            f"Composite top-{top_n} UNDERPERFORMED the universe median "
            f"({headline_bps:.0f} bps per 20d). The current factor set is not "
            f"predictive — do not trade on it."
        )

    lookahead = _spotcheck_lookahead(raws, prices, used_dates, n=5)

    meta = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "config": {
            "years": years,
            "period": period,
            "top_n": top_n,
            "universe_size": len(universe),
            "scored_coverage_pct": round(coverage * 100, 1),
            "sample_dates": len(used_dates),
            "window": f"{used_dates[0].date()} → {used_dates[-1].date()}" if used_dates else "n/a",
        },
        "headline": {"composite_20d_spread_bps": headline_bps, "verdict": verdict},
        "per_factor": per_factor,
        "composite_by_date": composite_by_date,
        "lookahead_check": lookahead,
        "caveats": [
            "Survivorship bias — universe is the current S&P 500; delisted names are absent.",
            "Short Interest Decline is held neutral (50) — yfinance has no historical short data.",
            "Cumulative-spread chart sums overlapping 20d windows; it is illustrative, not a tradable equity curve.",
        ],
    }
    try:
        META.write_text(json.dumps(meta, indent=2))
    except Exception as e:
        log.warning("meta write failed: %s", e)

    return BacktestResult(meta=meta, rows=rows_df)


def _safe(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except Exception:
        return None


def load() -> BacktestResult | None:
    if not META.exists():
        return None
    try:
        meta = json.loads(META.read_text())
        rows = pd.read_parquet(PARQUET) if PARQUET.exists() else pd.DataFrame()
        return BacktestResult(meta=meta, rows=rows)
    except Exception:
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    res = run_backtest()
    m = res.meta
    print(f"\nWindow: {m['config']['window']}  ·  {m['config']['sample_dates']} weekly samples")
    print(f"Universe coverage: {m['config']['scored_coverage_pct']}%\n")
    print(f"HEADLINE: composite top-10 20d spread = {m['headline']['composite_20d_spread_bps']} bps")
    print(f"VERDICT: {m['headline']['verdict']}\n")
    print("Per-factor (top decile, 20d forward vs universe median):")
    for pf in m["per_factor"]:
        if pf.get("backtestable"):
            print(
                f"  {pf['factor']:16s} spread={pf.get('avg_20d_spread_bps'):>7} bps  "
                f"win={pf.get('win_rate_pct')}%  sharpe={pf.get('sharpe_per_20d')}  "
                f"best={pf.get('best_year')}  worst={pf.get('worst_year')}"
            )
        else:
            print(f"  {pf['factor']:16s} {pf.get('note')}")
    print("\nLook-ahead spot-check:")
    for ch in m["lookahead_check"]:
        print(f"  {ch['ticker']} @ {ch['date']}: match={ch['match']}")
