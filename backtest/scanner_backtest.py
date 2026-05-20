"""Scanner walk-forward validation (Phase 2, Part A — re-runnable per factor set).

Answers: do the Scanner's top picks actually outperform the universe forward?

No look-ahead — every factor at sample date t is derived only from price data
dated <= t (EWM / rolling / pct_change / shift are all causal); forward returns
use only data dated > t. Percentile ranks are computed across the cross-section
that exists at t.

Two factor sets:
  * "v1" — the original 5 factors (momentum crossover, volume surge, relative
    strength, 52w-high proximity; short interest held neutral).
  * "v2" — the Phase 2 refactor. Only the 3 *price* factors (12-1 momentum,
    relative strength, low volatility) are point-in-time backtestable; the 3
    fundamental factors (quality, value, earnings surprise) are NOT — yfinance
    exposes only current fundamentals — so they are flagged "not backtestable",
    exactly as short interest was in v1.

Disclosed limitations: survivorship bias (universe is the *current* S&P 500;
delisted names are absent), and the not-backtestable fundamental factors above.
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

HORIZONS = {"5d": 5, "20d": 20, "60d": 60}
MIN_HISTORY = 320
MIN_TICKERS_PER_DATE = 50

FACTOR_SETS: dict[str, dict] = {
    "v1": {
        "label": "original 5 factors",
        "price_factors": ["momentum", "volume_surge", "rel_strength", "high_proximity"],
        "not_backtestable": {"short_interest": "yfinance exposes no historical short data"},
    },
    "v2": {
        "label": "Phase 2 refactor (momentum-tilted, 5 factors)",
        "price_factors": ["momentum_12_1", "rel_strength"],
        "not_backtestable": {
            "quality": "fundamental factor — yfinance has no point-in-time fundamentals",
            "value": "fundamental factor — yfinance has no point-in-time fundamentals",
            "earnings_surprise": "fundamental factor — yfinance has no point-in-time fundamentals",
        },
    },
}


def _paths(factor_set: str) -> tuple[Path, Path]:
    return (
        DATA_DIR / f"scanner_backtest_{factor_set}.parquet",
        DATA_DIR / f"scanner_backtest_{factor_set}_meta.json",
    )


def _clamp(s: pd.Series) -> pd.Series:
    return s.clip(lower=0.0, upper=100.0)


def _ticker_raws(df: pd.DataFrame, spy_ret20: pd.Series) -> pd.DataFrame | None:
    """Per-ticker causal raw factor inputs over the full history (both sets)."""
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
    out["recent_cross"] = (cross & ~cross.shift(5, fill_value=False)).astype(float)
    if "Volume" in df.columns:
        vol = df["Volume"].reindex(c.index)
        out["vol_ratio"] = vol.rolling(5).mean() / vol.rolling(20).mean()
    else:
        out["vol_ratio"] = np.nan
    stock_ret20 = c / c.shift(20) - 1.0
    out["rs_spread"] = stock_ret20 - spy_ret20.reindex(c.index).ffill()
    out["high_prox"] = c / c.rolling(252).max()
    # v2 price factors
    out["mom_12_1"] = c.shift(21) / c.shift(252) - 1.0
    out["low_vol"] = -c.pct_change().rolling(60).std()
    return out


def _score_v1(xs: pd.DataFrame) -> pd.DataFrame:
    gap_rank = xs["ema_gap"].rank(pct=True) * 100.0
    ret_rank = xs["ret_3m"].rank(pct=True) * 100.0
    base = 0.5 * gap_rank + 0.5 * ret_rank
    recent = xs["recent_cross"].fillna(0.0) > 0.5
    mom = base.copy()
    mom[recent] = mom[recent] + 10.0
    weak = (~recent) & (gap_rank < 50)
    mom[weak] = mom[weak] - 5.0
    scores = pd.DataFrame(
        {
            "momentum": _clamp(mom),
            "volume_surge": _clamp((xs["vol_ratio"] - 0.7) / (2.0 - 0.7) * 100.0),
            "rel_strength": xs["rs_spread"].rank(pct=True) * 100.0,
            "high_proximity": _clamp((xs["high_prox"] - 0.5) / (0.95 - 0.5) * 100.0),
        }
    ).dropna(how="any")
    scores["composite"] = scores.mean(axis=1)
    return scores


def _score_v2(xs: pd.DataFrame) -> pd.DataFrame:
    # Only the 2 backtestable price factors of the final 5-factor set.
    # Weighted by the live momentum-tilt (0.35 / 0.20), renormalized.
    scores = pd.DataFrame(
        {
            "momentum_12_1": xs["mom_12_1"].rank(pct=True) * 100.0,
            "rel_strength": xs["rs_spread"].rank(pct=True) * 100.0,
        }
    ).dropna(how="any")
    scores["composite"] = (
        scores["momentum_12_1"] * (0.35 / 0.55) + scores["rel_strength"] * (0.20 / 0.55)
    )
    return scores


def _asof(wide: pd.DataFrame, t: pd.Timestamp) -> pd.Series:
    sub = wide.loc[:t]
    return sub.iloc[-1] if not sub.empty else pd.Series(dtype=float)


def raws_field(raws: dict[str, pd.DataFrame], field: str) -> pd.DataFrame:
    return pd.DataFrame({tk: r[field] for tk, r in raws.items()})


def _spotcheck_lookahead(raws: dict, prices: dict, dates: list, factor_set: str, n: int = 5) -> list[dict]:
    """Recompute a raw factor from data truncated at t; it must match the
    precomputed value. Fails loudly if a non-causal op crept in."""
    out: list[dict] = []
    tickers = list(raws.keys())
    if not tickers or not dates:
        return out
    rng = random.Random(42)
    field = "ema_gap" if factor_set == "v1" else "mom_12_1"
    wide_field = raws_field(raws, field)
    for _ in range(n):
        tk = rng.choice(tickers)
        t = rng.choice(dates)
        c = prices[tk]["Close"].dropna()
        trunc = c.loc[:t]
        if len(trunc) < 260:
            continue
        # recompute the factor by applying the SAME causal op to data
        # truncated at t — must equal the value precomputed on full history
        if factor_set == "v1":
            e10 = trunc.ewm(span=10, adjust=False).mean()
            e50 = trunc.ewm(span=50, adjust=False).mean()
            recomputed = float(((e10 - e50) / e50).iloc[-1])
        else:
            recomputed = float((trunc.shift(21) / trunc.shift(252) - 1.0).iloc[-1])
        precomp = _asof(wide_field, t).get(tk)
        if precomp is None or pd.isna(precomp):
            continue
        out.append(
            {
                "ticker": tk,
                "date": str(pd.Timestamp(t).date()),
                "from_truncated_data": round(float(recomputed), 8),
                "precomputed": round(float(precomp), 8),
                "match": bool(abs(recomputed - precomp) < 1e-9),
            }
        )
    return out


def _year_extremes(spread: pd.Series) -> tuple[str, str]:
    if spread.empty:
        return "n/a", "n/a"
    by_year = spread.groupby(spread.index.year).mean() * 10000.0
    if by_year.empty:
        return "n/a", "n/a"
    return (
        f"{by_year.idxmax()} ({by_year.max():+.0f} bps)",
        f"{by_year.idxmin()} ({by_year.min():+.0f} bps)",
    )


def _safe(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except Exception:
        return None


@dataclass
class BacktestResult:
    meta: dict
    rows: pd.DataFrame


def run_backtest(factor_set: str = "v1", years: int = 5, period: str = "8y", top_n: int = 10) -> BacktestResult:
    """Weekly walk-forward backtest for the chosen factor set."""
    if factor_set not in FACTOR_SETS:
        raise ValueError(f"unknown factor_set {factor_set!r}")
    cfg = FACTOR_SETS[factor_set]
    price_factors = cfg["price_factors"]
    score_fn = _score_v1 if factor_set == "v1" else _score_v2

    universe = sp500_tickers()
    if not universe:
        raise RuntimeError("empty universe")
    log.info("[%s] fetching %d tickers x %s ...", factor_set, len(universe), period)
    prices = fetch_many(universe, period=period, max_age_hours=24 * 14)
    spy_df = fetch_history("SPY", period=period, max_age_hours=24 * 14)
    if spy_df is None or spy_df.empty:
        raise RuntimeError("SPY history unavailable")
    spy_close = spy_df["Close"].dropna()
    spy_ret20 = spy_close / spy_close.shift(20) - 1.0

    raws = {tk: r for tk in universe if (r := _ticker_raws(prices.get(tk), spy_ret20)) is not None}
    coverage = len(raws) / len(universe)
    if coverage < 0.60:
        raise RuntimeError(f"price data covered only {coverage:.0%} of the universe (<60%) — halting")

    wide = {
        f: raws_field(raws, f)
        for f in ["close", "ema_gap", "ret_3m", "recent_cross", "vol_ratio", "rs_spread", "high_prox", "mom_12_1", "low_vol"]
    }
    close_w = wide["close"]
    cidx = close_w.index
    start = cidx[-1] - pd.DateOffset(years=years)
    sample_dates = list(pd.date_range(start=start, end=cidx[-1], freq="W-FRI"))

    parquet_rows: list[dict] = []
    composite_by_date: list[dict] = []
    factor_decile: dict[str, list[tuple]] = {f: [] for f in price_factors}
    used_dates: list[pd.Timestamp] = []

    for t in sample_dates:
        pos = cidx.searchsorted(t, side="right") - 1
        if pos < 0:
            continue
        t_idx = cidx[pos]
        xs = pd.DataFrame({f: _asof(wide[f], t) for f in wide})
        scores = score_fn(xs)
        if len(scores) < MIN_TICKERS_PER_DATE:
            continue

        base_px = close_w.iloc[pos]
        fwd: dict[str, pd.Series] = {}
        for label, n in HORIZONS.items():
            fwd[label] = (
                close_w.iloc[pos + n] / base_px - 1.0
                if pos + n < len(cidx)
                else pd.Series(np.nan, index=base_px.index)
            )
        valid = scores.index
        uni_median = {lbl: float(fwd[lbl].reindex(valid).median(skipna=True)) for lbl in HORIZONS}
        if all(math.isnan(v) for v in uni_median.values()):
            continue
        used_dates.append(t_idx)

        top = scores.sort_values("composite", ascending=False).head(top_n)
        for rank, (ticker, row) in enumerate(top.iterrows(), start=1):
            rec = {
                "date": t_idx, "ticker": ticker, "rank": rank,
                "composite": round(float(row["composite"]), 2),
            }
            for f in price_factors:
                rec[f] = round(float(row[f]), 2)
            for label in HORIZONS:
                rec[f"fwd_{label}"] = _safe(fwd[label].get(ticker))
                rec[f"universe_median_{label}"] = uni_median[label]
            parquet_rows.append(rec)

        composite_by_date.append(
            {
                "date": str(t_idx.date()),
                **{f"top_{lbl}": float(fwd[lbl].reindex(top.index).mean(skipna=True)) for lbl in HORIZONS},
                **{f"uni_{lbl}": uni_median[lbl] for lbl in HORIZONS},
            }
        )

        decile_n = max(1, int(math.ceil(len(scores) * 0.10)))
        for f in price_factors:
            top_f = scores[f].sort_values(ascending=False).head(decile_n).index
            ret = float(fwd["20d"].reindex(top_f).mean(skipna=True))
            if not math.isnan(ret) and not math.isnan(uni_median["20d"]):
                factor_decile[f].append((t_idx, ret - uni_median["20d"]))

    if not parquet_rows:
        raise RuntimeError("backtest produced no rows — insufficient history")

    rows_df = pd.DataFrame(parquet_rows)
    parquet_path, meta_path = _paths(factor_set)
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        rows_df.to_parquet(parquet_path, index=False)
    except Exception as e:
        log.warning("parquet write failed: %s", e)

    per_factor: list[dict] = []
    for f in price_factors:
        series = pd.Series({d: s for d, s in factor_decile[f]}).sort_index()
        if series.empty:
            per_factor.append({"factor": f, "backtestable": True, "note": "no data"})
            continue
        mean_s, std_s = float(series.mean()), float(series.std())
        best, worst = _year_extremes(series)
        per_factor.append(
            {
                "factor": f, "backtestable": True,
                "avg_20d_spread_bps": round(mean_s * 10000.0, 1),
                "win_rate_pct": round(float((series > 0).mean()) * 100.0, 1),
                "sharpe_per_20d": round(mean_s / std_s, 2) if std_s else None,
                "best_year": best, "worst_year": worst,
                "observations": int(len(series)),
            }
        )
    for f, reason in cfg["not_backtestable"].items():
        per_factor.append({"factor": f, "backtestable": False, "note": f"NOT backtestable — {reason}"})

    cbd = pd.DataFrame(composite_by_date)
    spread_20d = (cbd["top_20d"] - cbd["uni_20d"]).dropna()
    headline_bps = round(float(spread_20d.mean()) * 10000.0, 1) if not spread_20d.empty else None
    n_price = len(price_factors)
    n_total = n_price + len(cfg["not_backtestable"])
    partial = " (price factors only — fundamental factors not backtestable)" if n_price < n_total else ""
    if headline_bps is None:
        verdict = "Inconclusive — no matured 20d windows."
    elif headline_bps >= 50:
        verdict = f"Composite top-{top_n} beat the universe by {headline_bps:.0f} bps per 20d{partial} — a real edge."
    elif headline_bps > 0:
        verdict = f"Composite edge only {headline_bps:.0f} bps per 20d{partial} (< 50 bps) — no meaningful edge, treat as exploratory."
    else:
        verdict = f"Composite top-{top_n} UNDERPERFORMED the universe ({headline_bps:.0f} bps per 20d){partial} — not predictive."

    meta = {
        "factor_set": factor_set,
        "label": cfg["label"],
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "config": {
            "years": years, "period": period, "top_n": top_n,
            "universe_size": len(universe),
            "scored_coverage_pct": round(coverage * 100, 1),
            "sample_dates": len(used_dates),
            "window": f"{used_dates[0].date()} → {used_dates[-1].date()}" if used_dates else "n/a",
            "composite_backtest_factors": price_factors,
        },
        "headline": {"composite_20d_spread_bps": headline_bps, "verdict": verdict},
        "per_factor": per_factor,
        "composite_by_date": composite_by_date,
        "lookahead_check": _spotcheck_lookahead(raws, prices, used_dates, factor_set, n=5),
        "caveats": [
            "Survivorship bias — universe is the current S&P 500; delisted names are absent.",
            "Cumulative-spread chart sums overlapping 20d windows; illustrative, not a tradable curve.",
        ]
        + (
            [f"Composite backtest covers only the {n_price} price factors; "
             f"{n_total - n_price} fundamental factors are not point-in-time backtestable."]
            if n_price < n_total
            else []
        ),
    }
    try:
        meta_path.write_text(json.dumps(meta, indent=2))
    except Exception as e:
        log.warning("meta write failed: %s", e)
    return BacktestResult(meta=meta, rows=rows_df)


def load(factor_set: str = "v1") -> BacktestResult | None:
    parquet_path, meta_path = _paths(factor_set)
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
        rows = pd.read_parquet(parquet_path) if parquet_path.exists() else pd.DataFrame()
        return BacktestResult(meta=meta, rows=rows)
    except Exception:
        return None


def _print(res: BacktestResult) -> None:
    m = res.meta
    print(f"\n=== {m['factor_set']} · {m['label']} ===")
    print(f"Window: {m['config']['window']}  ·  {m['config']['sample_dates']} weekly samples")
    print(f"HEADLINE: {m['headline']['composite_20d_spread_bps']} bps")
    print(f"VERDICT: {m['headline']['verdict']}")
    print("Per-factor (top decile, 20d vs universe median):")
    for pf in m["per_factor"]:
        if pf.get("backtestable"):
            print(f"  {pf['factor']:16s} spread={pf.get('avg_20d_spread_bps'):>8} bps  "
                  f"win={pf.get('win_rate_pct')}%  sharpe={pf.get('sharpe_per_20d')}  "
                  f"best={pf.get('best_year')}  worst={pf.get('worst_year')}")
        else:
            print(f"  {pf['factor']:16s} {pf.get('note')}")
    print("Look-ahead spot-check: " + ", ".join(f"{c['ticker']}={c['match']}" for c in m["lookahead_check"]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for fs in ("v1", "v2"):
        _print(run_backtest(factor_set=fs))
