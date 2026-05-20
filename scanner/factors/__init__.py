"""Factor orchestration — raw factors → sector-neutral ranks → composite.

Each factor module produces a raw signal per ticker. This module:
  1. gathers the 6 raw factor series,
  2. percentile-ranks each *within sector* (Part B.6) so the scanner can't
     pile into whatever sector is hot — sectors too small to rank fall back
     to a universe-wide rank,
  3. drops any ticker missing any factor (Part B.7 — drop, never impute),
  4. equal-weights the per-factor 0-100 scores into a composite in [0,100].
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from scanner.constants import FACTOR_COLS, FACTOR_WEIGHTS, MIN_SECTOR_SIZE
from scanner.factors.earnings_surprise import raw_earnings_surprise
from scanner.factors.low_volatility import raw_low_volatility
from scanner.factors.momentum import raw_momentum
from scanner.factors.quality import raw_quality
from scanner.factors.relative_strength import raw_rel_strength
from scanner.factors.value import raw_value
from scanner.fundamentals import get_fundamentals
from scanner.universe import sp500_tickers
from utils.data_fetch import fetch_history, fetch_many

log = logging.getLogger(__name__)


def _sector_neutral_rank(raw: pd.Series, sectors: pd.Series) -> pd.Series:
    """Percentile-rank a raw factor within each sector (0-100)."""
    ranked = pd.Series(np.nan, index=raw.index, dtype=float)
    sec = sectors.reindex(raw.index).fillna("Unknown")
    for sector, members in sec.groupby(sec).groups.items():
        grp = raw.loc[members]
        if grp.notna().sum() >= MIN_SECTOR_SIZE:
            ranked.loc[members] = grp.rank(pct=True) * 100.0
    # universe-wide fallback for tickers in sectors too small to rank
    needs_fallback = ranked.isna() & raw.notna()
    if needs_fallback.any():
        uni = raw.rank(pct=True) * 100.0
        ranked.loc[needs_fallback] = uni.loc[needs_fallback]
    return ranked


def compute_factors(universe: list[str] | None = None) -> pd.DataFrame:
    """Return a DataFrame indexed by ticker: 6 factor scores + sector + price + composite.

    Only tickers with all 6 factors present are returned (drop, not impute).
    """
    if universe is None:
        universe = sp500_tickers()
    if not universe:
        raise RuntimeError("empty universe")

    prices = fetch_many(universe, period="2y", max_age_hours=6)
    spy_df = fetch_history("SPY", period="2y")
    spy_close = spy_df["Close"].dropna() if (spy_df is not None and not spy_df.empty) else pd.Series(dtype=float)
    fundamentals = get_fundamentals(universe)

    raw = pd.DataFrame(
        {
            "momentum_12_1": raw_momentum(prices),
            "rel_strength": raw_rel_strength(prices, spy_close),
            "low_volatility": raw_low_volatility(prices),
            "quality": raw_quality(fundamentals),
            "value": raw_value(fundamentals),
            "earnings_surprise": raw_earnings_surprise(fundamentals),
        }
    )
    sectors = fundamentals["sector"] if "sector" in fundamentals.columns else pd.Series(dtype=object)

    scores = pd.DataFrame(index=raw.index)
    for col in FACTOR_COLS:
        scores[col] = _sector_neutral_rank(raw[col], sectors)

    # drop, never impute — a ticker missing any factor is out of the ranking
    scores = scores.dropna(how="any", subset=FACTOR_COLS)
    if scores.empty:
        log.warning("no ticker has all 6 factors present")
        return scores

    scores["composite"] = sum(scores[c] * FACTOR_WEIGHTS[c] for c in FACTOR_COLS).clip(0, 100)
    scores["sector"] = sectors.reindex(scores.index).fillna("Unknown")

    last_price: dict[str, float] = {}
    for t in scores.index:
        df = prices.get(t)
        if df is not None and not df.empty and "Close" in df.columns:
            c = df["Close"].dropna()
            if not c.empty:
                last_price[t] = float(c.iloc[-1])
    scores["price"] = pd.Series(last_price).reindex(scores.index)
    scores = scores.dropna(subset=["price"])

    # keep raw values around for the inspector / debugging
    for col in FACTOR_COLS:
        scores[f"{col}_raw"] = raw[col].reindex(scores.index)

    return scores
