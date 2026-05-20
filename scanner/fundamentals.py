"""Fundamentals fetch + cache for the Phase 2 factors.

Pulls per-ticker fundamentals from yfinance — enterprise value, EBITDA, sector
(from `info`), gross profit + total assets (from quarterly statements), and the
last two quarters' EPS surprise (from `earnings_dates`).

Cached to data/fundamentals_cache.parquet, one row per ticker, refreshed once
per day — fundamentals change quarterly, so intraday re-fetching is waste.

If fundamentals fail outright for more than 30% of the universe, this raises
instead of producing a half-empty composite (Part B data note).
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

CACHE = Path(__file__).resolve().parent.parent / "data" / "fundamentals_cache.parquet"
COLUMNS = [
    "fetched_date", "sector", "enterprise_value", "ebitda",
    "gross_profit_ttm", "total_assets", "surprise_recent", "surprise_prior",
]
FETCH_SLEEP = 0.1  # politeness delay between per-ticker calls


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _row_lookup(df: pd.DataFrame, names: list[str]) -> pd.Series | None:
    if df is None or df.empty:
        return None
    for name in names:
        for idx in df.index:
            if isinstance(idx, str) and idx.strip().lower() == name.lower():
                return df.loc[idx]
    return None


def _fetch_one(ticker: str) -> dict:
    """Best-effort fundamentals for a single ticker. Missing pieces stay NaN."""
    row: dict[str, object] = {
        "fetched_date": _today(), "sector": None,
        "enterprise_value": np.nan, "ebitda": np.nan,
        "gross_profit_ttm": np.nan, "total_assets": np.nan,
        "surprise_recent": np.nan, "surprise_prior": np.nan,
    }
    t = yf.Ticker(ticker)

    try:
        info = t.info or {}
        row["sector"] = info.get("sector")
        ev = info.get("enterpriseValue")
        eb = info.get("ebitda")
        row["enterprise_value"] = float(ev) if ev is not None else np.nan
        row["ebitda"] = float(eb) if eb is not None else np.nan
    except Exception as e:
        log.debug("info failed for %s: %s", ticker, e)

    try:
        inc = getattr(t, "quarterly_income_stmt", None)
        if inc is None or (hasattr(inc, "empty") and inc.empty):
            inc = getattr(t, "quarterly_financials", None)
        gp = _row_lookup(inc, ["Gross Profit"])
        if gp is not None:
            gp = gp.dropna().sort_index(ascending=False)
            if len(gp) >= 4:
                row["gross_profit_ttm"] = float(gp.iloc[:4].sum())
    except Exception as e:
        log.debug("income_stmt failed for %s: %s", ticker, e)

    try:
        bal = getattr(t, "quarterly_balance_sheet", None)
        ta = _row_lookup(bal, ["Total Assets"])
        if ta is not None:
            ta = ta.dropna().sort_index()
            if not ta.empty:
                row["total_assets"] = float(ta.iloc[-1])
    except Exception as e:
        log.debug("balance_sheet failed for %s: %s", ticker, e)

    try:
        ed = t.get_earnings_dates(limit=12)
        if ed is not None and not ed.empty:
            est_col = next((c for c in ed.columns if "estimate" in c.lower()), None)
            act_col = next((c for c in ed.columns if "reported" in c.lower()), None)
            if est_col and act_col:
                rep = ed[[est_col, act_col]].dropna().sort_index(ascending=False)
                surprises: list[float] = []
                for _, r in rep.iterrows():
                    est = float(r[est_col])
                    act = float(r[act_col])
                    if est != 0:
                        surprises.append((act - est) / abs(est) * 100.0)
                    if len(surprises) == 2:
                        break
                if len(surprises) >= 1:
                    row["surprise_recent"] = surprises[0]
                if len(surprises) >= 2:
                    row["surprise_prior"] = surprises[1]
    except Exception as e:
        log.debug("earnings_dates failed for %s: %s", ticker, e)

    return row


def get_fundamentals(universe: list[str], halt_threshold: float = 0.30) -> pd.DataFrame:
    """Fundamentals for the universe, served from cache where fresh."""
    universe = [t for t in universe if t]
    if not universe:
        return pd.DataFrame(columns=COLUMNS)

    cache = pd.DataFrame(columns=COLUMNS)
    if CACHE.exists():
        try:
            cache = pd.read_parquet(CACHE)
        except Exception:
            cache = pd.DataFrame(columns=COLUMNS)

    today = _today()
    fresh = set()
    if not cache.empty and "fetched_date" in cache.columns:
        fresh = set(cache.index[cache["fetched_date"] == today])

    stale = [t for t in universe if t not in fresh]
    if stale:
        log.info("fetching fundamentals for %d tickers (%d cached)...", len(stale), len(universe) - len(stale))
        fetched: dict[str, dict] = {}
        for i, ticker in enumerate(stale):
            fetched[ticker] = _fetch_one(ticker)
            time.sleep(FETCH_SLEEP)
            if (i + 1) % 50 == 0:
                log.info("  ...%d/%d", i + 1, len(stale))
        new_df = pd.DataFrame.from_dict(fetched, orient="index")
        cache = pd.concat([cache[~cache.index.isin(new_df.index)], new_df])
        try:
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            cache.to_parquet(CACHE)
        except Exception as e:
            log.warning("fundamentals cache write failed: %s", e)

    result = cache.reindex(universe)

    # halt if fundamentals failed outright for too much of the universe
    total_fail = result[["enterprise_value", "gross_profit_ttm", "surprise_recent"]].isna().all(axis=1)
    fail_rate = float(total_fail.mean())
    if fail_rate > halt_threshold:
        raise RuntimeError(
            f"fundamentals failed for {fail_rate:.0%} of the universe "
            f"(> {halt_threshold:.0%}) — halting instead of ranking on partial data"
        )
    if fail_rate > 0:
        log.warning("fundamentals: %.0f%% of universe has no usable data (dropped)", fail_rate * 100)
    return result
