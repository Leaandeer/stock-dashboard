"""yfinance wrappers with parquet caching keyed by trading day."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PRICE_DIR = ROOT / "data" / "prices"
PRICE_DIR.mkdir(parents=True, exist_ok=True)


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _cache_path(ticker: str, period: str) -> Path:
    safe = ticker.replace("^", "_").replace("/", "_")
    return PRICE_DIR / f"{safe}__{period}.parquet"


def _is_fresh(path: Path, max_age_hours: int = 6) -> bool:
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return datetime.now(timezone.utc) - mtime < timedelta(hours=max_age_hours)


def fetch_history(ticker: str, period: str = "2y", max_age_hours: int = 6) -> pd.DataFrame:
    """Download daily OHLCV; cache to parquet."""
    path = _cache_path(ticker, period)
    if _is_fresh(path, max_age_hours):
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    try:
        df = yf.Ticker(ticker).history(period=period, auto_adjust=False)
    except Exception as e:
        log.warning("fetch_history failed for %s: %s", ticker, e)
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    try:
        df.to_parquet(path)
    except Exception as e:
        log.warning("parquet write failed for %s: %s", ticker, e)
    return df


def fetch_many(tickers: Iterable[str], period: str = "1y", max_age_hours: int = 6) -> dict[str, pd.DataFrame]:
    """Batch fetch using yf.download for speed; falls back to per-ticker on miss."""
    tickers = [t for t in tickers if t]
    if not tickers:
        return {}
    out: dict[str, pd.DataFrame] = {}
    need: list[str] = []
    for t in tickers:
        p = _cache_path(t, period)
        if _is_fresh(p, max_age_hours):
            try:
                out[t] = pd.read_parquet(p)
                continue
            except Exception:
                pass
        need.append(t)
    if not need:
        return out
    try:
        bulk = yf.download(
            tickers=" ".join(need),
            period=period,
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
    except Exception as e:
        log.warning("yf.download bulk failed: %s", e)
        bulk = None
    if bulk is None or bulk.empty:
        for t in need:
            out[t] = fetch_history(t, period=period, max_age_hours=max_age_hours)
        return out
    if isinstance(bulk.columns, pd.MultiIndex):
        for t in need:
            if t in bulk.columns.get_level_values(0):
                df = bulk[t].dropna(how="all").copy()
                if not df.empty:
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                    out[t] = df
                    try:
                        df.to_parquet(_cache_path(t, period))
                    except Exception:
                        pass
                else:
                    out[t] = pd.DataFrame()
            else:
                out[t] = pd.DataFrame()
    else:
        t = need[0]
        df = bulk.copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        out[t] = df
        try:
            df.to_parquet(_cache_path(t, period))
        except Exception:
            pass
    return out


def fetch_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception as e:
        log.warning("fetch_info failed for %s: %s", ticker, e)
        return {}


def fetch_financials(ticker: str) -> dict:
    """Return income/cashflow/balance-sheet quarterly frames."""
    try:
        t = yf.Ticker(ticker)
        return {
            "income": t.quarterly_financials if hasattr(t, "quarterly_financials") else pd.DataFrame(),
            "cashflow": t.quarterly_cashflow if hasattr(t, "quarterly_cashflow") else pd.DataFrame(),
            "balance": t.quarterly_balance_sheet if hasattr(t, "quarterly_balance_sheet") else pd.DataFrame(),
            "info": fetch_info(ticker),
        }
    except Exception as e:
        log.warning("fetch_financials failed for %s: %s", ticker, e)
        return {"income": pd.DataFrame(), "cashflow": pd.DataFrame(), "balance": pd.DataFrame(), "info": {}}


def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
