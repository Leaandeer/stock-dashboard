"""S&P 500 constituent list — Wikipedia scrape with static fallback."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

CACHE = Path(__file__).resolve().parent.parent / "data" / "sp500_tickers.csv"
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Static fallback — a representative subset; the Wikipedia scrape is preferred.
FALLBACK: list[str] = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "TSLA", "BRK-B", "JPM",
    "V", "JNJ", "WMT", "UNH", "PG", "MA", "HD", "XOM", "AVGO", "CVX",
    "LLY", "MRK", "ABBV", "PEP", "KO", "COST", "ADBE", "BAC", "CRM", "ORCL",
    "ACN", "MCD", "TMO", "ABT", "CSCO", "AMD", "DHR", "WFC", "DIS", "TXN",
    "INTC", "VZ", "NEE", "PM", "NFLX", "QCOM", "CMCSA", "RTX", "PFE", "T",
    "AMGN", "SPGI", "INTU", "UNP", "AMAT", "HON", "LIN", "LOW", "GS", "IBM",
    "BA", "ELV", "AXP", "SBUX", "CAT", "BLK", "DE", "NOW", "MDT", "GILD",
    "MS", "BKNG", "LMT", "SYK", "PLD", "ADI", "TJX", "C", "REGN", "ADP",
    "MMC", "ISRG", "VRTX", "MO", "ZTS", "SLB", "GE", "DUK", "SO", "EOG",
    "CI", "BSX", "PGR", "AON", "PNC", "BDX", "TGT", "CB", "MU", "EQIX",
    "APD", "FCX", "FDX", "USB", "NSC", "EW", "TFC", "ITW", "WM", "GD",
    "MNST", "CL", "ICE", "EMR", "MCK", "FISV", "MPC", "PSA", "ROP", "AEP",
    "F", "GM", "PYPL", "SHW", "MRNA", "KMB", "DG", "EL", "AIG", "ECL",
    "STZ", "MAR", "AFL", "TT", "ORLY", "TRV", "ROST", "PSX", "NXPI", "CTAS",
    "PAYX", "CMG", "MSCI", "ALL", "SRE", "PCAR", "WMB", "KHC", "FTNT", "AZO",
    "HUM", "WELL", "OXY", "DLR", "TEL", "VLO", "BIIB", "IDXX", "FAST", "ANET",
    "STT", "DOW", "HCA", "EXC", "PRU", "DLTR", "HSY", "XEL", "WEC", "CTSH",
    "ED", "AVB", "EBAY", "OTIS", "GIS", "KR", "WBA", "PEG", "DD", "RMD",
    "VRSK", "AME", "GPN", "ZBH", "ETN", "KEYS", "ADM", "GLW", "LHX", "VICI",
    "ON", "SBAC", "TSCO", "FANG", "WST", "WBD", "AMP", "ETR", "AWK", "CHTR",
    "STLD", "DXCM", "CDNS", "SNPS", "PANW", "CRWD", "DDOG", "TEAM", "MELI", "ABNB",
    "UBER", "LYFT", "ROKU", "SQ", "PINS", "SNAP", "TWLO", "DOCU", "OKTA", "ZS",
]


def _stale(path: Path, max_age_days: int = 7) -> bool:
    if not path.exists():
        return True
    age = datetime.now(timezone.utc) - datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return age > timedelta(days=max_age_days)


def sp500_tickers() -> list[str]:
    if not _stale(CACHE):
        try:
            df = pd.read_csv(CACHE)
            tickers = [str(s).strip() for s in df["Symbol"].tolist() if str(s).strip()]
            return [t.replace(".", "-") for t in tickers]
        except Exception:
            pass
    try:
        tables = pd.read_html(WIKI_URL)
        df = tables[0]
        sym_col = "Symbol" if "Symbol" in df.columns else df.columns[0]
        tickers = [str(s).strip() for s in df[sym_col].tolist() if str(s).strip()]
        tickers = [t.replace(".", "-") for t in tickers]
        try:
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            df[[sym_col]].rename(columns={sym_col: "Symbol"}).to_csv(CACHE, index=False)
        except Exception:
            pass
        return tickers
    except Exception as e:
        log.warning("Wikipedia scrape failed: %s — falling back to static list", e)
        return list(FALLBACK)
