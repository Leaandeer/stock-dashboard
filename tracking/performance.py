"""Forward-performance tracking — the live, growing track record.

Every scanner run snapshots its top-N picks. A nightly job then computes the
1d/5d/20d forward return of every snapshot ever taken, vs. SPY over the same
window. This is how you tell whether the composite is predictive or whether
you are tuning to noise.

The DB lives at data/track_record.db and is intentionally NOT gitignored: it
is the asset. Run the nightly job locally (cron) and commit the file so the
record survives Streamlit Cloud restarts.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from utils.data_fetch import fetch_history, fetch_many

log = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "track_record.db"
HORIZONS = {"1d": 1, "5d": 5, "20d": 20}


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.execute(
        """CREATE TABLE IF NOT EXISTS scanner_snapshots (
            snapshot_date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            rank INTEGER,
            composite REAL,
            price_at_snapshot REAL,
            sector TEXT,
            momentum_12_1 REAL,
            rel_strength REAL,
            low_volatility REAL,
            quality REAL,
            value REAL,
            earnings_surprise REAL,
            macro_zone TEXT,
            macro_score REAL,
            created_at TEXT,
            PRIMARY KEY (snapshot_date, ticker)
        )"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS snapshot_returns (
            snapshot_date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            base_price REAL,
            ret_1d REAL,
            ret_5d REAL,
            ret_20d REAL,
            spy_1d REAL,
            spy_5d REAL,
            spy_20d REAL,
            updated_at TEXT,
            PRIMARY KEY (snapshot_date, ticker)
        )"""
    )
    return c


def snapshot_scanner(results: dict, top_n: int = 20) -> int:
    """Persist the top-N scanner candidates as a dated snapshot."""
    candidates = results.get("candidates", [])
    if not candidates:
        return 0
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    now = datetime.now(timezone.utc).isoformat()
    macro_zone = results.get("macro_zone")
    macro_score = results.get("macro_score")
    conn = _conn()
    try:
        for c in candidates[:top_n]:
            conn.execute(
                "INSERT OR REPLACE INTO scanner_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    date, c["ticker"], c["rank"], c["composite"], c["price"],
                    c.get("sector", "Unknown"),
                    c["momentum_12_1"], c["rel_strength"], c["low_volatility"],
                    c["quality"], c["value"], c["earnings_surprise"],
                    macro_zone, macro_score, now,
                ),
            )
        conn.commit()
        return min(len(candidates), top_n)
    finally:
        conn.close()


def _fwd_return(close: pd.Series, snap_ts: pd.Timestamp, n: int) -> tuple[float | None, float | None]:
    """Close-to-close return n trading days after the snapshot date.

    Returns (base_price, forward_return). forward_return is None until n
    trading days have actually elapsed.
    """
    if close is None or close.empty:
        return None, None
    pos = close.index.searchsorted(snap_ts, side="right") - 1
    if pos < 0:
        return None, None
    base = float(close.iloc[pos])
    if base == 0 or pos + n >= len(close):
        return base, None
    fwd = float(close.iloc[pos + n])
    return base, (fwd / base - 1.0)


def update_returns(max_age_hours: int = 6) -> dict:
    """Nightly job — (re)compute forward returns for every snapshot."""
    conn = _conn()
    try:
        snaps = pd.read_sql_query(
            "SELECT snapshot_date, ticker FROM scanner_snapshots", conn
        )
    finally:
        conn.close()
    if snaps.empty:
        return {"snapshots": 0, "updated": 0}

    tickers = sorted(snaps["ticker"].unique().tolist())
    prices = fetch_many(tickers, period="2y", max_age_hours=max_age_hours)
    spy_df = fetch_history("SPY", period="2y", max_age_hours=max_age_hours)
    spy_close = spy_df["Close"].dropna() if (spy_df is not None and not spy_df.empty) else pd.Series(dtype=float)

    now = datetime.now(timezone.utc).isoformat()
    updated = 0
    conn = _conn()
    try:
        for _, row in snaps.iterrows():
            date_str = row["snapshot_date"]
            ticker = row["ticker"]
            snap_ts = pd.Timestamp(date_str)
            df = prices.get(ticker)
            close = df["Close"].dropna() if (df is not None and not df.empty and "Close" in df.columns) else pd.Series(dtype=float)

            rets: dict[str, float | None] = {}
            spys: dict[str, float | None] = {}
            base_price = None
            for label, n in HORIZONS.items():
                base_price, r = _fwd_return(close, snap_ts, n)
                rets[label] = r
                _, sr = _fwd_return(spy_close, snap_ts, n)
                spys[label] = sr

            conn.execute(
                "INSERT OR REPLACE INTO snapshot_returns VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    date_str, ticker, base_price,
                    rets["1d"], rets["5d"], rets["20d"],
                    spys["1d"], spys["5d"], spys["20d"], now,
                ),
            )
            updated += 1
        conn.commit()
    finally:
        conn.close()
    return {"snapshots": len(snaps), "updated": updated}


def load_track_record() -> pd.DataFrame:
    """Snapshots joined with their (possibly partial) forward returns."""
    conn = _conn()
    try:
        df = pd.read_sql_query(
            """SELECT s.snapshot_date, s.ticker, s.rank, s.composite,
                      s.macro_zone, s.macro_score,
                      r.ret_1d, r.ret_5d, r.ret_20d,
                      r.spy_1d, r.spy_5d, r.spy_20d
               FROM scanner_snapshots s
               LEFT JOIN snapshot_returns r
                 ON s.snapshot_date = r.snapshot_date AND s.ticker = r.ticker
               ORDER BY s.snapshot_date DESC, s.rank ASC""",
            conn,
        )
    finally:
        conn.close()
    return df


def summary_by_bucket(df: pd.DataFrame, horizon: str = "20d") -> pd.DataFrame:
    """Avg forward / excess return + hit rate, bucketed by composite score.

    A predictive composite shows monotonically rising excess return across
    higher score buckets. A flat or inverted table means the score is noise.
    """
    ret_col = f"ret_{horizon}"
    spy_col = f"spy_{horizon}"
    if ret_col not in df.columns:
        return pd.DataFrame()
    d = df.dropna(subset=[ret_col, spy_col]).copy()
    if d.empty:
        return pd.DataFrame()
    d["excess"] = d[ret_col] - d[spy_col]
    d["bucket"] = pd.cut(
        d["composite"],
        bins=[0, 65, 75, 85, 100.01],
        labels=["<65", "65-75", "75-85", "85+"],
    )
    g = d.groupby("bucket", observed=True)
    return pd.DataFrame(
        {
            "picks": g.size(),
            f"avg_{horizon}_return_%": (g[ret_col].mean() * 100).round(2),
            f"avg_{horizon}_excess_%": (g["excess"].mean() * 100).round(2),
            "beat_spy_%": (g["excess"].apply(lambda s: (s > 0).mean()) * 100).round(1),
        }
    )


def overall_stats(df: pd.DataFrame, horizon: str = "20d") -> dict:
    ret_col = f"ret_{horizon}"
    spy_col = f"spy_{horizon}"
    total_snaps = len(df)
    if ret_col not in df.columns:
        return {"total_picks": total_snaps, "matured": 0}
    d = df.dropna(subset=[ret_col, spy_col]).copy()
    if d.empty:
        return {"total_picks": total_snaps, "matured": 0}
    d["excess"] = d[ret_col] - d[spy_col]
    return {
        "total_picks": total_snaps,
        "matured": len(d),
        "avg_return_pct": round(d[ret_col].mean() * 100, 2),
        "avg_excess_pct": round(d["excess"].mean() * 100, 2),
        "beat_spy_pct": round((d["excess"] > 0).mean() * 100, 1),
        "snapshot_days": int(df["snapshot_date"].nunique()),
    }
