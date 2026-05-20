"""Composite ranking.

Phase 2, Part C — the macro gate no longer shifts the scanner threshold or
disables it. Walk-forward validation showed the gate is not predictive, so it
is informational context only. The scanner always runs with one fixed
threshold regardless of macro zone.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from scanner.constants import COMPOSITE_THRESHOLD, FACTOR_COLS, TOP_N
from scanner.factors import compute_factors
from signals.composite import MacroState

log = logging.getLogger(__name__)

RESULTS_PATH = Path(__file__).resolve().parent.parent / "data" / "scanner_results.json"


@dataclass
class Candidate:
    rank: int
    ticker: str
    price: float
    composite: float
    sector: str
    momentum_12_1: float
    rel_strength: float
    low_volatility: float
    quality: float
    value: float
    earnings_surprise: float


def run(macro: MacroState | None = None, threshold: float = COMPOSITE_THRESHOLD) -> dict:
    """Compute factors, rank, return candidates at or above the threshold.

    `macro` is recorded for display context only — it does not gate anything.
    """
    factors_df = compute_factors()
    candidates: list[Candidate] = []

    if not factors_df.empty:
        ranked = factors_df.sort_values("composite", ascending=False)
        passing = ranked[ranked["composite"] >= threshold].head(TOP_N)
        for rank, (ticker, row) in enumerate(passing.iterrows(), start=1):
            candidates.append(
                Candidate(
                    rank=rank,
                    ticker=ticker,
                    price=round(float(row["price"]), 2),
                    composite=round(float(row["composite"]), 1),
                    sector=str(row.get("sector", "Unknown")),
                    momentum_12_1=round(float(row["momentum_12_1"]), 1),
                    rel_strength=round(float(row["rel_strength"]), 1),
                    low_volatility=round(float(row["low_volatility"]), 1),
                    quality=round(float(row["quality"]), 1),
                    value=round(float(row["value"]), 1),
                    earnings_surprise=round(float(row["earnings_surprise"]), 1),
                )
            )

    out = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "threshold": threshold,
        "scored_universe": int(len(factors_df)),
        "macro_zone": macro.zone if macro else None,
        "macro_score": macro.score if macro else None,
        "factor_cols": FACTOR_COLS,
        "candidates": [asdict(c) for c in candidates],
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(out, indent=2))

    # Snapshot top-N picks into the forward-performance track record.
    try:
        from tracking.performance import snapshot_scanner

        snapshot_scanner(out)
    except Exception as e:  # never let tracking break a scan
        log.warning("scanner snapshot failed: %s", e)

    return out


def load() -> dict | None:
    if not RESULTS_PATH.exists():
        return None
    try:
        return json.loads(RESULTS_PATH.read_text())
    except Exception:
        return None
