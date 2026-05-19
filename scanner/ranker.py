"""Composite ranking + macro-gated threshold."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from scanner.factors import FactorRow, compute_factors
from scanner.universe import sp500_tickers
from signals.composite import MacroState

RESULTS_PATH = Path(__file__).resolve().parent.parent / "data" / "scanner_results.json"

WEIGHTS = {
    "momentum": 0.20,
    "volume_surge": 0.20,
    "rel_strength": 0.20,
    "high_proximity": 0.20,
    "short_decline": 0.20,
}


@dataclass
class Candidate:
    rank: int
    ticker: str
    price: float
    composite: float
    momentum: float
    volume_surge: float
    rel_strength: float
    high_proximity: float
    short_decline: float
    extras: dict


def _row_score(r: FactorRow) -> float:
    return (
        WEIGHTS["momentum"] * r.momentum
        + WEIGHTS["volume_surge"] * r.volume_surge
        + WEIGHTS["rel_strength"] * r.rel_strength
        + WEIGHTS["high_proximity"] * r.high_proximity
        + WEIGHTS["short_decline"] * r.short_decline
    )


def threshold_for(macro: MacroState | None) -> float | None:
    """Return the composite threshold based on macro zone. None = scanner disabled."""
    if macro is None:
        return 65.0
    if macro.zone == "DEFENSIVE":
        return None
    if macro.zone == "REDUCED":
        return 75.0
    return 65.0  # FULL DEPLOY


def run(macro: MacroState | None = None) -> dict:
    threshold = threshold_for(macro)
    if threshold is None:
        out = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "threshold": None,
            "macro_zone": macro.zone if macro else None,
            "macro_score": macro.score if macro else None,
            "candidates": [],
            "note": "scanner disabled: macro zone DEFENSIVE",
        }
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(out, indent=2))
        return out

    universe = sp500_tickers()
    rows = compute_factors(universe)
    scored = [(r, _row_score(r)) for r in rows]
    scored.sort(key=lambda kv: kv[1], reverse=True)

    candidates: list[Candidate] = []
    rank = 0
    for r, s in scored:
        if s < threshold:
            continue
        rank += 1
        candidates.append(
            Candidate(
                rank=rank,
                ticker=r.ticker,
                price=r.price,
                composite=round(s, 1),
                momentum=r.momentum,
                volume_surge=r.volume_surge,
                rel_strength=r.rel_strength,
                high_proximity=r.high_proximity,
                short_decline=r.short_decline,
                extras=r.extras,
            )
        )

    out = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "threshold": threshold,
        "macro_zone": macro.zone if macro else None,
        "macro_score": macro.score if macro else None,
        "candidates": [asdict(c) for c in candidates],
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(out, indent=2))
    return out


def load() -> dict | None:
    if not RESULTS_PATH.exists():
        return None
    try:
        return json.loads(RESULTS_PATH.read_text())
    except Exception:
        return None
