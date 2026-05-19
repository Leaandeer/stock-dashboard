"""60/40 quant/Claude blend + rank-delta flags."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from analyst.analyzer import DEFAULT_MODEL, analyze

log = logging.getLogger(__name__)

RESULTS_PATH = Path(__file__).resolve().parent.parent / "data" / "analyst_results.json"


@dataclass
class BlendedRow:
    quant_rank: int
    new_rank: int
    rank_delta: int
    ticker: str
    price: float
    quant_score: float
    quant_score_10: float
    claude_score: float
    blended: float
    summary: str
    key_observations: list[str]
    sub_scores: dict
    financial_table: dict
    cached: bool


def _scale_to_10(quant_0_100: float) -> float:
    return round(quant_0_100 / 10.0, 2)


def run(scanner_results: dict, model: str = DEFAULT_MODEL, force: bool = False) -> dict:
    candidates = scanner_results.get("candidates", [])
    rows: list[BlendedRow] = []
    errors: list[dict] = []

    for c in candidates:
        ticker = c["ticker"]
        try:
            ar = analyze(ticker, model=model, force=force)
            payload = ar.payload
            claude_overall = float(payload.get("overall", 5.0))
            quant_10 = _scale_to_10(c["composite"])
            blended = round(0.60 * quant_10 + 0.40 * claude_overall, 2)
            rows.append(
                BlendedRow(
                    quant_rank=c["rank"],
                    new_rank=0,
                    rank_delta=0,
                    ticker=ticker,
                    price=c["price"],
                    quant_score=c["composite"],
                    quant_score_10=quant_10,
                    claude_score=claude_overall,
                    blended=blended,
                    summary=payload.get("summary", ""),
                    key_observations=payload.get("key_observations", []),
                    sub_scores=payload.get("scores", {}),
                    financial_table=payload.get("_financial_table", {}),
                    cached=ar.cached,
                )
            )
        except Exception as e:
            log.warning("analyst failed for %s: %s", ticker, e)
            errors.append({"ticker": ticker, "error": str(e)})

    rows.sort(key=lambda r: r.blended, reverse=True)
    for new_idx, r in enumerate(rows, start=1):
        r.new_rank = new_idx
        r.rank_delta = r.quant_rank - r.new_rank

    out = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "model": model,
        "rows": [asdict(r) for r in rows],
        "errors": errors,
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
