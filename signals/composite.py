"""Layer 1 composite — weighted blend + zone classification."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from signals import breadth, credit_spreads, crowding, put_call, vix_level, vix_term_structure

STATE_PATH = Path(__file__).resolve().parent.parent / "data" / "macro_state.json"

WEIGHTS: dict[str, float] = {
    "vix_level": 0.25,
    "vix_term_structure": 0.20,
    "breadth": 0.20,
    "credit_spreads": 0.15,
    "put_call": 0.10,
    "crowding": 0.10,
}

LABELS: dict[str, str] = {
    "vix_level": "VIX Level",
    "vix_term_structure": "VIX Term Structure",
    "breadth": "Market Breadth",
    "credit_spreads": "Credit Spreads",
    "put_call": "Put/Call",
    "crowding": "Factor Crowding",
}

SIGNAL_FUNCS: dict[str, Callable] = {
    "vix_level": vix_level.compute,
    "vix_term_structure": vix_term_structure.compute,
    "breadth": breadth.compute,
    "credit_spreads": credit_spreads.compute,
    "put_call": put_call.compute,
    "crowding": crowding.compute,
}


@dataclass
class MacroState:
    score: float
    zone: str
    sizing_pct: int
    signals: dict[str, dict] = field(default_factory=dict)
    timestamp: str = ""


def classify(score: float) -> tuple[str, int]:
    if score >= 70:
        return "FULL DEPLOY", 100
    if score >= 40:
        return "REDUCED", 60
    return "DEFENSIVE", 25


def compute(skip: tuple[str, ...] = ()) -> MacroState:
    sigs: dict[str, dict] = {}
    total = 0.0
    wsum = 0.0
    for key, fn in SIGNAL_FUNCS.items():
        if key in skip:
            continue
        try:
            r = fn()
            sigs[key] = {"score": r.score, "raw": r.raw, "weight": WEIGHTS[key], "label": LABELS[key]}
            total += r.score * WEIGHTS[key]
            wsum += WEIGHTS[key]
        except Exception as e:
            sigs[key] = {"score": 50.0, "raw": {"error": str(e)}, "weight": WEIGHTS[key], "label": LABELS[key]}
            total += 50.0 * WEIGHTS[key]
            wsum += WEIGHTS[key]
    composite = total / wsum if wsum else 50.0
    zone, sizing = classify(composite)
    return MacroState(
        score=round(composite, 1),
        zone=zone,
        sizing_pct=sizing,
        signals=sigs,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )


def save(state: MacroState) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(asdict(state), default=str, indent=2))


def load() -> MacroState | None:
    if not STATE_PATH.exists():
        return None
    try:
        d = json.loads(STATE_PATH.read_text())
        return MacroState(**d)
    except Exception:
        return None
