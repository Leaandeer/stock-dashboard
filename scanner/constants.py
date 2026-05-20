"""Scanner configuration constants (Phase 2)."""
from __future__ import annotations

# Composite threshold — scanner returns candidates at or above this score.
# No longer shifted by the macro gate (Phase 2, Part C): the gate failed
# walk-forward validation, so it is informational context, not a hard gate.
COMPOSITE_THRESHOLD = 65.0

# Max candidates surfaced / snapshotted.
TOP_N = 30

# Final factor set after the Phase 2 refactor.
# Removed: Volume Surge, Short Interest Decline (Part B.1).
# Removed: 52-Week High Proximity — Part A showed a negative spread (-16 bps).
# Replaced 52W High with Low Volatility (Part B.4 conditional).
FACTOR_COLS = [
    "momentum_12_1",
    "rel_strength",
    "low_volatility",
    "quality",
    "value",
    "earnings_surprise",
]

FACTOR_DISPLAY = {
    "momentum_12_1": "MOMENTUM 12-1",
    "rel_strength": "REL STRENGTH",
    "low_volatility": "LOW VOL",
    "quality": "QUALITY",
    "value": "VALUE",
    "earnings_surprise": "EARN SURPRISE",
}

# Equal weight across the final factors (Part B.5). The plan text said "5
# factors @ 0.20"; the plan's own add/remove instructions actually yield 6
# factors, so equal weight is 1/6 each. Revisit per-factor weights after the
# Part A re-validation if a factor shows a materially weak spread.
FACTOR_WEIGHTS = {f: 1.0 / len(FACTOR_COLS) for f in FACTOR_COLS}

# Sector-neutral ranking: sectors with fewer than this many scored names fall
# back to a universe-wide percentile rank (a 2-name sector can't be ranked).
MIN_SECTOR_SIZE = 8
