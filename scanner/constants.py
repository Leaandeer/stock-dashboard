"""Scanner configuration constants (Phase 2)."""
from __future__ import annotations

# Composite threshold — scanner returns candidates at or above this score.
# No longer shifted by the macro gate (Phase 2, Part C): the gate failed
# walk-forward validation, so it is informational context, not a hard gate.
COMPOSITE_THRESHOLD = 65.0

# Max candidates surfaced / snapshotted.
TOP_N = 30

# Final factor set after the Phase 2 refactor + re-validation.
# Removed: Volume Surge, Short Interest Decline (Part B.1).
# Removed: 52-Week High Proximity — Part A showed a negative spread (-16 bps).
# Removed: Low Volatility — the Part A re-run showed it too is non-predictive
#   (-7 bps, 48% win rate), and equal-weighting it dragged the whole composite
#   negative. Dropped rather than carried as dead weight.
FACTOR_COLS = [
    "momentum_12_1",
    "rel_strength",
    "quality",
    "value",
    "earnings_surprise",
]

FACTOR_DISPLAY = {
    "momentum_12_1": "MOMENTUM 12-1",
    "rel_strength": "REL STRENGTH",
    "quality": "QUALITY",
    "value": "VALUE",
    "earnings_surprise": "EARN SURPRISE",
}

# Momentum-tilted weights — driven by the Part A re-validation, not equal
# weight. 12-1 Momentum was by far the strongest factor (+95 bps per 20d);
# Relative Strength was solid (+38 bps). Quality / Value / Earnings Surprise
# cannot be point-in-time backtested with yfinance, so they are kept as
# fundamental ballast at a smaller, untested weight. Weights sum to 1.0.
FACTOR_WEIGHTS = {
    "momentum_12_1": 0.35,
    "rel_strength": 0.20,
    "quality": 0.15,
    "value": 0.15,
    "earnings_surprise": 0.15,
}

# Sector-neutral ranking: sectors with fewer than this many scored names fall
# back to a universe-wide percentile rank (a 2-name sector can't be ranked).
MIN_SECTOR_SIZE = 8
