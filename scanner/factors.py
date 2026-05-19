"""5 scanner factors — each returns a 0-100 score per ticker."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from utils.data_fetch import fetch_history, fetch_info, fetch_many

log = logging.getLogger(__name__)


@dataclass
class FactorRow:
    ticker: str
    price: float
    momentum: float = 50.0
    volume_surge: float = 50.0
    rel_strength: float = 50.0
    high_proximity: float = 50.0
    short_decline: float = 50.0
    extras: dict = field(default_factory=dict)


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, float(x)))


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _percentile_rank(values: pd.Series) -> pd.Series:
    return values.rank(pct=True) * 100.0


def compute_factors(
    universe: Iterable[str],
    spy_history: pd.DataFrame | None = None,
) -> list[FactorRow]:
    universe = [t for t in universe if t]
    prices = fetch_many(universe, period="1y", max_age_hours=6)
    if spy_history is None or spy_history.empty:
        spy_history = fetch_history("SPY", period="1y")
    spy_close = spy_history["Close"].dropna() if (spy_history is not None and not spy_history.empty) else pd.Series(dtype=float)

    # --- intermediate per-ticker raw values ---
    raw_momentum_gap: dict[str, float] = {}
    raw_3m_return: dict[str, float] = {}
    raw_recent_cross: dict[str, bool] = {}
    raw_vol_ratio: dict[str, float] = {}
    raw_rs_spread: dict[str, float] = {}
    raw_high_prox: dict[str, float] = {}
    prices_last: dict[str, float] = {}

    spy_20d = None
    if len(spy_close) >= 21:
        spy_20d = float(spy_close.iloc[-1] / spy_close.iloc[-21] - 1.0)

    for t in universe:
        df = prices.get(t)
        if df is None or df.empty or "Close" not in df.columns:
            continue
        c = df["Close"].dropna()
        if len(c) < 60:
            continue
        prices_last[t] = float(c.iloc[-1])

        ema10 = _ema(c, 10)
        ema50 = _ema(c, 50)
        gap = float((ema10.iloc[-1] - ema50.iloc[-1]) / ema50.iloc[-1])
        raw_momentum_gap[t] = gap

        if len(c) >= 65:
            raw_3m_return[t] = float(c.iloc[-1] / c.iloc[-63] - 1.0)
        else:
            raw_3m_return[t] = 0.0

        recent = False
        if len(ema10) >= 6 and len(ema50) >= 6:
            cross_now = ema10.iloc[-1] > ema50.iloc[-1]
            cross_5ago = ema10.iloc[-6] > ema50.iloc[-6]
            recent = bool(cross_now and not cross_5ago)
        raw_recent_cross[t] = recent

        if "Volume" in df.columns:
            vol = df["Volume"].dropna()
            if len(vol) >= 21:
                ratio = float(vol.tail(5).mean() / max(vol.tail(20).mean(), 1.0))
                raw_vol_ratio[t] = ratio

        if len(c) >= 21 and spy_20d is not None:
            stock_20d = float(c.iloc[-1] / c.iloc[-21] - 1.0)
            raw_rs_spread[t] = stock_20d - spy_20d

        if len(c) >= 252:
            hi = float(c.tail(252).max())
            raw_high_prox[t] = float(c.iloc[-1] / hi) if hi else 0.0
        else:
            hi = float(c.max())
            raw_high_prox[t] = float(c.iloc[-1] / hi) if hi else 0.0

    # --- transform to 0-100 scores ---
    mom_gap_s = pd.Series(raw_momentum_gap)
    ret_3m_s = pd.Series(raw_3m_return)
    rs_spread_s = pd.Series(raw_rs_spread)
    hi_prox_s = pd.Series(raw_high_prox)
    vol_ratio_s = pd.Series(raw_vol_ratio)

    gap_rank = _percentile_rank(mom_gap_s) if not mom_gap_s.empty else pd.Series(dtype=float)
    ret_rank = _percentile_rank(ret_3m_s) if not ret_3m_s.empty else pd.Series(dtype=float)
    rs_rank = _percentile_rank(rs_spread_s) if not rs_spread_s.empty else pd.Series(dtype=float)

    rows: list[FactorRow] = []
    for t in universe:
        if t not in prices_last:
            continue
        # Momentum: blend gap percentile + 3m return percentile, with crossover bonus
        gap_score = float(gap_rank.get(t, 50.0))
        ret_score = float(ret_rank.get(t, 50.0))
        base = 0.5 * gap_score + 0.5 * ret_score
        if raw_recent_cross.get(t, False):
            base = min(100.0, base + 10.0)
        elif gap_score < 50:
            base = max(0.0, base - 5.0)
        momentum = _clamp(base)

        # Volume surge — explicit linear map
        vr = vol_ratio_s.get(t)
        if vr is None or np.isnan(vr):
            volume_surge = 50.0
        else:
            volume_surge = _clamp((float(vr) - 0.7) / (2.0 - 0.7) * 100.0)

        # Relative strength — percentile rank of spread
        rel_strength = float(rs_rank.get(t, 50.0))

        # 52w high proximity — > 0.95 scores highest, scaled
        hp = hi_prox_s.get(t, 0.5)
        high_proximity = _clamp((float(hp) - 0.5) / (0.95 - 0.5) * 100.0)

        # Short decline — pulled per-ticker via info (best-effort)
        short_decline = 50.0
        try:
            info = fetch_info(t)
            cur = info.get("sharesShort")
            prev = info.get("sharesShortPriorMonth")
            if cur and prev and prev > 0:
                delta = (cur - prev) / prev  # negative = decline = bullish
                short_decline = _clamp((-delta + 0.2) / 0.4 * 100.0)
        except Exception:
            pass

        rows.append(
            FactorRow(
                ticker=t,
                price=prices_last[t],
                momentum=round(momentum, 1),
                volume_surge=round(volume_surge, 1),
                rel_strength=round(rel_strength, 1),
                high_proximity=round(high_proximity, 1),
                short_decline=round(short_decline, 1),
                extras={
                    "ema_gap_pct": round(raw_momentum_gap.get(t, 0.0) * 100, 2),
                    "ret_3m_pct": round(raw_3m_return.get(t, 0.0) * 100, 2),
                    "rs_spread_pct": round(rs_spread_s.get(t, 0.0) * 100, 2) if t in rs_spread_s else None,
                    "high_proximity_ratio": round(raw_high_prox.get(t, 0.0), 3),
                    "vol_ratio_5_20": round(raw_vol_ratio.get(t, 1.0), 2) if t in raw_vol_ratio else None,
                    "recent_crossover": raw_recent_cross.get(t, False),
                },
            )
        )
    return rows
