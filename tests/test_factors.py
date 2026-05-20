"""Unit tests for the Phase 2 factor set — synthetic input, known output."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from scanner.constants import FACTOR_COLS, FACTOR_WEIGHTS
from scanner.factors import _sector_neutral_rank
from scanner.factors.earnings_surprise import raw_earnings_surprise
from scanner.factors.momentum import raw_momentum
from scanner.factors.quality import raw_quality
from scanner.factors.relative_strength import raw_rel_strength
from scanner.factors.value import raw_value


def _price_df(values: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2023-01-02", periods=len(values), freq="B")
    return pd.DataFrame({"Close": values, "Volume": [1_000_000] * len(values)}, index=idx)


# --- 12-1 momentum -----------------------------------------------------

def test_momentum_12_1_known_value():
    closes = [100.0] * 300
    closes[300 - 252] = 100.0   # price 12 months ago
    closes[300 - 21] = 130.0    # price 1 month ago
    prices = {"AAA": _price_df(closes)}
    out = raw_momentum(prices)
    assert out["AAA"] == pytest.approx(0.30)


def test_momentum_skips_short_history():
    prices = {"BBB": _price_df([100.0] * 100)}
    assert "BBB" not in raw_momentum(prices).index


# --- relative strength -------------------------------------------------

def test_rel_strength_spread():
    stock = [100.0] * 30
    stock[-21] = 100.0
    stock[-1] = 110.0           # +10% over 20d
    spy = [100.0] * 30
    spy[-21] = 100.0
    spy[-1] = 104.0             # +4% over 20d
    out = raw_rel_strength({"AAA": _price_df(stock)}, pd.Series(spy))
    assert out["AAA"] == pytest.approx(0.06, abs=1e-9)


# --- quality -----------------------------------------------------------

def test_quality_gross_profitability():
    fund = pd.DataFrame(
        {"gross_profit_ttm": [50.0, 20.0], "total_assets": [100.0, 100.0]},
        index=["AAA", "BBB"],
    )
    out = raw_quality(fund)
    assert out["AAA"] == pytest.approx(0.50)
    assert out["BBB"] == pytest.approx(0.20)


def test_quality_nan_on_bad_assets():
    fund = pd.DataFrame(
        {"gross_profit_ttm": [50.0], "total_assets": [0.0]}, index=["AAA"]
    )
    assert math.isnan(raw_quality(fund)["AAA"])


# --- value -------------------------------------------------------------

def test_value_inverts_ev_ebitda():
    fund = pd.DataFrame(
        {"enterprise_value": [1000.0, 1000.0], "ebitda": [100.0, 50.0]},
        index=["CHEAP", "RICH"],
    )
    out = raw_value(fund)
    # CHEAP has EV/EBITDA 10, RICH has 20 → CHEAP scores higher
    assert out["CHEAP"] > out["RICH"]
    assert out["CHEAP"] == pytest.approx(0.10)


def test_value_excludes_negative_ebitda():
    fund = pd.DataFrame(
        {"enterprise_value": [1000.0], "ebitda": [-50.0]}, index=["LOSS"]
    )
    assert math.isnan(raw_value(fund)["LOSS"])


# --- earnings surprise -------------------------------------------------

def test_earnings_surprise_weighted():
    fund = pd.DataFrame(
        {"surprise_recent": [10.0], "surprise_prior": [5.0]}, index=["AAA"]
    )
    out = raw_earnings_surprise(fund)
    assert out["AAA"] == pytest.approx(0.7 * 10.0 + 0.3 * 5.0)


def test_earnings_surprise_nan_when_prior_missing():
    fund = pd.DataFrame(
        {"surprise_recent": [10.0], "surprise_prior": [np.nan]}, index=["AAA"]
    )
    assert math.isnan(raw_earnings_surprise(fund)["AAA"])


# --- sector-neutral ranking -------------------------------------------

def test_sector_neutral_rank_ranks_within_sector():
    # two sectors, 10 names each so neither hits the small-sector fallback
    raw = pd.Series(
        list(range(10)) + list(range(100, 110)),
        index=[f"T{i}" for i in range(20)],
        dtype=float,
    )
    sectors = pd.Series(["Tech"] * 10 + ["Energy"] * 10, index=raw.index)
    ranked = _sector_neutral_rank(raw, sectors)
    assert ranked.between(0, 100).all()
    # the top name of each sector should rank ~100 despite different raw scales
    assert ranked["T9"] == pytest.approx(100.0)
    assert ranked["T19"] == pytest.approx(100.0)


def test_sector_neutral_rank_preserves_nan():
    raw = pd.Series([1.0, np.nan, 3.0], index=["A", "B", "C"])
    sectors = pd.Series(["X", "X", "X"], index=raw.index)
    ranked = _sector_neutral_rank(raw, sectors)
    assert math.isnan(ranked["B"])


# --- composite ---------------------------------------------------------

def test_composite_stays_in_range():
    rng = np.random.default_rng(1)
    scores = pd.DataFrame(
        {c: rng.uniform(0, 100, 50) for c in FACTOR_COLS}
    )
    composite = sum(scores[c] * FACTOR_WEIGHTS[c] for c in FACTOR_COLS)
    assert composite.between(0, 100).all()
    assert sum(FACTOR_WEIGHTS.values()) == pytest.approx(1.0)


def test_drop_not_impute():
    scores = pd.DataFrame(
        {c: [50.0, 50.0] for c in FACTOR_COLS}, index=["GOOD", "BAD"]
    )
    scores.loc["BAD", "quality"] = np.nan
    kept = scores.dropna(how="any", subset=FACTOR_COLS)
    assert "GOOD" in kept.index and "BAD" not in kept.index
