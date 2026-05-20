"""CLI: run the scanner walk-forward backtest (Phase 2, Part A).

Runs both factor sets and prints a comparison:
  v1 — the original 5 factors
  v2 — the Phase 2 refactor (price factors only; fundamentals not backtestable)
"""
from __future__ import annotations

import logging

from dotenv import load_dotenv

from backtest.scanner_backtest import run_backtest


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results = {}
    for fs in ("v1", "v2"):
        res = run_backtest(factor_set=fs)
        results[fs] = res.meta["headline"]["composite_20d_spread_bps"]
        print(f"\n[{fs}] {res.meta['headline']['verdict']}")
    print("\n--- comparison ---")
    print(f"v1 composite 20d spread: {results['v1']} bps")
    print(f"v2 composite 20d spread: {results['v2']} bps")


if __name__ == "__main__":
    main()
