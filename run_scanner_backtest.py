"""CLI: run the scanner walk-forward backtest (Phase 2, Part A).

Reconstructs the 5-factor scanner weekly over the last 5 years with no
look-ahead and reports whether the top picks beat the universe forward.
"""
from __future__ import annotations

import logging

from dotenv import load_dotenv

from backtest.scanner_backtest import run_backtest


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    res = run_backtest()
    h = res.meta["headline"]
    print(f"\nHEADLINE: {h['composite_20d_spread_bps']} bps  ·  {h['verdict']}")


if __name__ == "__main__":
    main()
