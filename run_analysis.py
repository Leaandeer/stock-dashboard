"""CLI: end-to-end pipeline or analyst-only run."""
from __future__ import annotations

import argparse

from dotenv import load_dotenv

from analyst.blender import run as run_blender
from scanner.ranker import load as load_scanner, run as run_scanner
from signals.composite import compute as compute_macro, load as load_macro, save as save_macro


def main() -> None:
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--scan-and-analyze", action="store_true", help="macro gate → scanner → analyst")
    g.add_argument("--analyst-only", action="store_true", help="re-run analyst on existing scanner output")
    parser.add_argument("--force", action="store_true", help="bypass analyst cache")
    args = parser.parse_args()
    load_dotenv()

    if args.scan_and_analyze:
        macro = compute_macro()
        save_macro(macro)
        scanner_results = run_scanner(macro)
        if scanner_results.get("threshold") is None:
            print(f"Macro DEFENSIVE — scanner disabled, skipping analyst.")
            return
    else:
        scanner_results = load_scanner()
        if not scanner_results:
            raise SystemExit("No scanner_results.json — run scanner first.")

    cands = scanner_results.get("candidates", [])
    if not cands:
        print("No candidates passed the scanner threshold.")
        return
    out = run_blender(scanner_results, force=args.force)
    cached = sum(1 for r in out["rows"] if r["cached"])
    print(f"Analyzed {len(out['rows'])} tickers · {cached} cache hits · {len(out['errors'])} errors")


if __name__ == "__main__":
    main()
