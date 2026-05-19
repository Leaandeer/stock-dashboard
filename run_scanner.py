"""CLI: refresh scanner output, gated by macro state."""
from __future__ import annotations

import json

from dotenv import load_dotenv

from scanner.ranker import run
from signals.composite import compute as compute_macro, load as load_macro, save as save_macro


def main() -> None:
    load_dotenv()
    macro = load_macro()
    if macro is None:
        macro = compute_macro()
        save_macro(macro)
    results = run(macro)
    print(f"Macro: {macro.zone} ({macro.score:.0f}) → {len(results['candidates'])} candidates above threshold {results['threshold']}")


if __name__ == "__main__":
    main()
