"""CLI: refresh all 6 macro signals and persist macro_state.json."""
from __future__ import annotations

import json
from dataclasses import asdict

from dotenv import load_dotenv

from signals.composite import compute, save


def main() -> None:
    load_dotenv()
    state = compute()
    save(state)
    print(json.dumps(asdict(state), indent=2, default=str))


if __name__ == "__main__":
    main()
