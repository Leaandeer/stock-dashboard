"""CLI: nightly forward-return computation for every scanner snapshot.

Cron this after run_scanner.py so each day's picks are snapshotted and every
prior snapshot's 1d/5d/20d forward return is refreshed as it matures.
"""
from __future__ import annotations

from dotenv import load_dotenv

from tracking.performance import update_returns


def main() -> None:
    load_dotenv()
    res = update_returns()
    print(
        f"Track record: {res['snapshots']} snapshot rows · "
        f"{res['updated']} return rows refreshed"
    )


if __name__ == "__main__":
    main()
