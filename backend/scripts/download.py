"""Download NYC DOT traffic speed data from Socrata.

Dataset: i4gi-tjb9 (DOT Traffic Speeds NBE)
Docs: https://data.cityofnewyork.us/Transportation/DOT-Traffic-Speeds-NBE/i4gi-tjb9

Usage:
    python scripts/download.py --start 2024-04-01 --end 2024-05-01
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

DATASET_ID = "i4gi-tjb9"
BASE_URL = f"https://data.cityofnewyork.us/resource/{DATASET_ID}.csv"
PAGE_SIZE = 50_000
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def fetch_page(start: str, end: str, offset: int, max_retries: int = 5) -> pd.DataFrame:
    where = f"data_as_of between '{start}T00:00:00' and '{end}T00:00:00'"
    params = {
        "$where": where,
        "$order": "data_as_of",
        "$limit": PAGE_SIZE,
        "$offset": offset,
    }
    from io import StringIO
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(BASE_URL, params=params, timeout=120)
            r.raise_for_status()
            return pd.read_csv(StringIO(r.text))
        except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as e:
            last_err = e
            wait = min(2 ** attempt, 30)
            print(f"    page fetch failed (attempt {attempt}/{max_retries}): {e} — retrying in {wait}s")
            time.sleep(wait)
    raise last_err


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--end", required=True, help="YYYY-MM-DD exclusive")
    p.add_argument("--out", default=None, help="Output CSV path (default: data/raw/speeds_<start>_<end>.csv)")
    args = p.parse_args()

    out_path = Path(args.out) if args.out else RAW_DIR / f"speeds_{args.start}_{args.end}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.start} → {args.end} into {out_path}")
    # Append each page to disk immediately — if retries are exhausted and the script
    # dies, the pages fetched so far stay on disk instead of forcing a full re-download
    if out_path.exists():
        out_path.unlink()
    total = 0
    offset = 0
    while True:
        t0 = time.time()
        df = fetch_page(args.start, args.end, offset)
        dt = time.time() - t0
        print(f"  offset={offset:>9} got={len(df):>6} rows in {dt:.1f}s")
        if df.empty:
            break
        df.to_csv(out_path, mode="a", header=(offset == 0), index=False)
        total += len(df)
        if len(df) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    if total == 0:
        print("No rows returned. Check the date range.")
        return 1

    print(f"Done: {total:,} rows → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
