"""Clean raw NYC speed CSV → two parquet files.

Inputs:
    data/raw/speeds_*.csv (one or more)
Outputs:
    data/processed/segments.parquet   one row per link_id (stable metadata + polyline)
    data/processed/speeds.parquet     long format: link_id, timestamp, speed_mph, travel_time_s

Usage:
    python scripts/prepare.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"


def parse_link_points(s: str) -> list[tuple[float, float]] | None:
    """NYC link_points format: 'lat,lon lat,lon ...' (space-separated pairs).

    Some rows are malformed; return None if anything fails.
    """
    if not isinstance(s, str) or not s.strip():
        return None
    out: list[tuple[float, float]] = []
    for token in s.split():
        try:
            lat_str, lon_str = token.split(",")
            out.append((float(lat_str), float(lon_str)))
        except (ValueError, IndexError):
            return None
    return out if len(out) >= 2 else None


def main() -> int:
    raw_files = sorted(RAW_DIR.glob("speeds_*.csv"))
    if not raw_files:
        print(f"No raw CSVs found in {RAW_DIR}. Run download.py first.")
        return 1

    print(f"Reading {len(raw_files)} raw file(s)...")
    df = pd.concat((pd.read_csv(f) for f in raw_files), ignore_index=True)
    print(f"  raw rows: {len(df):,}")

    # Normalize column names (Socrata returns lowercase already, but be defensive).
    df.columns = [c.lower() for c in df.columns]

    # Required columns.
    needed = {"link_id", "speed", "travel_time", "data_as_of", "link_points", "borough", "link_name"}
    missing = needed - set(df.columns)
    if missing:
        print(f"Missing columns: {missing}")
        return 1

    # Type cleanup.
    df["timestamp"] = pd.to_datetime(df["data_as_of"], errors="coerce")
    df["speed_mph"] = pd.to_numeric(df["speed"], errors="coerce")
    df["travel_time_s"] = pd.to_numeric(df["travel_time"], errors="coerce")
    df["link_id"] = pd.to_numeric(df["link_id"], errors="coerce").astype("Int64")

    before = len(df)
    df = df.dropna(subset=["timestamp", "speed_mph", "travel_time_s", "link_id"])
    df = df[(df["speed_mph"] >= 0) & (df["speed_mph"] <= 100)]
    df = df[df["travel_time_s"] > 0]
    print(f"  rows after cleaning: {len(df):,} (dropped {before - len(df):,})")

    # ---- speeds table (long format, no polyline) ----
    speeds = df[["link_id", "timestamp", "speed_mph", "travel_time_s"]].copy()
    speeds = speeds.sort_values(["link_id", "timestamp"]).reset_index(drop=True)

    # ---- segments table (one row per link_id, stable metadata) ----
    # Take the most recent metadata observation per link.
    seg_src = df.sort_values("timestamp").groupby("link_id", as_index=False).last()
    seg_src["polyline"] = seg_src["link_points"].apply(parse_link_points)
    bad_polys = seg_src["polyline"].isna().sum()
    if bad_polys:
        print(f"  warning: {bad_polys} segments have unparseable polylines (will be dropped)")
    seg_src = seg_src.dropna(subset=["polyline"])

    segments = seg_src[["link_id", "link_name", "borough", "polyline"]].copy()
    segments = segments.sort_values("link_id").reset_index(drop=True)
    # Parquet can't store mixed types in a list column easily — convert to list-of-list.
    segments["polyline"] = segments["polyline"].apply(lambda pts: [list(p) for p in pts])

    # Drop speeds for segments we couldn't parse.
    speeds = speeds[speeds["link_id"].isin(segments["link_id"])].reset_index(drop=True)

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    seg_path = PROC_DIR / "segments.parquet"
    spd_path = PROC_DIR / "speeds.parquet"
    segments.to_parquet(seg_path, index=False)
    speeds.to_parquet(spd_path, index=False)

    print()
    print(f"segments → {seg_path}  ({len(segments):,} rows)")
    print(f"speeds   → {spd_path}  ({len(speeds):,} rows)")
    print(f"date range: {speeds['timestamp'].min()} → {speeds['timestamp'].max()}")
    print(f"boroughs:   {sorted(segments['borough'].dropna().unique())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
