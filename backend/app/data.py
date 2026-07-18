"""In-memory pandas data store. Loaded once at server startup."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
SEGMENTS_PATH = DATA_DIR / "segments.parquet"
SPEEDS_PATH = DATA_DIR / "speeds.parquet"


class DataStore:
    """Holds the two DataFrames for the lifetime of the app."""

    def __init__(self) -> None:
        self.segments: pd.DataFrame | None = None
        self.speeds: pd.DataFrame | None = None

    @property
    def loaded(self) -> bool:
        return self.segments is not None and self.speeds is not None

    def load(self) -> None:
        if not SEGMENTS_PATH.exists() or not SPEEDS_PATH.exists():
            raise FileNotFoundError(
                f"Processed data missing in {DATA_DIR}. "
                "Run scripts/download.py and scripts/prepare.py first."
            )
        self.segments = pd.read_parquet(SEGMENTS_PATH)
        speeds = pd.read_parquet(SPEEDS_PATH)
        speeds["timestamp"] = pd.to_datetime(speeds["timestamp"])
        speeds = speeds.sort_values("timestamp").reset_index(drop=True)
        # Pre-compute dow/hour columns once — every aggregate query uses them.
        speeds["dow"] = speeds["timestamp"].dt.dayofweek.astype("int8")
        speeds["hour"] = speeds["timestamp"].dt.hour.astype("int8")
        self.speeds = speeds

    def stats(self) -> dict:
        assert self.segments is not None and self.speeds is not None
        return {
            "segments": int(len(self.segments)),
            "observations": int(len(self.speeds)),
            "date_min": self.speeds["timestamp"].min().isoformat(),
            "date_max": self.speeds["timestamp"].max().isoformat(),
            "boroughs": sorted(self.segments["borough"].dropna().unique().tolist()),
        }


store = DataStore()
