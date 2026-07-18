"""All pandas operations that back the API.

Pure functions: read from `store`, return JSON-safe dicts/lists. Every aggregation,
filter, and time-bucketing here uses pandas — there is no other data layer.
"""
from __future__ import annotations

import json
import math
from datetime import datetime

import pandas as pd

from . import los
from .data import store


def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame → list[dict] with JSON-safe types (numpy/pyarrow → native)."""
    return json.loads(df.to_json(orient="records", date_format="iso"))


# ---------------------------------------------------------------------------
# Map: snapshot at a moment
# ---------------------------------------------------------------------------

def _to_naive(ts: datetime) -> pd.Timestamp:
    """Parquet timestamps are naive (NYC local). Strip any tz from input to match."""
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert(None)
    return t


def segments_at(at: datetime, window_minutes: int = 30) -> list[dict]:
    spd = store.speeds
    seg = store.segments
    assert spd is not None and seg is not None

    half = pd.Timedelta(minutes=window_minutes / 2)
    at_ts = _to_naive(at)
    sub = spd[(spd["timestamp"] >= at_ts - half) & (spd["timestamp"] <= at_ts + half)]
    if sub.empty:
        return []

    latest = sub.sort_values("timestamp").groupby("link_id", as_index=False).last()
    merged = seg.merge(latest[["link_id", "speed_mph", "travel_time_s"]], on="link_id", how="inner")
    merged["los"] = los.classify_series(merged["speed_mph"])
    merged["color"] = los.color_series(merged["los"])
    return _records(merged[[
        "link_id", "link_name", "borough", "polyline",
        "speed_mph", "travel_time_s", "los", "color",
    ]])


# ---------------------------------------------------------------------------
# Map: aggregate by (dow, hour)
# ---------------------------------------------------------------------------

def segments_aggregate(dow: int | None, hour: int | None) -> list[dict]:
    spd = store.speeds
    seg = store.segments
    assert spd is not None and seg is not None

    mask = pd.Series(True, index=spd.index)
    if dow is not None:
        mask &= spd["dow"] == dow
    if hour is not None:
        mask &= spd["hour"] == hour
    sub = spd[mask]

    if sub.empty:
        return []

    agg = sub.groupby("link_id", as_index=False).agg(
        median_speed_mph=("speed_mph", "median"),
        p10_speed_mph=("speed_mph", lambda s: s.quantile(0.10)),
        observations=("speed_mph", "count"),
    )
    merged = seg.merge(agg, on="link_id", how="inner")
    merged["los"] = los.classify_series(merged["median_speed_mph"])
    merged["color"] = los.color_series(merged["los"])
    return _records(merged[[
        "link_id", "link_name", "borough", "polyline",
        "median_speed_mph", "p10_speed_mph", "observations", "los", "color",
    ]])


# ---------------------------------------------------------------------------
# Detail panel
# ---------------------------------------------------------------------------

def segment_meta(link_id: int) -> dict | None:
    seg = store.segments
    spd = store.speeds
    assert seg is not None and spd is not None

    row = seg[seg["link_id"] == link_id]
    if row.empty:
        return None
    sub = spd[spd["link_id"] == link_id]

    base = json.loads(row.iloc[[0]].to_json(orient="records"))[0]
    base["stats"] = {
        "observations": int(len(sub)),
        "median_speed_mph": float(sub["speed_mph"].median()) if len(sub) else None,
        "median_travel_time_s": float(sub["travel_time_s"].median()) if len(sub) else None,
    }
    return base


def segment_series(
    link_id: int,
    start: datetime,
    end: datetime,
    granularity: str = "1h",
) -> list[dict]:
    spd = store.speeds
    assert spd is not None

    start_ts = _to_naive(start)
    end_ts = _to_naive(end)
    sub = spd[
        (spd["link_id"] == link_id)
        & (spd["timestamp"] >= start_ts)
        & (spd["timestamp"] < end_ts)
    ]
    if sub.empty:
        return []

    grp = sub.set_index("timestamp")["speed_mph"].resample(granularity)
    out = pd.DataFrame({
        "median_speed_mph": grp.median(),
        "p10_speed_mph": grp.quantile(0.10),
        "observations": grp.count().astype(int),
    }).dropna(subset=["median_speed_mph"]).reset_index()
    return _records(out)


# ---------------------------------------------------------------------------
# OD travel time (heuristic — see method note in response)
# ---------------------------------------------------------------------------

def _segment_summary(spd: pd.DataFrame, link_id: int) -> dict:
    sub = spd[spd["link_id"] == link_id]
    if sub.empty:
        return {"link_id": int(link_id), "observations": 0}
    return {
        "link_id": int(link_id),
        "observations": int(len(sub)),
        "median_travel_time_s": float(sub["travel_time_s"].median()),
        "p10_travel_time_s": float(sub["travel_time_s"].quantile(0.10)),
        "p90_travel_time_s": float(sub["travel_time_s"].quantile(0.90)),
        "median_speed_mph": float(sub["speed_mph"].median()),
    }


def _haversine_mi(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 3958.7613  # Earth radius in miles
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def od_estimate(
    from_id: int,
    to_id: int,
    dow: int | None,
    hour: int | None,
) -> dict:
    spd = store.speeds
    seg = store.segments
    assert spd is not None and seg is not None

    mask = pd.Series(True, index=spd.index)
    if dow is not None:
        mask &= spd["dow"] == dow
    if hour is not None:
        mask &= spd["hour"] == hour
    filtered = spd[mask]

    a = _segment_summary(filtered, from_id)
    b = _segment_summary(filtered, to_id)

    seg_a = seg[seg["link_id"] == from_id]
    seg_b = seg[seg["link_id"] == to_id]
    if seg_a.empty or seg_b.empty:
        return {"error": "segment not found", "from_segment": a, "to_segment": b}

    poly_a = list(seg_a.iloc[0]["polyline"])
    poly_b = list(seg_b.iloc[0]["polyline"])
    mid_a = poly_a[len(poly_a) // 2]
    mid_b = poly_b[len(poly_b) // 2]
    distance_mi = _haversine_mi(float(mid_a[0]), float(mid_a[1]), float(mid_b[0]), float(mid_b[1]))

    avg_speed = float(filtered["speed_mph"].median()) if len(filtered) else 15.0
    between_s = (distance_mi / avg_speed) * 3600.0 if avg_speed > 0 else 0.0
    total_s = (a.get("median_travel_time_s") or 0.0) + (b.get("median_travel_time_s") or 0.0) + between_s

    return {
        "from_segment": a,
        "to_segment": b,
        "midpoint_distance_mi": round(distance_mi, 3),
        "between_avg_speed_mph": round(avg_speed, 1),
        "between_estimated_s": round(between_s, 1),
        "estimated_total_s": round(total_s, 1),
        "method": (
            "Sum of segment median travel times + haversine midpoint distance ÷ "
            "median network speed. Coarse estimate; not a routed travel time."
        ),
    }
