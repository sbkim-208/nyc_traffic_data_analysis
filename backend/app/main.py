"""FastAPI app — thin layer over pandas queries."""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from . import queries
from .data import store


@asynccontextmanager
async def lifespan(_: FastAPI):
    store.load()
    yield


app = FastAPI(
    title="NYC Traffic Congestion API",
    description="Pandas-backed analysis of NYC DOT traffic speed data.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/meta", summary="Dataset metadata")
def meta():
    return store.stats()


@app.get("/api/segments", summary="Segment speeds at a moment")
def segments_at(
    at: datetime = Query(..., description="ISO 8601 timestamp"),
    window_minutes: int = Query(30, ge=1, le=240),
):
    return queries.segments_at(at, window_minutes)


@app.get("/api/segments/aggregate", summary="Median speed per segment for (dow, hour)")
def segments_aggregate(
    dow: int | None = Query(None, ge=0, le=6, description="0=Mon ... 6=Sun"),
    hour: int | None = Query(None, ge=0, le=23),
):
    return queries.segments_aggregate(dow, hour)


@app.get("/api/segments/{link_id}", summary="Single-segment metadata + summary stats")
def segment_detail(link_id: int):
    res = queries.segment_meta(link_id)
    if res is None:
        raise HTTPException(404, f"segment {link_id} not found")
    return res


@app.get("/api/segments/{link_id}/series", summary="Resampled speed time series")
def segment_series(
    link_id: int,
    start: datetime,
    end: datetime,
    granularity: str = Query("1h", description="pandas offset alias, e.g. 15min, 1h, 1D"),
):
    return queries.segment_series(link_id, start, end, granularity)


@app.get("/api/od", summary="Heuristic O–D travel-time estimate")
def od(
    from_id: int = Query(..., alias="from"),
    to_id: int = Query(..., alias="to"),
    dow: int | None = Query(None, ge=0, le=6),
    hour: int | None = Query(None, ge=0, le=23),
):
    return queries.od_estimate(from_id, to_id, dow, hour)
