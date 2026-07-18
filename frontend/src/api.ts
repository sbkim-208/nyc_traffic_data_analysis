import type { Meta, Segment, SegmentDetail, SeriesPoint, ODResult } from "./types";

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "http://localhost:8000";

/**
 * Format a Date as a naive ISO string (no Z suffix) using its local-time
 * components. The backend stores parquet timestamps in NYC local time without
 * tz info; sending Z-suffixed UTC would shift queries by the local offset.
 */
export function toLocalIso(d: Date): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return (
    `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}` +
    `T${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
  );
}

async function get<T>(path: string, params?: Record<string, string | number | null | undefined>): Promise<T> {
  const url = new URL(`${BASE}${path}`);
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v !== null && v !== undefined) url.searchParams.set(k, String(v));
    }
  }
  const r = await fetch(url.toString());
  if (!r.ok) throw new Error(`${r.status} ${r.statusText} on ${path}`);
  return r.json();
}

export const fetchMeta = () => get<Meta>("/api/meta");

export const fetchSegmentsAt = (at: string, window_minutes = 30) =>
  get<Segment[]>("/api/segments", { at, window_minutes });

export const fetchSegmentsAggregate = (dow: number | null, hour: number | null) =>
  get<Segment[]>("/api/segments/aggregate", { dow, hour });

export const fetchSegmentMeta = (link_id: number) =>
  get<SegmentDetail>(`/api/segments/${link_id}`);

export const fetchSegmentSeries = (
  link_id: number,
  start: string,
  end: string,
  granularity = "1h",
) => get<SeriesPoint[]>(`/api/segments/${link_id}/series`, { start, end, granularity });

export const fetchOD = (
  from: number,
  to: number,
  dow: number | null,
  hour: number | null,
) => get<ODResult>("/api/od", { from, to, dow, hour });
