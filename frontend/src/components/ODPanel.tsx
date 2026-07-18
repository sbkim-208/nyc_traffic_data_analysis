import { useEffect, useState } from "react";
import { fetchOD } from "../api";
import type { ODResult, ODSegmentSummary } from "../types";

interface Props {
  selectedIds: number[];
  dow: number | null;
  hour: number | null;
}

export function ODPanel({ selectedIds, dow, hour }: Props) {
  const [result, setResult] = useState<ODResult | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (selectedIds.length !== 2) {
      setResult(null);
      return;
    }
    setLoading(true);
    fetchOD(selectedIds[0], selectedIds[1], dow, hour)
      .then(setResult)
      .finally(() => setLoading(false));
  }, [selectedIds, dow, hour]);

  if (selectedIds.length === 0) {
    return (
      <div className="p-6 text-sm text-slate-400 space-y-2">
        <p>Origin–Destination travel time estimate.</p>
        <p>Click a segment for the <em>origin</em>, then a second for the <em>destination</em>.</p>
        <p className="text-xs text-slate-500">Filters at the bottom (day-of-week, hour) condition the estimate.</p>
      </div>
    );
  }
  if (selectedIds.length === 1) {
    return (
      <div className="p-6 text-sm text-slate-400">
        <div>Origin selected: <span className="font-mono text-slate-200">#{selectedIds[0]}</span></div>
        <div className="mt-2">Click a destination segment.</div>
      </div>
    );
  }
  if (loading || !result) {
    return <div className="p-6 text-sm text-slate-400">Estimating…</div>;
  }

  const fmtMin = (s?: number) => s != null ? `${(s / 60).toFixed(1)} min` : "—";

  return (
    <div className="p-5 space-y-4">
      <div>
        <h2 className="text-base font-semibold">Estimated travel time</h2>
        <p className="text-xs text-slate-400 mt-0.5">{describeFilter(dow, hour)}</p>
      </div>

      <div className="bg-blue-500/10 border border-blue-500/20 rounded-md p-4">
        <div className="text-[10px] uppercase tracking-wide text-blue-300/80">Total estimate</div>
        <div className="text-3xl font-semibold mt-1 tabular-nums">
          {fmtMin(result.estimated_total_s)}
        </div>
      </div>

      <Breakdown label="Origin segment" data={result.from_segment} />
      <Breakdown label="Destination segment" data={result.to_segment} />

      <div className="bg-slate-800/60 rounded-md p-3 text-xs space-y-1.5">
        <Row label="Midpoint distance" value={`${result.midpoint_distance_mi} mi`} />
        <Row label="Avg speed (between)" value={`${result.between_avg_speed_mph} mph`} />
        <Row label="Between-time" value={fmtMin(result.between_estimated_s)} />
      </div>

      <p className="text-[10px] text-slate-500 italic leading-relaxed">{result.method}</p>
    </div>
  );
}

function Breakdown({ label, data }: { label: string; data: ODSegmentSummary }) {
  return (
    <div className="bg-slate-800/60 rounded-md p-3">
      <div className="text-[10px] uppercase tracking-wide text-slate-400 mb-2">
        {label} <span className="text-slate-500">· #{data.link_id}</span>
      </div>
      {data.observations === 0 ? (
        <div className="text-xs text-slate-500 italic">No data for this filter.</div>
      ) : (
        <div className="grid grid-cols-3 gap-2 text-center">
          <Pct label="P10" value={data.p10_travel_time_s} />
          <Pct label="Median" value={data.median_travel_time_s} highlight />
          <Pct label="P90" value={data.p90_travel_time_s} />
        </div>
      )}
    </div>
  );
}

function Pct({ label, value, highlight }: { label: string; value?: number; highlight?: boolean }) {
  return (
    <div>
      <div className="text-[9px] uppercase tracking-wide text-slate-500">{label}</div>
      <div className={`text-sm tabular-nums ${highlight ? "font-semibold text-slate-100" : "text-slate-300"}`}>
        {value != null ? `${value.toFixed(0)}s` : "—"}
      </div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <span className="text-slate-400">{label}</span>
      <span className="tabular-nums">{value}</span>
    </div>
  );
}

function describeFilter(dow: number | null, hour: number | null): string {
  const dowName = dow != null ? ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][dow] : "any day";
  const hourLabel = hour != null ? `${String(hour).padStart(2,"0")}:00` : "any hour";
  return `${dowName} · ${hourLabel}`;
}
