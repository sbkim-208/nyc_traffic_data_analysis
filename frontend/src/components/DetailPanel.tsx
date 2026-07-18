import { useEffect, useState } from "react";
import { fetchSegmentMeta, fetchSegmentSeries } from "../api";
import type { Meta, SegmentDetail, SeriesPoint } from "../types";
import { TimeSeriesChart } from "./TimeSeriesChart";

interface Props {
  selectedId: number | null;
  meta: Meta | null;
}

export function DetailPanel({ selectedId, meta }: Props) {
  const [detail, setDetail] = useState<SegmentDetail | null>(null);
  const [series, setSeries] = useState<SeriesPoint[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (selectedId == null || !meta) {
      setDetail(null);
      setSeries([]);
      return;
    }
    setLoading(true);
    Promise.all([
      fetchSegmentMeta(selectedId),
      fetchSegmentSeries(selectedId, meta.date_min, meta.date_max, "1h"),
    ])
      .then(([d, s]) => { setDetail(d); setSeries(s); })
      .finally(() => setLoading(false));
  }, [selectedId, meta]);

  if (selectedId == null) {
    return (
      <div className="p-6 text-sm text-slate-400">
        Click a segment on the map to see its details and history.
      </div>
    );
  }
  if (loading || !detail) {
    return <div className="p-6 text-sm text-slate-400">Loading…</div>;
  }

  return (
    <div className="p-5 space-y-4">
      <div>
        <h2 className="text-base font-semibold leading-tight">
          {detail.link_name || `Segment ${detail.link_id}`}
        </h2>
        <p className="text-xs text-slate-400 mt-0.5">
          {detail.borough} · #{detail.link_id}
        </p>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <Stat
          label="Median speed"
          value={detail.stats.median_speed_mph?.toFixed(1)}
          unit="mph"
        />
        <Stat
          label="Median travel"
          value={detail.stats.median_travel_time_s?.toFixed(0)}
          unit="s"
        />
        <Stat
          label="Observations"
          value={detail.stats.observations.toLocaleString()}
          full
        />
      </div>
      <div>
        <h3 className="text-[10px] uppercase tracking-wide text-slate-400 mb-2">
          Speed over time (1h median)
        </h3>
        <TimeSeriesChart data={series} />
      </div>
    </div>
  );
}

function Stat({ label, value, unit, full }: {
  label: string; value?: string | number; unit?: string; full?: boolean;
}) {
  return (
    <div className={`bg-slate-800/60 rounded-md p-3 ${full ? "col-span-2" : ""}`}>
      <div className="text-[10px] uppercase tracking-wide text-slate-400">{label}</div>
      <div className="text-lg font-semibold mt-0.5 tabular-nums">
        {value ?? "—"}
        {value != null && unit ? <span className="text-xs text-slate-400 ml-1">{unit}</span> : null}
      </div>
    </div>
  );
}
