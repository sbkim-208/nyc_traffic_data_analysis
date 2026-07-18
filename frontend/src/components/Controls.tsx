import { toLocalIso } from "../api";
import type { Meta } from "../types";
import type { Mode } from "./Header";

interface Props {
  mode: Mode;
  meta: Meta | null;
  at: string;
  onAtChange: (at: string) => void;
  dow: number | null;
  onDowChange: (d: number | null) => void;
  hour: number | null;
  onHourChange: (h: number | null) => void;
}

const DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const FIVE_MIN = 5 * 60 * 1000;

export function Controls(props: Props) {
  if (!props.meta) {
    return <div className="border-t border-slate-800 bg-slate-900/80 px-5 py-3 text-xs text-slate-500">
      Loading metadata…
    </div>;
  }
  return props.mode === "snapshot" ? <SnapshotControls {...props} /> : <AggregateControls {...props} />;
}

function SnapshotControls({ meta, at, onAtChange }: Props) {
  const min = new Date(meta!.date_min).getTime();
  const max = new Date(meta!.date_max).getTime();
  const cur = at ? new Date(at).getTime() : (min + max) / 2;

  return (
    <div className="border-t border-slate-800 bg-slate-900/80 backdrop-blur px-5 py-3 flex items-center gap-4 shrink-0">
      <div className="text-[10px] uppercase tracking-wide text-slate-500 shrink-0">Time</div>
      <div className="text-[11px] text-slate-400 w-28 shrink-0 tabular-nums">
        {new Date(meta!.date_min).toLocaleDateString()}
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={FIVE_MIN}
        value={cur}
        onChange={(e) => onAtChange(toLocalIso(new Date(Number(e.target.value))))}
        className="flex-1 accent-blue-500"
      />
      <div className="text-[11px] text-slate-400 w-28 text-right shrink-0 tabular-nums">
        {new Date(meta!.date_max).toLocaleDateString()}
      </div>
      <div className="text-sm font-medium w-52 text-right shrink-0 tabular-nums">
        {new Date(cur).toLocaleString(undefined, {
          weekday: "short", month: "short", day: "numeric",
          hour: "numeric", minute: "2-digit",
        })}
      </div>
    </div>
  );
}

function AggregateControls({ dow, onDowChange, hour, onHourChange }: Props) {
  return (
    <div className="border-t border-slate-800 bg-slate-900/80 backdrop-blur px-5 py-3 grid grid-cols-[auto_1fr_auto] gap-6 items-center shrink-0">
      <div className="flex items-center gap-2">
        <div className="text-[10px] uppercase tracking-wide text-slate-500">Day</div>
        <div className="flex gap-1">
          <Chip active={dow == null} onClick={() => onDowChange(null)}>Any</Chip>
          {DOW_LABELS.map((d, i) => (
            <Chip key={d} active={dow === i} onClick={() => onDowChange(i)}>{d}</Chip>
          ))}
        </div>
      </div>
      <div className="flex items-center gap-3 min-w-0">
        <div className="text-[10px] uppercase tracking-wide text-slate-500 shrink-0">Hour</div>
        <input
          type="range"
          min={0}
          max={23}
          step={1}
          value={hour ?? 0}
          onChange={(e) => onHourChange(Number(e.target.value))}
          className="flex-1 accent-blue-500 max-w-md"
        />
        <div className="text-sm font-medium tabular-nums w-14 text-right shrink-0">
          {hour != null ? `${String(hour).padStart(2, "0")}:00` : "Any"}
        </div>
        {hour != null && (
          <button
            onClick={() => onHourChange(null)}
            className="text-[10px] uppercase tracking-wide text-slate-400 hover:text-white"
          >
            clear
          </button>
        )}
      </div>
      <div className="text-xs text-slate-500 tabular-nums">
        {dow != null ? DOW_LABELS[dow] : "any day"} · {hour != null ? `${hour}:00` : "any hour"}
      </div>
    </div>
  );
}

function Chip({ active, onClick, children }: {
  active: boolean; onClick: () => void; children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`px-2 py-1 rounded text-xs font-medium transition ${
        active ? "bg-blue-500 text-white" : "bg-slate-800 text-slate-300 hover:bg-slate-700"
      }`}
    >
      {children}
    </button>
  );
}
