import type { Meta } from "../types";

export type Mode = "snapshot" | "aggregate" | "od";

interface Props {
  mode: Mode;
  onModeChange: (m: Mode) => void;
  meta: Meta | null;
}

const MODES: { id: Mode; label: string; hint: string }[] = [
  { id: "snapshot",  label: "Snapshot",  hint: "Speeds at a single moment" },
  { id: "aggregate", label: "Aggregate", hint: "Median speed by day-of-week & hour" },
  { id: "od",        label: "O–D",       hint: "Origin–destination travel time estimate" },
];

export function Header({ mode, onModeChange, meta }: Props) {
  return (
    <header className="flex items-center justify-between px-5 py-3 border-b border-slate-800 bg-slate-900/80 backdrop-blur shrink-0">
      <div className="flex items-baseline gap-3">
        <div className="text-base font-semibold tracking-tight">
          NYC Traffic Congestion Explorer
        </div>
        {meta && (
          <div className="text-[11px] text-slate-500 tabular-nums">
            {meta.observations.toLocaleString()} obs · {meta.segments} segments ·{" "}
            {new Date(meta.date_min).toLocaleDateString()} → {new Date(meta.date_max).toLocaleDateString()}
          </div>
        )}
      </div>
      <div className="flex items-center gap-1 bg-slate-800/60 p-1 rounded-lg">
        {MODES.map(m => (
          <button
            key={m.id}
            onClick={() => onModeChange(m.id)}
            title={m.hint}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition ${
              mode === m.id
                ? "bg-slate-100 text-slate-900"
                : "text-slate-300 hover:text-white hover:bg-slate-700/40"
            }`}
          >
            {m.label}
          </button>
        ))}
      </div>
    </header>
  );
}
