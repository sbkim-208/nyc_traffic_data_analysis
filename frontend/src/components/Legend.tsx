import { LOS_BANDS } from "../los";

export function Legend() {
  return (
    <div className="absolute bottom-4 right-4 bg-slate-900/90 backdrop-blur rounded-lg p-3 text-xs shadow-lg border border-slate-800">
      <div className="text-[10px] uppercase tracking-wide text-slate-400 mb-2">
        Level of Service
      </div>
      <div className="flex flex-col gap-1">
        {LOS_BANDS.map(b => (
          <div key={b.label} className="flex items-center gap-2">
            <span
              className="inline-block w-5 h-2 rounded-sm"
              style={{ background: b.color }}
            />
            <span className="font-mono w-3 text-slate-200">{b.label}</span>
            <span className="text-slate-500 tabular-nums">{b.range}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
