import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { SeriesPoint } from "../types";

export function TimeSeriesChart({ data }: { data: SeriesPoint[] }) {
  if (data.length === 0) {
    return <div className="text-xs text-slate-500 italic py-6 text-center">No data in range.</div>;
  }
  return (
    <div className="h-48 -mx-2">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
          <CartesianGrid stroke="#334155" strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="timestamp"
            tickFormatter={(t) =>
              new Date(t).toLocaleDateString(undefined, { month: "short", day: "numeric" })
            }
            stroke="#64748b"
            fontSize={10}
            minTickGap={30}
          />
          <YAxis stroke="#64748b" fontSize={10} domain={[0, "auto"]} />
          <Tooltip
            contentStyle={{
              background: "#0f172a",
              border: "1px solid #334155",
              borderRadius: 4,
              fontSize: 12,
            }}
            labelFormatter={(t) => new Date(t).toLocaleString()}
            formatter={(v: number) => [`${Number(v).toFixed(1)} mph`, "Median speed"]}
          />
          <Line
            type="monotone"
            dataKey="median_speed_mph"
            stroke="#60a5fa"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
