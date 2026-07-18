export interface LosBand {
  label: string;
  color: string;
  range: string;
}

export const LOS_BANDS: LosBand[] = [
  { label: "A", color: "#1a9850", range: "≥ 30 mph" },
  { label: "B", color: "#66bd63", range: "25–30 mph" },
  { label: "C", color: "#fee08b", range: "18–25 mph" },
  { label: "D", color: "#fdae61", range: "12–18 mph" },
  { label: "E", color: "#f46d43", range: "7–12 mph" },
  { label: "F", color: "#a50026", range: "< 7 mph" },
];
