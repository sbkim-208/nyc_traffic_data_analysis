export interface Meta {
  segments: number;
  observations: number;
  date_min: string;
  date_max: string;
  boroughs: string[];
}

export interface Segment {
  link_id: number;
  link_name: string;
  borough: string;
  polyline: [number, number][]; // [lat, lon] pairs
  speed_mph?: number;
  median_speed_mph?: number;
  travel_time_s?: number;
  observations?: number;
  los: string;
  color: string;
}

export interface SegmentDetail {
  link_id: number;
  link_name: string;
  borough: string;
  polyline: [number, number][];
  stats: {
    observations: number;
    median_speed_mph: number | null;
    median_travel_time_s: number | null;
  };
}

export interface SeriesPoint {
  timestamp: string;
  median_speed_mph: number;
  p10_speed_mph: number;
  observations: number;
}

export interface ODSegmentSummary {
  link_id: number;
  observations: number;
  median_travel_time_s?: number;
  p10_travel_time_s?: number;
  p90_travel_time_s?: number;
  median_speed_mph?: number;
}

export interface ODResult {
  from_segment: ODSegmentSummary;
  to_segment: ODSegmentSummary;
  midpoint_distance_mi: number;
  between_avg_speed_mph: number;
  between_estimated_s: number;
  estimated_total_s: number;
  method: string;
  error?: string;
}
