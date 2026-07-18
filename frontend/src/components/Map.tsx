import { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import type { Segment } from "../types";

const NYC_CENTER: [number, number] = [40.74, -73.97];
const TILE_URL = "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png";
const TILE_ATTR =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>';

interface Props {
  segments: Segment[];
  selectedIds: number[];
  onSegmentClick: (linkId: number) => void;
}

export function Map({ segments, selectedIds, onSegmentClick }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<L.Map | null>(null);
  const layerRef = useRef<L.LayerGroup | null>(null);
  const polylinesRef = useRef<Record<number, L.Polyline>>({});
  const onClickRef = useRef(onSegmentClick);
  const selectedIdsRef = useRef<number[]>(selectedIds);

  useEffect(() => { onClickRef.current = onSegmentClick; }, [onSegmentClick]);

  // Initialize map once.
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = L.map(containerRef.current, {
      center: NYC_CENTER,
      zoom: 12,
      preferCanvas: true,  // Canvas renderer is faster for many polylines.
      zoomControl: true,
    });
    L.tileLayer(TILE_URL, {
      attribution: TILE_ATTR,
      subdomains: "abcd",
      maxZoom: 19,
    }).addTo(map);
    layerRef.current = L.layerGroup().addTo(map);
    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
      layerRef.current = null;
      polylinesRef.current = {};
    };
  }, []);

  // Render segment polylines whenever the segments array changes.
  useEffect(() => {
    const layer = layerRef.current;
    if (!layer) return;
    layer.clearLayers();
    polylinesRef.current = {};

    for (const s of segments) {
      const speed = s.speed_mph ?? s.median_speed_mph;
      const isSelected = selectedIdsRef.current.includes(s.link_id);
      const line = L.polyline(s.polyline as L.LatLngExpression[], {
        color: isSelected ? "#3b82f6" : s.color,
        weight: isSelected ? 7 : 4,
        opacity: isSelected ? 1 : 0.85,
        interactive: true,
      });
      line.bindTooltip(
        `<div style="font: 12px -apple-system, sans-serif;">
           <div style="font-weight: 600;">${escapeHtml(s.link_name || "Segment")}</div>
           <div style="opacity: 0.7;">${escapeHtml(s.borough || "")} · #${s.link_id}</div>
           <div style="margin-top: 4px;">
             <strong>${speed != null ? speed.toFixed(1) : "—"} mph</strong>
             <span style="opacity: 0.6; margin-left: 4px;">LOS ${escapeHtml(s.los)}</span>
           </div>
         </div>`,
        { sticky: true, direction: "top", offset: [0, -4] },
      );
      line.on("click", () => onClickRef.current(s.link_id));
      line.on("mouseover", () => {
        if (!selectedIdsRef.current.includes(s.link_id)) {
          line.setStyle({ weight: 6, opacity: 1 });
        }
      });
      line.on("mouseout", () => {
        if (!selectedIdsRef.current.includes(s.link_id)) {
          line.setStyle({ weight: 4, opacity: 0.85 });
        }
      });
      line.addTo(layer);
      polylinesRef.current[s.link_id] = line;
    }
  }, [segments]);

  // Apply selection styling.
  useEffect(() => {
    selectedIdsRef.current = selectedIds;
    for (const [idStr, line] of Object.entries(polylinesRef.current)) {
      const id = Number(idStr);
      const segment = segments.find(s => s.link_id === id);
      if (!segment) continue;
      if (selectedIds.includes(id)) {
        line.setStyle({ color: "#3b82f6", weight: 7, opacity: 1 });
        line.bringToFront();
      } else {
        line.setStyle({ color: segment.color, weight: 4, opacity: 0.85 });
      }
    }
  }, [selectedIds, segments]);

  return <div ref={containerRef} className="w-full h-full" />;
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, c => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]!
  ));
}
