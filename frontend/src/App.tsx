import { useEffect, useState } from "react";
import { Header, type Mode } from "./components/Header";
import { Map } from "./components/Map";
import { Controls } from "./components/Controls";
import { DetailPanel } from "./components/DetailPanel";
import { ODPanel } from "./components/ODPanel";
import { Legend } from "./components/Legend";
import { fetchMeta, fetchSegmentsAggregate, fetchSegmentsAt, toLocalIso } from "./api";
import type { Meta, Segment } from "./types";

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [mode, setMode] = useState<Mode>("snapshot");
  const [at, setAt] = useState<string>("");
  const [dow, setDow] = useState<number | null>(1); // Tuesday
  const [hour, setHour] = useState<number | null>(8);
  const [segments, setSegments] = useState<Segment[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<number[]>([]);

  useEffect(() => {
    fetchMeta()
      .then(m => {
        setMeta(m);
        const min = new Date(m.date_min).getTime();
        const max = new Date(m.date_max).getTime();
        setAt(toLocalIso(new Date((min + max) / 2)));
      })
      .catch(e => setError(`Failed to load metadata: ${e.message}`));
  }, []);

  useEffect(() => {
    setSelected([]);
  }, [mode]);

  useEffect(() => {
    if (!meta) return;
    if (mode === "snapshot" && !at) return;
    setLoading(true);
    setError(null);
    const promise =
      mode === "snapshot"
        ? fetchSegmentsAt(at, 30)
        : fetchSegmentsAggregate(dow, hour);
    promise
      .then(setSegments)
      .catch(e => setError(`Failed to load segments: ${e.message}`))
      .finally(() => setLoading(false));
  }, [meta, mode, at, dow, hour]);

  const handleSegmentClick = (linkId: number) => {
    if (mode === "od") {
      setSelected(prev => {
        if (prev.includes(linkId)) return prev.filter(x => x !== linkId);
        if (prev.length >= 2) return [prev[1], linkId];
        return [...prev, linkId];
      });
    } else {
      setSelected([linkId]);
    }
  };

  return (
    <div className="h-screen w-screen flex flex-col bg-slate-950 text-slate-100">
      <Header mode={mode} onModeChange={setMode} meta={meta} />
      <div className="flex-1 flex relative overflow-hidden">
        <div className="flex-1 relative">
          <Map
            segments={segments}
            selectedIds={selected}
            onSegmentClick={handleSegmentClick}
          />
          <Legend />
          {loading && (
            <div className="absolute top-4 left-4 px-3 py-1.5 rounded-md bg-slate-900/80 backdrop-blur text-xs">
              Loading…
            </div>
          )}
          {error && (
            <div className="absolute top-4 left-4 px-3 py-1.5 rounded-md bg-red-900/90 text-xs max-w-md">
              {error}
            </div>
          )}
        </div>
        <aside className="w-96 shrink-0 border-l border-slate-800 bg-slate-900/60 backdrop-blur overflow-y-auto">
          {mode === "od" ? (
            <ODPanel selectedIds={selected} dow={dow} hour={hour} />
          ) : (
            <DetailPanel selectedId={selected[0] ?? null} meta={meta} />
          )}
        </aside>
      </div>
      <Controls
        mode={mode}
        meta={meta}
        at={at}
        onAtChange={setAt}
        dow={dow}
        onDowChange={setDow}
        hour={hour}
        onHourChange={setHour}
      />
    </div>
  );
}
