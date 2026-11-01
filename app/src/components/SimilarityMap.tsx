"use client";

import * as d3 from "d3";
import { useEffect, useMemo, useRef, useState } from "react";
import { league, money } from "@/lib/api";

export type MapPoint = {
  player_id: string;
  name: string;
  club: string | null;
  group: string;
  competition_id: string;
  age: number | null;
  value_eur: number | null;
  x: number;
  y: number;
};

const W = 800;
const H = 560;

/** Every qualifying player placed by how he contributes (embedding space),
 * zoomable and pannable. The target is red, his matches chalk. */
export function SimilarityMap({
  points,
  target,
  matches,
  onPick,
}: {
  points: MapPoint[];
  target: string | null;
  matches: string[];
  onPick: (id: string) => void;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [transform, setTransform] = useState(d3.zoomIdentity);
  const [hover, setHover] = useState<MapPoint | null>(null);

  const { sx, sy } = useMemo(() => {
    const pad = 24;
    const xs = d3.extent(points, (p) => p.x) as [number, number];
    const ys = d3.extent(points, (p) => p.y) as [number, number];
    return {
      sx: d3.scaleLinear().domain(xs).range([pad, W - pad]),
      sy: d3.scaleLinear().domain(ys).range([H - pad, pad]),
    };
  }, [points]);

  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  const reduce = useReducedMotion();

  useEffect(() => {
    if (!svgRef.current) return;
    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([1, 12])
      .translateExtent([
        [0, 0],
        [W, H],
      ])
      .on("zoom", (e) => setTransform(e.transform));
    zoomRef.current = zoom;
    d3.select(svgRef.current).call(zoom);
  }, []);

  const matchSet = useMemo(() => new Set(matches), [matches]);
  const highlighted = points.filter((p) => p.player_id === target || matchSet.has(p.player_id));

  // Move the camera onto the target and his matches whenever they change.
  const focusKey = highlighted.map((p) => p.player_id).join(",");
  useEffect(() => {
    const svg = svgRef.current;
    const zoom = zoomRef.current;
    if (!svg || !zoom || highlighted.length < 2) return;
    const xs = highlighted.map((p) => sx(p.x));
    const ys = highlighted.map((p) => sy(p.y));
    const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
    const [y0, y1] = [Math.min(...ys), Math.max(...ys)];
    const k = Math.min(8, 0.55 / Math.max((x1 - x0) / W, (y1 - y0) / H, 0.02));
    const t = d3.zoomIdentity.translate(W / 2, H / 2).scale(k).translate(-(x0 + x1) / 2, -(y0 + y1) / 2);
    d3.select(svg).transition().duration(reduce ? 0 : 750).ease(d3.easeCubicInOut).call(zoom.transform, t);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- refocus only when the highlighted set changes
  }, [focusKey, sx, sy, reduce]);
  const k = transform.k;

  return (
    <div className="relative">
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="w-full cursor-grab touch-none rounded-sm bg-turf active:cursor-grabbing"
        role="img"
        aria-label="Map of players by playing style. Scroll to zoom, drag to pan."
      >
        <g transform={transform.toString()}>
          {points.map((p) => {
            const isTarget = p.player_id === target;
            const isMatch = matchSet.has(p.player_id);
            if (isTarget || isMatch) return null;
            return (
              <circle
                key={p.player_id}
                cx={sx(p.x)}
                cy={sy(p.y)}
                r={2.2 / Math.sqrt(k)}
                fill="var(--fog)"
                fillOpacity={0.35}
                onMouseEnter={() => setHover(p)}
                onMouseLeave={() => setHover(null)}
                onClick={() => onPick(p.player_id)}
                className="cursor-pointer"
              />
            );
          })}
          {highlighted.map((p) => {
            const isTarget = p.player_id === target;
            return (
              <g key={p.player_id} onClick={() => onPick(p.player_id)} className="cursor-pointer"
                 onMouseEnter={() => setHover(p)} onMouseLeave={() => setHover(null)}>
                <circle
                  cx={sx(p.x)}
                  cy={sy(p.y)}
                  r={(isTarget ? 6 : 4) / Math.sqrt(k)}
                  fill={isTarget ? "var(--club)" : "var(--chalk)"}
                  stroke="var(--night)"
                  strokeWidth={1 / k}
                />
                <text
                  x={sx(p.x) + 7 / k}
                  y={sy(p.y) + 3.5 / k}
                  fontSize={11 / k}
                  fill={isTarget ? "var(--chalk)" : "var(--fog)"}
                  fontWeight={isTarget ? 600 : 400}
                >
                  {p.name.split(" ").slice(-1)[0]}
                </text>
              </g>
            );
          })}
        </g>
      </svg>
      {hover && (
        <div className="pointer-events-none absolute top-3 left-3 rounded border border-line bg-night/95 px-3 py-2 text-sm">
          <div className="font-semibold">{hover.name}</div>
          <div className="font-mono text-[11px] text-fog">
            {[hover.group, hover.club, league(hover.competition_id), hover.age && `${hover.age} yrs`, money(hover.value_eur)]
              .filter(Boolean)
              .join(" · ")}
          </div>
        </div>
      )}
      <p className="mt-2 font-mono text-[10px] text-fog uppercase">
        Scroll to zoom · drag to pan · click a player to centre the search on him
      </p>
    </div>
  );
}
