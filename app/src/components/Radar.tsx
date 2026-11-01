"use client";

import { motion, useReducedMotion } from "motion/react";

export type RadarSeries = { id: string; name: string; values: (number | null)[] };

const COLOURS = ["var(--club)", "var(--chalk)", "var(--club-2)"];
const SIZE = 420;
const C = SIZE / 2;
const R = 150;

function point(i: number, n: number, pct: number) {
  const a = -Math.PI / 2 + (2 * Math.PI * i) / n;
  const r = (R * Math.max(0, Math.min(100, pct))) / 100;
  return [C + r * Math.cos(a), C + r * Math.sin(a)] as const;
}

/** Percentile radar; series morph between shapes when players change. */
export function Radar({ axes, series }: { axes: string[]; series: RadarSeries[] }) {
  const reduce = useReducedMotion();
  const n = axes.length;
  if (n < 3) return null;
  return (
    <svg viewBox={`0 0 ${SIZE} ${SIZE}`} className="w-full max-w-[520px]" role="img"
         aria-label={`Percentile radar: ${series.map((s) => s.name).join(" vs ")}`}>
      {[25, 50, 75, 100].map((ring) => (
        <polygon
          key={ring}
          points={axes.map((_, i) => point(i, n, ring).join(",")).join(" ")}
          fill="none"
          stroke="var(--line)"
          strokeWidth={ring === 50 ? 1.4 : 0.8}
        />
      ))}
      {axes.map((label, i) => {
        const [x, y] = point(i, n, 100);
        const [lx, ly] = point(i, n, 118);
        const anchor = Math.abs(lx - C) < 12 ? "middle" : lx > C ? "start" : "end";
        return (
          <g key={label}>
            <line x1={C} y1={C} x2={x} y2={y} stroke="var(--line)" strokeWidth={0.8} />
            <text x={lx} y={ly} textAnchor={anchor} dominantBaseline="middle" fill="var(--fog)" fontSize={11}>
              {label}
            </text>
          </g>
        );
      })}
      {series.map((s, si) => {
        const pts = s.values.map((v, i) => point(i, n, v ?? 0).join(",")).join(" ");
        return (
          <motion.polygon
            key={s.id}
            initial={false}
            animate={{ points: pts }}
            transition={{ duration: reduce ? 0 : 0.6, ease: [0.4, 0, 0.2, 1] }}
            fill={COLOURS[si]}
            fillOpacity={0.14}
            stroke={COLOURS[si]}
            strokeWidth={2}
            strokeLinejoin="round"
          />
        );
      })}
    </svg>
  );
}

export const SERIES_COLOURS = COLOURS;
