"use client";

import { motion, useReducedMotion } from "motion/react";
import { useEffect, useRef, useState } from "react";
import type { Slot } from "@/lib/api";

const LENGTH = 105;
const WIDTH = 68;
const PERSPECTIVE = 1500; // px, matches the CSS perspective below
const TILT_DEG = 40;

/** Chalk markings of a full pitch in metres (attacking left to right). */
function Markings() {
  const box = (x: number, depth: number, width: number) => (
    <rect x={x} y={(WIDTH - width) / 2} width={depth} height={width} />
  );
  return (
    <g fill="none" stroke="var(--chalk)" strokeOpacity={0.32} strokeWidth={0.28}>
      <rect x={0} y={0} width={LENGTH} height={WIDTH} />
      <line x1={LENGTH / 2} y1={0} x2={LENGTH / 2} y2={WIDTH} />
      <circle cx={LENGTH / 2} cy={WIDTH / 2} r={9.15} />
      {box(0, 16.5, 40.3)}
      {box(LENGTH - 16.5, 16.5, 40.3)}
      {box(0, 5.5, 18.3)}
      {box(LENGTH - 5.5, 5.5, 18.3)}
      <path d={`M 16.5 ${WIDTH / 2 - 7.3} A 9.15 9.15 0 0 1 16.5 ${WIDTH / 2 + 7.3}`} />
      <path
        d={`M ${LENGTH - 16.5} ${WIDTH / 2 - 7.3} A 9.15 9.15 0 0 0 ${LENGTH - 16.5} ${WIDTH / 2 + 7.3}`}
      />
      <g fill="var(--chalk)" fillOpacity={0.4} stroke="none">
        <circle cx={11} cy={WIDTH / 2} r={0.35} />
        <circle cx={LENGTH - 11} cy={WIDTH / 2} r={0.35} />
        <circle cx={LENGTH / 2} cy={WIDTH / 2} r={0.35} />
      </g>
    </g>
  );
}

/** Tilting brings the near touchline toward the viewer, so it spreads wider
 * than the container; shrinking the pitch by this factor keeps it inside. */
function fitFactor(h: number, tiltRad: number) {
  return PERSPECTIVE / (PERSPECTIVE + (h / 2) * Math.sin(tiltRad));
}

/** Where a point on the (shrunk, tilted) pitch lands on screen. (u, v) are in
 * container pixels as if the pitch filled the container. */
function project(u: number, v: number, w: number, h: number, tiltRad: number) {
  const k = fitFactor(h, tiltRad);
  const du = (u - w / 2) * k;
  const dy = (v - h / 2) * k;
  const z = dy * Math.sin(tiltRad); // negative = away from the viewer
  const scale = PERSPECTIVE / (PERSPECTIVE - z);
  return {
    x: w / 2 + du * scale,
    y: h / 2 + dy * Math.cos(tiltRad) * scale,
    scale,
  };
}

const fitColour = (fit: number) => {
  // Single ramp from line (poor) to chalk (strong); red is kept for problems.
  const t = Math.min(1, Math.max(0, (fit - 30) / 60));
  return `color-mix(in oklab, var(--chalk) ${Math.round(t * 100)}%, var(--line))`;
};

function Plate({ slot, selected, onSelect }: { slot: Slot; selected: boolean; onSelect: () => void }) {
  const s = slot.starter;
  const urgent = slot.severity === "critical" || slot.severity === "high";
  const flagged = urgent || slot.severity === "medium";
  const label = s ? `${slot.key}, ${s.name}, fit ${s.fit}` : `${slot.key}, no rated starter`;
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-pressed={selected}
      aria-label={`${label}${slot.severity ? `, ${slot.severity} priority` : ""}`}
      className="group flex flex-col items-center gap-1 focus-visible:outline-none"
    >
      <span
        className={`flex size-12 items-center justify-center rounded-full border-2 bg-night/85 font-display text-[22px] leading-none font-black tabular transition-transform group-hover:scale-105 group-focus-visible:ring-2 group-focus-visible:ring-chalk ${
          urgent ? "gap-pulse" : ""
        } ${selected ? "ring-2 ring-chalk ring-offset-2 ring-offset-night" : ""}`}
        style={{
          borderColor: flagged ? "var(--club)" : s ? fitColour(s.fit) : "var(--line)",
          borderStyle: s ? "solid" : "dashed",
        }}
      >
        {s ? s.fit : "?"}
      </span>
      <span className="flex max-w-32 items-baseline gap-1 truncate rounded-sm bg-night/85 px-1.5 py-0.5 text-[12px] leading-tight font-medium">
        <span className="font-mono text-[9px] text-fog">{slot.key}</span>
        {s ? s.name.split(" ").slice(-1)[0] : slot.role}
      </span>
    </button>
  );
}

export function Pitch({
  slots,
  selected,
  onSelect,
  tilted,
}: {
  slots: Slot[];
  selected: string | null;
  onSelect: (key: string) => void;
  tilted: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const reduce = useReducedMotion();
  const tilt = tilted ? (TILT_DEG * Math.PI) / 180 : 0;
  const k = size.h ? fitFactor(size.h, tilt) : 1;
  const inset = `${((1 - k) / 2) * 100}%`;

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) =>
      setSize({ w: entry.contentRect.width, h: entry.contentRect.height }),
    );
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  return (
    <div
      ref={ref}
      className="relative mx-auto w-full"
      style={{ aspectRatio: `${LENGTH} / ${WIDTH}`, perspective: `${PERSPECTIVE}px` }}
    >
      <motion.div
        className="absolute rounded-sm bg-turf"
        initial={false}
        animate={{ rotateX: tilted ? TILT_DEG : 0, top: inset, bottom: inset, left: inset, right: inset }}
        transition={{ duration: reduce ? 0 : 0.8, ease: [0.4, 0, 0.2, 1] }}
        style={{
          backgroundImage:
            "repeating-linear-gradient(90deg, transparent 0 9.52%, color-mix(in oklab, var(--turf-2) 70%, transparent) 9.52% 19.04%)",
        }}
      >
        <svg viewBox={`0 0 ${LENGTH} ${WIDTH}`} className="absolute inset-0 size-full" aria-hidden>
          <Markings />
        </svg>
      </motion.div>

      {size.w > 0 &&
        slots.map((slot) => {
          // Formation x runs across the pitch (left to right touchline) and y up
          // it (own goal to theirs); on screen the team attacks to the right.
          const p = project((slot.y / 100) * size.w, (slot.x / 100) * size.h, size.w, size.h, tilt);
          return (
            <motion.div
              key={slot.starter?.player_id ?? `empty-${slot.key}`}
              className="absolute top-0 left-0"
              initial={false}
              animate={{ x: p.x, y: p.y, scale: p.scale }}
              transition={{ duration: reduce ? 0 : 0.7, ease: [0.4, 0, 0.2, 1] }}
              style={{ translateX: "-50%", translateY: "-38%", zIndex: Math.round(p.y) }}
            >
              <Plate slot={slot} selected={selected === slot.key} onSelect={() => onSelect(slot.key)} />
            </motion.div>
          );
        })}
    </div>
  );
}
