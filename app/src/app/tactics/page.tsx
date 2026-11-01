"use client";

import { useState } from "react";
import { Pitch } from "@/components/lineup/Pitch";
import { type Club, type Lineup, query, useApi } from "@/lib/api";

type Tactics = {
  context: {
    team: string;
    traits: string;
    manager: string | null;
    manager_signature: string | null;
    manager_clubs: number | null;
  } | null;
  levers: { lever: string; squad: number | null; verdict: string; best: string[]; worst: string[]; advice: string }[];
  instructions: { player: string; text: string }[];
  settings: Record<string, string[]>;
  what_if: {
    slot: string;
    before: string | null;
    before_fit: number | null;
    after: string | null;
    after_fit: number | null;
    severity_before: string | null;
    severity_after: string | null;
  }[];
};

const LEVER_LABEL: Record<string, string> = { press: "Pressing", buildup: "Build-up", width: "Width", finishing: "Finishing" };
const OPTION_LABEL: Record<string, string> = {
  high: "High press",
  low: "Low block",
  short: "Play short",
  direct: "Go direct",
  wide: "Wide",
  narrow: "Narrow",
};

function Segmented({
  name,
  options,
  value,
  onChange,
}: {
  name: string;
  options: string[];
  value: string | null;
  onChange: (v: string | null) => void;
}) {
  const all: (string | null)[] = [options[1] ?? null, null, options[0]];
  return (
    <fieldset>
      <legend className="mb-1.5 text-sm">{LEVER_LABEL[name] ?? name}</legend>
      <div className="grid grid-cols-3 overflow-hidden rounded border border-line">
        {all.map((opt) => (
          <button
            key={opt ?? "current"}
            type="button"
            aria-pressed={value === opt}
            onClick={() => onChange(opt)}
            className={`px-2 py-1.5 text-xs ${value === opt ? "bg-chalk text-night" : "text-fog hover:text-chalk"}`}
          >
            {opt ? OPTION_LABEL[opt] ?? opt : "As now"}
          </button>
        ))}
      </div>
    </fieldset>
  );
}

function LeverBar({ value }: { value: number | null }) {
  return (
    <div className="relative h-1.5 w-full rounded-full bg-line" aria-hidden>
      {value != null && <div className="absolute inset-y-0 left-0 rounded-full bg-chalk" style={{ width: `${value}%` }} />}
      <div className="absolute -inset-y-1 left-1/2 w-px bg-fog" />
    </div>
  );
}

export default function TacticsPage() {
  const { data: club } = useApi<Club>("/api/club");
  const [choices, setChoices] = useState<Record<string, string | null>>({ press: null, buildup: null, width: null });
  const params = { press: choices.press, buildup: choices.buildup, width: choices.width };
  const { data: tactics, error } = useApi<Tactics>(`/api/tactics${query(params)}`, { keepPreviousData: true });
  const { data: lineup } = useApi<Lineup>(
    club ? `/api/lineup${query({ formation: club.formation, ...params })}` : null,
    { keepPreviousData: true },
  );
  const active = Object.values(choices).some(Boolean);

  return (
    <div className="grid min-h-screen grid-cols-1 xl:grid-cols-[minmax(0,1fr)_380px]">
      <div className="px-4 py-6 md:px-8">
        <header className="mb-6">
          <h1 className="font-display text-4xl leading-none font-black uppercase md:text-6xl">Tactics</h1>
          {tactics?.context && (
            <p className="mt-2 max-w-2xl text-sm">
              <span className="text-fog">How {tactics.context.team} plays now, for its strength: </span>
              {tactics.context.traits}.
              {tactics.context.manager && (
                <>
                  {" "}
                  <span className="text-fog">{tactics.context.manager}: </span>
                  {tactics.context.manager_signature ?? "not enough matches to profile"}
                  {tactics.context.manager_clubs === 1 && <span className="text-fog"> (one club so far)</span>}.
                </>
              )}
            </p>
          )}
        </header>

        {tactics && (
          <div className="mb-6 grid gap-4 sm:grid-cols-3">
            {Object.entries(tactics.settings).map(([name, options]) => (
              <Segmented
                key={name}
                name={name}
                options={options}
                value={choices[name] ?? null}
                onChange={(v) => setChoices((c) => ({ ...c, [name]: v }))}
              />
            ))}
          </div>
        )}
        <p className="mb-3 font-mono text-[10px] text-fog uppercase">
          {active
            ? "What-if XI: roles reweighted for the chosen settings. A rule-based simulation, not a prediction of results."
            : "Current best XI. Change a setting to see who fits it."}
        </p>
        {lineup && <Pitch slots={lineup.slots} selected={null} onSelect={() => {}} tilted />}

        {tactics && tactics.what_if.length > 0 && (
          <section aria-labelledby="what-if" className="mt-6">
            <h2 id="what-if" className="font-mono text-[10px] text-fog uppercase">
              What changes
            </h2>
            <ul className="mt-2 divide-y divide-line border-y border-line text-sm">
              {tactics.what_if.map((c) => (
                <li key={c.slot} className="grid grid-cols-[3.5rem_1fr_auto] items-baseline gap-3 py-2">
                  <span className="font-display text-xl font-black">{c.slot}</span>
                  <span>
                    {c.before === c.after ? c.before : `${c.before ?? "nobody"} → ${c.after ?? "nobody"}`}
                    {c.severity_before !== c.severity_after && (
                      <span className={c.severity_after ? "text-club" : "text-fog"}>
                        {" "}
                        · {c.severity_after ? `becomes a ${c.severity_after} priority` : "no longer a priority"}
                      </span>
                    )}
                  </span>
                  <span className="font-mono tabular">
                    {c.before_fit ?? "–"} → {c.after_fit ?? "–"}
                  </span>
                </li>
              ))}
            </ul>
          </section>
        )}
      </div>

      <aside className="space-y-7 border-t border-line bg-turf/40 px-5 py-6 xl:border-t-0 xl:border-l">
        {error && <p className="text-sm">{error.message}</p>}
        {tactics && (
          <section aria-labelledby="levers">
            <h2 id="levers" className="font-mono text-[10px] text-fog uppercase">
              What the XI is suited to
            </h2>
            <ul className="mt-3 space-y-4">
              {tactics.levers.map((l) => (
                <li key={l.lever}>
                  <div className="flex items-baseline justify-between">
                    <span className="font-semibold">{LEVER_LABEL[l.lever] ?? l.lever}</span>
                    <span className="font-mono text-xs tabular">{l.squad?.toFixed(0) ?? "–"}</span>
                  </div>
                  <div className="mt-1.5">
                    <LeverBar value={l.squad} />
                  </div>
                  <p className="mt-1.5 text-sm text-fog">{l.advice.replace(/^squad \S+ pct(: )?/, "")}</p>
                  {l.best.length > 0 && (
                    <p className="mt-1 font-mono text-[11px] text-fog">
                      ↑ {l.best.join(", ")}
                      {l.worst.length > 0 && `  ↓ ${l.worst.join(", ")}`}
                    </p>
                  )}
                </li>
              ))}
            </ul>
          </section>
        )}
        {tactics && (
          <section aria-labelledby="instructions">
            <h2 id="instructions" className="font-mono text-[10px] text-fog uppercase">
              Position by position
            </h2>
            <ul className="mt-3 space-y-2.5 text-sm">
              {tactics.instructions.map((i) => (
                <li key={i.text} className="border-l-2 border-line pl-3">
                  {i.text}
                </li>
              ))}
            </ul>
          </section>
        )}
      </aside>
    </div>
  );
}
