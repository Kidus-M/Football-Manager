"use client";

import Link from "next/link";
import { useState } from "react";
import { Pitch } from "@/components/lineup/Pitch";
import { type Club, type Lineup, query, type Slot, useApi } from "@/lib/api";

const SEVERITY_LABEL: Record<string, string> = {
  critical: "Critical",
  high: "High",
  medium: "Medium",
  low: "Low",
};

function SlotPanel({ slot }: { slot: Slot }) {
  const s = slot.starter;
  return (
    <section aria-labelledby="slot-title" className="space-y-5">
      <header>
        <div className="font-mono text-[10px] text-fog uppercase">{slot.groups.join(" / ")}</div>
        <h2 id="slot-title" className="font-display text-5xl leading-none font-black uppercase">
          {slot.key}
        </h2>
        <p className="mt-1 text-sm text-fog">{slot.role}</p>
      </header>

      {s ? (
        <div className="border-t border-line pt-4">
          <div className="flex items-baseline justify-between gap-3">
            <Link href={`/compare?ids=${s.player_id}`} className="text-lg font-semibold hover:underline">
              {s.name}
            </Link>
            <span className="font-display text-4xl font-black tabular">{s.fit}</span>
          </div>
          <dl className="mt-2 grid grid-cols-3 gap-2 font-mono text-[11px] text-fog">
            <div>
              <dt className="uppercase">Minutes</dt>
              <dd className="text-chalk tabular">{s.minutes}</dd>
            </div>
            <div>
              <dt className="uppercase">Age</dt>
              <dd className="text-chalk tabular">{slot.age ?? "–"}</dd>
            </div>
            <div>
              <dt className="uppercase">Contract</dt>
              <dd className="text-chalk tabular">{slot.contract ?? "–"}</dd>
            </div>
          </dl>
        </div>
      ) : (
        <p className="border-t border-line pt-4 text-sm">No squad player can be rated for this position yet.</p>
      )}

      {slot.issues.length > 0 && (
        <ul className="space-y-1.5 border-t border-line pt-4 text-sm">
          {slot.issues.map((issue) => (
            <li key={issue} className="flex gap-2">
              <span aria-hidden className="mt-1.5 size-1.5 shrink-0 rounded-full bg-club" />
              {issue}
            </li>
          ))}
        </ul>
      )}

      <div className="border-t border-line pt-4">
        <h3 className="font-mono text-[10px] text-fog uppercase">Cover</h3>
        {slot.backups.length === 0 && slot.unrated.length === 0 && (
          <p className="mt-1 text-sm">Nobody else in the squad plays here.</p>
        )}
        <ul className="mt-1.5 space-y-1 text-sm">
          {slot.backups.map((b) => (
            <li key={b.player_id} className="flex justify-between">
              <span>{b.name}</span>
              <span className="font-mono text-fog tabular">{b.fit}</span>
            </li>
          ))}
          {slot.unrated.map((name) => (
            <li key={name} className="flex justify-between text-fog">
              <span>{name}</span>
              <span className="font-mono text-[11px]">no minutes</span>
            </li>
          ))}
        </ul>
      </div>

      <Link
        href={`/recruit${query({ slot: slot.key })}`}
        className="inline-flex items-center gap-2 rounded bg-club px-4 py-2 text-sm font-semibold text-chalk hover:brightness-110"
      >
        Find candidates for {slot.key}
      </Link>
    </section>
  );
}

export default function LineupPage() {
  const { data: club } = useApi<Club>("/api/club");
  const [formation, setFormation] = useState<string | null>(null);
  const active = formation ?? club?.formation ?? null;
  const { data, error, isLoading } = useApi<Lineup>(
    club ? `/api/lineup${query({ formation: active })}` : null,
    { keepPreviousData: true },
  );
  const [selected, setSelected] = useState<string | null>(null);
  const [tilted, setTilted] = useState(true);

  const slot = data?.slots.find((s) => s.key === selected) ?? data?.slots.find((s) => s.severity === "critical") ?? data?.slots[0];

  return (
    <div className="grid min-h-screen grid-cols-1 xl:grid-cols-[minmax(0,1fr)_340px]">
      <div className="px-4 py-6 md:px-8">
        <header className="mb-6 flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="font-display text-4xl leading-none font-black uppercase md:text-6xl">Best XI</h1>
            <p className="mt-2 max-w-xl text-sm text-fog">
              Numbers are role fit against {club?.peer_group.split("-").slice(1).join("-") ?? "league"} peers,
              shrunk toward 50 for small samples. Red marks a position that needs recruiting.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <label className="flex items-center gap-2 text-sm">
              <span className="text-fog">Formation</span>
              <select
                value={active ?? ""}
                onChange={(e) => setFormation(e.target.value)}
                className="rounded border border-line bg-turf px-2 py-1.5 font-mono text-sm"
              >
                {club?.formations.map((f) => (
                  <option key={f} value={f}>
                    {f}
                  </option>
                ))}
              </select>
            </label>
            <button
              type="button"
              onClick={() => setTilted((t) => !t)}
              aria-pressed={!tilted}
              className="rounded border border-line px-3 py-1.5 text-sm text-fog hover:text-chalk"
            >
              {tilted ? "Flat view" : "Camera view"}
            </button>
          </div>
        </header>

        {error && !data && <p className="text-sm">The lineup couldn&apos;t be built: {error.message}</p>}
        {isLoading && !data && <div className="aspect-[105/68] w-full animate-pulse rounded-sm bg-turf" />}
        {data && (
          <Pitch slots={data.slots} selected={slot?.key ?? null} onSelect={setSelected} tilted={tilted} />
        )}

        {data && data.priorities.length > 0 && (
          <section aria-labelledby="priorities" className="mt-8">
            <h2 id="priorities" className="font-mono text-[10px] text-fog uppercase">
              Recruitment priorities
            </h2>
            <ol className="mt-2 divide-y divide-line border-y border-line">
              {data.priorities.map((p) => (
                <li key={p.slot}>
                  <button
                    type="button"
                    onClick={() => setSelected(p.slot)}
                    className="grid w-full grid-cols-[3.5rem_5.5rem_1fr] items-baseline gap-3 py-2.5 text-left text-sm hover:bg-turf"
                  >
                    <span className="font-display text-xl font-black">{p.slot}</span>
                    <span className={`font-mono text-[11px] uppercase ${p.severity === "critical" || p.severity === "high" ? "text-club" : "text-fog"}`}>
                      {SEVERITY_LABEL[p.severity] ?? p.severity}
                    </span>
                    <span className="text-fog">{p.issues.join("; ")}</span>
                  </button>
                </li>
              ))}
            </ol>
          </section>
        )}
      </div>

      <aside className="border-t border-line bg-turf/40 px-5 py-6 xl:border-t-0 xl:border-l">
        {slot ? <SlotPanel slot={slot} /> : <p className="text-sm text-fog">Loading the squad…</p>}
      </aside>
    </div>
  );
}
