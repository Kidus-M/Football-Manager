"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";
import { PlayerSearch } from "@/components/PlayerSearch";
import { type MapPoint, SimilarityMap } from "@/components/SimilarityMap";
import { league, money, type Player, query, useApi } from "@/lib/api";

type Similar = {
  player_id: string;
  name: string;
  club: string | null;
  group: string;
  competition_id: string;
  age: number | null;
  value_eur: number | null;
  distance: number;
}[];

const FILTERS = [
  ["younger", "Younger"],
  ["cheaper", "Cheaper"],
  ["same_position", "Same position"],
] as const;

export function SimilarView() {
  const router = useRouter();
  const id = useSearchParams().get("id");
  const [flags, setFlags] = useState({ younger: false, cheaper: false, same_position: false });
  const { data: points, error: mapError } = useApi<MapPoint[]>("/api/map");
  const { data: target } = useApi<Player>(id ? `/api/players/${id}` : null);
  const { data: similar, error } = useApi<Similar>(id ? `/api/similar/${id}${query({ ...flags, limit: 10 })}` : null, {
    keepPreviousData: true,
  });
  const pick = (pid: string) => router.replace(`/similar${query({ id: pid })}`);

  return (
    <div className="grid min-h-screen grid-cols-1 xl:grid-cols-[minmax(0,1fr)_360px]">
      <div className="px-4 py-6 md:px-8">
        <header className="mb-6">
          <h1 className="font-display text-4xl leading-none font-black uppercase md:text-6xl">Plays like</h1>
          <p className="mt-2 max-w-2xl text-sm text-fog">
            Players placed by how they contribute (shots, chance creation, build-up), not by listed position:
            neighbours on the map play alike, whatever their league. Last two seasons, 900+ minutes.
          </p>
        </header>
        {mapError && (
          <p className="text-sm">
            The style map needs the trained models: {mapError.message}
          </p>
        )}
        {points && <SimilarityMap points={points} target={id} matches={similar?.map((s) => s.player_id) ?? []} onPick={pick} />}
        {!points && !mapError && <div className="aspect-[800/560] w-full animate-pulse rounded-sm bg-turf" />}
      </div>

      <aside className="space-y-5 border-t border-line bg-turf/40 px-5 py-6 xl:border-t-0 xl:border-l">
        <PlayerSearch label="Find players like" placeholder="Players like…" onPick={(h) => pick(h.player_id)} />
        {!id && <p className="text-sm">Pick a player to see who plays like him, or click a dot on the map.</p>}
        {target && (
          <div>
            <div className="font-mono text-[10px] text-fog uppercase">Players like</div>
            <h2 className="font-display text-3xl leading-none font-black uppercase">{target.name}</h2>
            <p className="mt-1 font-mono text-[11px] text-fog">
              {[target.club, target.age && `${target.age} yrs`, money(target.value_eur)].filter(Boolean).join(" · ")}
            </p>
          </div>
        )}
        {id && (
          <fieldset className="flex flex-wrap gap-1.5">
            <legend className="sr-only">Filters</legend>
            {FILTERS.map(([key, label]) => (
              <button
                key={key}
                type="button"
                aria-pressed={flags[key]}
                onClick={() => setFlags((f) => ({ ...f, [key]: !f[key] }))}
                className={`rounded-full border px-2.5 py-1 text-xs ${flags[key] ? "border-chalk text-chalk" : "border-line text-fog"}`}
              >
                {label}
              </button>
            ))}
          </fieldset>
        )}
        {error && <p className="text-sm">{error.message}</p>}
        {similar && similar.length === 0 && <p className="text-sm">No player matches these filters. Turn one off.</p>}
        {similar && similar.length > 0 && (
          <ol className="divide-y divide-line border-y border-line">
            {similar.map((s) => (
              <li key={s.player_id} className="flex items-baseline justify-between gap-3 py-2.5">
                <div className="min-w-0">
                  <button type="button" onClick={() => pick(s.player_id)} className="font-semibold hover:underline">
                    {s.name}
                  </button>
                  <div className="truncate font-mono text-[11px] text-fog">
                    {[s.group, s.club, league(s.competition_id), s.age && `${s.age}`, money(s.value_eur)].filter(Boolean).join(" · ")}
                  </div>
                </div>
                <Link
                  href={`/compare${query({ ids: [id as string, s.player_id] })}`}
                  className="shrink-0 text-xs text-fog underline hover:text-chalk"
                >
                  Compare
                </Link>
              </li>
            ))}
          </ol>
        )}
      </aside>
    </div>
  );
}
