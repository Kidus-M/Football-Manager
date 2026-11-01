"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import useSWR from "swr";
import { PlayerSearch } from "@/components/PlayerSearch";
import { Radar, SERIES_COLOURS } from "@/components/Radar";
import { fetcher, league, money, type Player, query } from "@/lib/api";

const MAX = 3;

function PlayerCard({ p, colour, onRemove }: { p: Player; colour: string; onRemove: () => void }) {
  return (
    <article className="border-t-2 pt-3" style={{ borderColor: colour }}>
      <div className="flex items-start justify-between gap-2">
        <div>
          <h2 className="font-display text-3xl leading-none font-black uppercase">{p.name}</h2>
          <p className="mt-1 font-mono text-[11px] text-fog">
            {[p.detailed_position ?? p.group_label, p.club, league(p.competition)].filter(Boolean).join(" · ")}
          </p>
        </div>
        <button type="button" onClick={onRemove} className="text-xs text-fog hover:text-chalk" aria-label={`Remove ${p.name}`}>
          Remove
        </button>
      </div>
      <dl className="mt-3 grid grid-cols-4 gap-2 font-mono text-[11px] text-fog">
        {[
          ["Age", p.age ?? "–"],
          ["Value", money(p.value_eur)],
          ["Contract", p.contract_year ?? "–"],
          ["Minutes", p.minutes],
        ].map(([k, v]) => (
          <div key={k as string}>
            <dt className="uppercase">{k}</dt>
            <dd className="text-chalk tabular">{v}</dd>
          </div>
        ))}
      </dl>
      {p.qualified === false && (
        <p className="mt-2 text-xs text-club">Small sample: percentiles are a preview, not a verdict.</p>
      )}
      {p.defence_season != null && (
        <p className="mt-2 text-xs text-fog">
          Interceptions, blocks, duels and dribbled-past are from {p.defence_season}-
          {String((p.defence_season + 1) % 100).padStart(2, "0")}, the newest season the free data plan allows.
        </p>
      )}
      <p className="mt-3 text-sm">
        <span className="text-fog">Best roles </span>
        {p.roles.map((r) => `${r.role} ${r.fit}`).join(" · ") || "–"}
      </p>
      {p.strengths.length > 0 && (
        <p className="mt-1 text-sm">
          <span className="text-fog">Strengths </span>
          {p.strengths.join(", ")}
        </p>
      )}
      {p.weaknesses.length > 0 && (
        <p className="mt-1 text-sm">
          <span className="text-fog">Weaknesses </span>
          {p.weaknesses.join(", ")}
        </p>
      )}
      <p className="mt-2 font-mono text-[10px] text-fog uppercase">
        vs {p.peers ?? "–"} {p.group_label?.toLowerCase()} in {league(p.competition)}
        {p.basis === "own_league" ? " · last two seasons" : " · this season"}
      </p>
    </article>
  );
}

export function CompareView() {
  const router = useRouter();
  const ids = useSearchParams().getAll("ids").slice(0, MAX);
  const { data: players = [], error } = useSWR<Player[]>(
    ids.length ? ["compare", ...ids] : null,
    () => Promise.all(ids.map((id) => fetcher<Player>(`/api/players/${id}`))),
    { keepPreviousData: true, revalidateOnFocus: false },
  );
  const setIds = (next: string[]) => router.replace(`/compare${query({ ids: next })}`);

  // The radar uses the first player's position metrics; everyone is plotted
  // on those axes so the shapes are comparable.
  const axes = players[0]?.percentiles ?? [];
  const series = players.map((p) => ({
    id: p.player_id,
    name: p.name,
    values: axes.map((a) => p.all_percentiles[a.metric] ?? null),
  }));

  return (
    <div className="px-4 py-6 md:px-8">
      <header className="mb-6 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="font-display text-4xl leading-none font-black uppercase md:text-6xl">Compare</h1>
          <p className="mt-2 max-w-xl text-sm text-fog">
            Percentiles against each player&apos;s own league peers in the same position. Add up to three players.
          </p>
        </div>
        {ids.length < MAX && (
          <div className="w-full max-w-xs">
            <PlayerSearch
              label="Add a player to compare"
              placeholder="Add a player"
              onPick={(h) => !ids.includes(h.player_id) && setIds([...ids, h.player_id])}
            />
          </div>
        )}
      </header>

      {ids.length === 0 && (
        <p className="max-w-md text-sm">
          Search for a player above, or open one from the lineup or the recruitment shortlist.
        </p>
      )}
      {error && <p className="text-sm">A player couldn&apos;t be loaded: {error.message}</p>}

      {players.length > 0 && (
        <div className="grid gap-10 lg:grid-cols-[minmax(0,520px)_minmax(0,1fr)]">
          <div>
            <Radar axes={axes.map((a) => a.label)} series={series} />
            <table className="mt-4 w-full text-sm">
              <caption className="sr-only">Percentiles by metric</caption>
              <thead>
                <tr className="font-mono text-[10px] text-fog uppercase">
                  <th className="py-1 text-left font-normal">Metric</th>
                  {players.map((p, i) => (
                    <th key={p.player_id} className="py-1 text-right font-normal" style={{ color: SERIES_COLOURS[i] }}>
                      {p.name.split(" ").slice(-1)[0]}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-line">
                {axes.map((a, ai) => (
                  <tr key={a.metric}>
                    <td className="py-1.5">{a.label}</td>
                    {series.map((s) => (
                      <td key={s.id} className="py-1.5 text-right font-mono tabular">
                        {s.values[ai] == null ? "n/a" : Math.round(s.values[ai] as number)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="space-y-8">
            {players.map((p, i) => (
              <PlayerCard
                key={p.player_id}
                p={p}
                colour={SERIES_COLOURS[i]}
                onRemove={() => setIds(ids.filter((id) => id !== p.player_id))}
              />
            ))}
            {players[0] && (
              <Link href={`/similar${query({ id: players[0].player_id })}`} className="inline-block text-sm underline">
                Players who play like {players[0].name}
              </Link>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
