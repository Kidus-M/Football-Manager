"use client";

import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useDeferredValue, useState } from "react";
import { type Club, type Lineup, league, money, query, useApi } from "@/lib/api";

type Result = {
  slot: string;
  role: string;
  starter: { name: string; fit: number | null } | null;
  candidates: {
    player_id: string;
    name: string;
    club: string | null;
    competition_id: string;
    age: number | null;
    value_eur: number | null;
    contract_year: number | null;
    fit: number;
    fit_lo: number;
    fit_hi: number;
    confidence: number;
    own_data: boolean;
    score: number;
    vs_starter: number | null;
    why: string;
  }[];
};

const LEAGUES = [
  ["ENG", "Premier League"],
  ["ESP", "La Liga"],
  ["ITA", "Serie A"],
  ["GER", "Bundesliga"],
  ["FRA", "Ligue 1"],
] as const;

// Value slider runs on a log scale: equal steps feel equal from €2M to €150M.
const VALUE_STEPS = [2, 5, 10, 15, 20, 30, 40, 50, 60, 75, 100, 150];

function Range({
  id,
  label,
  value,
  display,
  min,
  max,
  onChange,
}: {
  id: string;
  label: string;
  value: number;
  display: string;
  min: number;
  max: number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <label htmlFor={id} className="text-sm">
          {label}
        </label>
        <output htmlFor={id} className="font-mono text-xs tabular">
          {display}
        </output>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="mt-2 w-full accent-[var(--club)]"
      />
    </div>
  );
}

/** The parts of the ranking's "why" that the row doesn't already show:
 * how the fit was adjusted, how far its data can be trusted and style fit. */
function explain(why: string, competition: string) {
  const parts = why
    .split(", ")
    .filter((p) => /^fit |^range |^confidence |^defence estimated|^style fit/.test(p));
  const fit = parts[0]?.match(/^fit (\d+) in .+ -> (\d+)$/);
  if (fit) parts[0] = `${league(competition)} fit ${fit[1]}, adjusted ${fit[2]}`;
  return parts
    .map((p) => (p === "defence estimated" ? "defending estimated: no data of his own" : p.replace(/^range /, "likely ")))
    .join(" · ");
}

/** Solid to the bottom of the likely range, faint across it, a tick at the fit. */
function FitBar({ fit, lo, hi, starter }: { fit: number; lo: number; hi: number; starter: number | null }) {
  const ranged = hi - lo >= 1;
  return (
    <div className="relative h-1.5 w-full rounded-full bg-line" aria-hidden>
      <div className="absolute inset-y-0 left-0 rounded-full bg-chalk" style={{ width: `${ranged ? lo : fit}%` }} />
      {ranged && (
        <>
          <div className="absolute inset-y-0 rounded-full bg-chalk/30" style={{ left: `${lo}%`, width: `${hi - lo}%` }} />
          <div className="absolute -inset-y-0.5 w-0.5 bg-chalk" style={{ left: `${fit}%` }} />
        </>
      )}
      {starter != null && (
        <div className="absolute -inset-y-1 w-0.5 bg-club" style={{ left: `${starter}%` }} title="Current starter" />
      )}
    </div>
  );
}

export function RecruitView() {
  const router = useRouter();
  const params = useSearchParams();
  const reduce = useReducedMotion();
  const { data: club } = useApi<Club>("/api/club");
  const { data: lineup } = useApi<Lineup>(club ? `/api/lineup${query({ formation: club.formation })}` : null);

  const [maxAge, setMaxAge] = useState<number | null>(null);
  const [valueStep, setValueStep] = useState(VALUE_STEPS.length - 1);
  const [contractUntil, setContractUntil] = useState<number | null>(null);
  const [leagues, setLeagues] = useState<string[]>(LEAGUES.map(([c]) => c));
  const [upgrades, setUpgrades] = useState(true);
  const [ownData, setOwnData] = useState(false);

  const priorities = lineup?.priorities.map((p) => p.slot) ?? [];
  const ordered = lineup
    ? [...lineup.slots].sort((a, b) => {
        const pa = priorities.indexOf(a.key);
        const pb = priorities.indexOf(b.key);
        return (pa < 0 ? 99 : pa) - (pb < 0 ? 99 : pb);
      })
    : [];
  const slot = params.get("slot") ?? priorities[0] ?? null;
  const age = maxAge ?? club?.recruitment.max_age ?? 27;
  const season = club?.season ?? 2026;
  const filters = useDeferredValue({
    slot,
    max_age: age,
    max_value_m: valueStep === VALUE_STEPS.length - 1 ? null : VALUE_STEPS[valueStep],
    contract_until: contractUntil,
    leagues: leagues.length === LEAGUES.length ? null : leagues,
    upgrades,
    own_data_only: ownData,
    limit: 15,
  });
  const { data, error, isLoading } = useApi<Result>(slot ? `/api/search${query(filters)}` : null, {
    keepPreviousData: true,
  });

  const toggleLeague = (code: string) =>
    setLeagues((ls) => (ls.includes(code) ? (ls.length > 1 ? ls.filter((l) => l !== code) : ls) : [...ls, code]));

  return (
    <div className="grid min-h-screen grid-cols-1 lg:grid-cols-[300px_minmax(0,1fr)]">
      <aside className="space-y-7 border-b border-line bg-turf/40 px-5 py-6 lg:border-r lg:border-b-0">
        <div>
          <h1 className="font-display text-4xl leading-none font-black uppercase">Recruit</h1>
          <p className="mt-2 text-sm text-fog">Candidates for one position, ranked as you change the filters.</p>
        </div>

        <fieldset>
          <legend className="font-mono text-[10px] text-fog uppercase">Position</legend>
          <div className="mt-2 grid grid-cols-4 gap-1.5">
            {ordered.map((s) => (
              <button
                key={s.key}
                type="button"
                aria-pressed={s.key === slot}
                onClick={() => router.replace(`/recruit${query({ slot: s.key })}`)}
                className={`rounded border px-1 py-1.5 font-display text-lg font-black ${
                  s.key === slot ? "border-chalk bg-chalk text-night" : "border-line hover:border-fog"
                } ${s.severity === "critical" || s.severity === "high" ? "underline decoration-club decoration-2 underline-offset-4" : ""}`}
              >
                {s.key}
              </button>
            ))}
          </div>
          <p className="mt-2 text-xs text-fog">Underlined positions are recruitment priorities.</p>
        </fieldset>

        <Range id="age" label="Maximum age" value={age} display={`${age}`} min={18} max={34} onChange={setMaxAge} />
        <Range
          id="value"
          label="Maximum market value"
          value={valueStep}
          display={valueStep === VALUE_STEPS.length - 1 ? "Any" : `€${VALUE_STEPS[valueStep]}M`}
          min={0}
          max={VALUE_STEPS.length - 1}
          onChange={setValueStep}
        />
        <Range
          id="contract"
          label="Contract ends by"
          value={contractUntil ?? season + 6}
          display={contractUntil == null ? "Any" : `${contractUntil}`}
          min={season + 1}
          max={season + 6}
          onChange={(v) => setContractUntil(v === season + 6 ? null : v)}
        />

        <fieldset>
          <legend className="text-sm">Leagues</legend>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {LEAGUES.map(([code, name]) => (
              <button
                key={code}
                type="button"
                aria-pressed={leagues.includes(code)}
                onClick={() => toggleLeague(code)}
                className={`rounded-full border px-2.5 py-1 text-xs ${
                  leagues.includes(code) ? "border-chalk text-chalk" : "border-line text-fog"
                }`}
              >
                {name}
              </button>
            ))}
          </div>
        </fieldset>

        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={upgrades} onChange={(e) => setUpgrades(e.target.checked)} className="accent-[var(--club)]" />
          Only players who&apos;d improve on the starter
        </label>
        <label className="flex items-start gap-2 text-sm">
          <input
            type="checkbox"
            checked={ownData}
            onChange={(e) => setOwnData(e.target.checked)}
            className="mt-1 accent-[var(--club)]"
          />
          <span>
            Only players with defensive data of their own
            <span className="mt-0.5 block text-xs text-fog">
              Defensive numbers for players with none of their own are estimated from playing style and team. They
              still rank, but count for less.
            </span>
          </span>
        </label>
      </aside>

      <section className="px-4 py-6 md:px-8" aria-live="polite" aria-busy={isLoading}>
        {data && (
          <header className="mb-5 flex flex-wrap items-end justify-between gap-3">
            <div>
              <div className="font-mono text-[10px] text-fog uppercase">{data.role}</div>
              <h2 className="font-display text-5xl leading-none font-black uppercase">{data.slot}</h2>
            </div>
            {data.starter && (
              <p className="text-sm text-fog">
                Current starter <span className="text-chalk">{data.starter.name}</span>, fit{" "}
                <span className="font-mono text-chalk">{data.starter.fit?.toFixed(0) ?? "–"}</span>
                <span className="ml-2 inline-block h-3 w-0.5 bg-club align-middle" aria-hidden /> on the bars
              </p>
            )}
          </header>
        )}
        {error && <p className="text-sm">The search failed: {error.message}</p>}
        {data && data.candidates.length === 0 && (
          <p className="max-w-md text-sm">
            No player matches these filters. Raise the age or value limit, add leagues, or include players who
            wouldn&apos;t improve on the starter.
          </p>
        )}
        <ol className="divide-y divide-line border-y border-line">
          <AnimatePresence initial={false}>
            {data?.candidates.map((c, i) => (
              <motion.li
                key={c.player_id}
                layout={!reduce}
                initial={reduce ? false : { opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={reduce ? undefined : { opacity: 0 }}
                transition={{ duration: 0.35, ease: [0.4, 0, 0.2, 1] }}
                className="grid grid-cols-[2rem_minmax(0,1fr)_4.5rem] items-center gap-x-4 gap-y-1.5 py-3 md:grid-cols-[2rem_minmax(0,1.3fr)_minmax(0,1fr)_4.5rem]"
              >
                <span className="font-mono text-xs text-fog tabular">{i + 1}</span>
                <div className="min-w-0">
                  <Link
                    href={`/compare${query({ ids: [c.player_id] })}`}
                    className="font-semibold hover:underline"
                  >
                    {c.name}
                  </Link>
                  <div className="truncate font-mono text-[11px] text-fog">
                    {[c.club, c.age && `${c.age} yrs`, money(c.value_eur), c.contract_year && `to ${c.contract_year}`]
                      .filter(Boolean)
                      .join(" · ")}
                  </div>
                </div>
                <div className="col-span-3 col-start-2 md:col-span-1 md:col-start-auto">
                  <FitBar fit={c.fit} lo={c.fit_lo} hi={c.fit_hi} starter={data.starter?.fit ?? null} />
                  <div className="mt-1 truncate text-[11px] text-fog" title={c.why}>
                    {explain(c.why, c.competition_id)}
                  </div>
                </div>
                <div className="row-start-1 text-right md:row-start-auto">
                  <div className="font-display text-3xl leading-none font-black tabular">{c.score.toFixed(0)}</div>
                  {c.vs_starter != null && (
                    <div className={`font-mono text-[11px] tabular ${c.vs_starter > 0 ? "text-club-2" : "text-fog"}`}>
                      {c.vs_starter > 0 ? "+" : ""}
                      {c.vs_starter.toFixed(0)} fit
                    </div>
                  )}
                </div>
              </motion.li>
            ))}
          </AnimatePresence>
        </ol>
      </section>
    </div>
  );
}
