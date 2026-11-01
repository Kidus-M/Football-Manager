"use client";

import { useDeferredValue, useId, useState } from "react";
import { league, type PlayerHit, query, useApi } from "@/lib/api";

/** Type-ahead over every rated player in the top five leagues. */
export function PlayerSearch({
  label,
  onPick,
  placeholder = "Search a player",
}: {
  label: string;
  onPick: (hit: PlayerHit) => void;
  placeholder?: string;
}) {
  const id = useId();
  const [text, setText] = useState("");
  const q = useDeferredValue(text.trim());
  const { data: hits = [] } = useApi<PlayerHit[]>(q.length >= 2 ? `/api/players${query({ q, limit: 6 })}` : null);

  return (
    <div className="relative">
      <label htmlFor={id} className="sr-only">
        {label}
      </label>
      <input
        id={id}
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder={placeholder}
        autoComplete="off"
        className="w-full rounded border border-line bg-turf px-3 py-2 text-sm placeholder:text-fog"
      />
      {q.length >= 2 && hits.length > 0 && (
        <ul className="absolute z-20 mt-1 w-full overflow-hidden rounded border border-line bg-turf shadow-xl">
          {hits.map((h) => (
            <li key={h.player_id}>
              <button
                type="button"
                onClick={() => {
                  onPick(h);
                  setText("");
                }}
                className="flex w-full items-baseline justify-between gap-3 px-3 py-2 text-left text-sm hover:bg-turf-2 focus-visible:bg-turf-2"
              >
                <span>{h.name}</span>
                <span className="font-mono text-[11px] text-fog">
                  {[h.group, h.club, league(h.competition_id)].filter(Boolean).join(" · ")}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
