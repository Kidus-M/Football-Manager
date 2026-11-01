"use client";

import useSWR, { type SWRConfiguration } from "swr";

/** Shapes returned by recruit_intel/api (see service.py). */
export type Club = {
  club: string;
  season: number;
  season_label: string;
  formation: string | null;
  peer_group: string;
  theme: { accent: string; secondary: string | null; ink: string | null } | null;
  recruitment: { pool: string[]; max_age: number; max_value_eur: number | null; min_minutes: number };
  formations: string[];
};

export type Candidate = { player_id: string; name: string; fit: number };

export type Slot = {
  key: string;
  x: number;
  y: number;
  groups: string[];
  role: string;
  starter: (Candidate & { minutes: number }) | null;
  backups: Candidate[];
  unrated: string[];
  age: number | null;
  contract: number | null;
  severity: "critical" | "high" | "medium" | "low" | null;
  issues: string[];
};

export type Lineup = {
  formation: string;
  slots: Slot[];
  priorities: { slot: string; role: string; severity: string; issues: string[] }[];
};

export type PlayerHit = {
  player_id: string;
  name: string;
  club: string | null;
  group: string | null;
  competition_id: string;
  age: number | null;
};

export type Player = {
  player_id: string;
  name: string;
  group: string;
  group_label: string | null;
  detailed_position: string | null;
  competition: string | null;
  basis: "peer_group" | "own_league";
  minutes: number;
  qualified: boolean | null;
  peers: number | null;
  club: string | null;
  age: number | null;
  value_eur: number | null;
  contract_year: number | null;
  percentiles: { metric: string; label: string; value: number | null }[];
  strengths: string[];
  weaknesses: string[];
  roles: { role: string; fit: number }[];
};

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

export async function fetcher<T>(path: string): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path);
  } catch {
    throw new ApiError("The analysis API isn't reachable.", 0);
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new ApiError(body?.detail ?? `Request failed (${response.status})`, response.status);
  }
  return response.json();
}

/** SWR over the API. `path` null = don't fetch yet. */
export function useApi<T>(path: string | null, config?: SWRConfiguration<T>) {
  return useSWR<T>(path, fetcher, { revalidateOnFocus: false, ...config });
}

export function query(params: Record<string, string | number | boolean | string[] | null | undefined>) {
  const q = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === null || value === undefined || value === "" || value === false) continue;
    if (Array.isArray(value)) value.forEach((v) => q.append(key, v));
    else q.set(key, String(value));
  }
  const s = q.toString();
  return s ? `?${s}` : "";
}

export const money = (eur: number | null | undefined) =>
  eur == null ? "–" : eur >= 1e6 ? `€${(eur / 1e6).toFixed(eur >= 1e7 ? 0 : 1)}M` : `€${Math.round(eur / 1e3)}K`;

export const league = (competition: string | null | undefined) =>
  competition ? competition.split("-").slice(1).join("-") : "–";
