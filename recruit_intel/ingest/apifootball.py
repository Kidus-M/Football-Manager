"""API-Football adapter (api-sports.io): defensive stats for every league.

The only free, permitted source we found with tackles, interceptions, blocks,
duels and dribbled-past for players outside the Premier League. Needs a key
(free plan: 100 requests a day) in `.env` as API_FOOTBALL_KEY.

Players are fetched club by club (/teams, then /players?team=, ~2 pages a
squad) because the free plan refuses /players pages past 3; the top five
leagues take ~200 requests, about two days of the free quota. Every page is cached
on disk, so a rerun only spends requests on pages not fetched yet: run
`recruit ingest --apifootball` once a day until it reports nothing left.

The quota running out comes back as HTTP 200 with an `errors` field; such
responses are never cached, and the run stops cleanly.

The free plan only covers seasons 2022-2024, so on a free key the newest
season it allows is loaded instead of the current one (see usable_season),
and features/per90.py uses it as a labelled, older fallback for defensive
metrics.
"""

import json
import os
import re
from pathlib import Path

import pandas as pd

from recruit_intel import ROOT
from recruit_intel.ingest.http import DAY, CachedClient

SOURCE = "apifootball"
BASE = "https://v3.football.api-sports.io"

# Country code used in club configs -> (API-Football league id, competition_id)
LEAGUES = {
    "ENG": (39, "ENG-Premier League"),
    "ESP": (140, "ESP-La Liga"),
    "ITA": (135, "ITA-Serie A"),
    "GER": (78, "GER-Bundesliga"),
    "FRA": (61, "FRA-Ligue 1"),
    "NED": (88, "NED-Eredivisie"),
    "POR": (94, "POR-Liga Portugal"),
    "BEL": (144, "BEL-Pro League"),
}

POSITIONS = {"Goalkeeper": "GK", "Defender": "DEF", "Midfielder": "MID", "Attacker": "FWD"}

# (section, field) in a statistics entry -> canonical metric
METRIC_FIELDS = {
    ("tackles", "total"): "tackles",
    ("tackles", "interceptions"): "interceptions",
    ("tackles", "blocks"): "blocks",
    ("duels", "won"): "duels_won",
    ("duels", "total"): "duels",
    ("dribbles", "success"): "dribbles_completed",
    ("dribbles", "past"): "dribbled_past",
    ("fouls", "committed"): "fouls_committed",
    ("passes", "total"): "passes",
    ("passes", "key"): "key_passes_af",
}


class QuotaExhausted(RuntimeError):
    """The daily request limit is used up; rerun tomorrow."""


class PlanRestricted(RuntimeError):
    """The key's plan doesn't cover the requested season. The free plan says
    e.g. "try from 2022 to 2024"; `latest` is the newest season it allows."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        years = [int(y) for y in re.findall(r"\b(20\d\d)\b", message)]
        self.latest = max(years) if years else None


def api_key() -> str | None:
    """API_FOOTBALL_KEY from the environment, or from the repo's .env file."""
    if key := os.environ.get("API_FOOTBALL_KEY"):
        return key.strip()
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "API_FOOTBALL_KEY" and value.strip():
                return value.strip().strip("\"'")
    return None


def make_client(key: str, ttl: float = 7 * DAY) -> CachedClient:
    return CachedClient(SOURCE, ttl=ttl, min_interval=6.5, headers={"x-apisports-key": key})


MAX_FREE_PAGE = 3  # the free plan refuses page > 3 on /players


def _get(client: CachedClient, path: str, params: dict) -> dict:
    """GET an endpoint; API errors (HTTP 200 + `errors`) raise and are never cached."""
    url = f"{BASE}/{path}"
    body = json.loads(client.get_text(url, params))
    errors = body.get("errors")
    if errors:  # never keep an error response in the cache
        client._path(url, params).unlink(missing_ok=True)
        text = json.dumps(errors)
        if isinstance(errors, dict) and "plan" in errors:
            raise PlanRestricted(str(errors["plan"]))
        if "limit" in text.lower() or "requests" in text.lower():
            raise QuotaExhausted(text)
        raise RuntimeError(f"API-Football error: {text}")
    return body


def fetch_page(client: CachedClient, league_id: int, season: int, page: int) -> dict:
    """One page of a league's /players (20 players). Paid plans only past page 3."""
    return _get(client, "players", {"league": league_id, "season": season, "page": page})


def fetch_teams(client: CachedClient, league_id: int, season: int) -> list[dict]:
    body = _get(client, "teams", {"league": league_id, "season": season})
    return [entry["team"] for entry in body.get("response", [])]


def fetch_team_players(client: CachedClient, team_id: int, season: int) -> list[dict]:
    """All /players pages for one club-season (a squad is ~2 pages)."""
    params = {"team": team_id, "season": season}
    pages = [_get(client, "players", {**params, "page": 1})]
    total = int((pages[0].get("paging") or {}).get("total") or 1)
    for page in range(2, min(total, MAX_FREE_PAGE) + 1):
        pages.append(_get(client, "players", {**params, "page": page}))
    return pages


def _plan_file(client: CachedClient) -> Path:
    return client.cache_dir / "plan.json"


def usable_season(client: CachedClient, wanted: int) -> int:
    """The newest season at or before `wanted` that the key's plan allows.

    Learned from the API's own refusal ("try from 2022 to 2024") the first
    time, then remembered in the cache folder so later runs don't spend a
    request finding out again."""
    plan = _plan_file(client)
    if plan.exists():
        latest = json.loads(plan.read_text()).get("latest_season")
        if latest is not None and latest < wanted:
            return latest
    league_id = LEAGUES["ENG"][0]
    try:
        fetch_teams(client, league_id, wanted)
        return wanted
    except PlanRestricted as exc:
        if exc.latest is None or exc.latest >= wanted:
            raise
        plan.parent.mkdir(parents=True, exist_ok=True)
        plan.write_text(json.dumps({"latest_season": exc.latest, "message": str(exc)}))
        return exc.latest


def _num(value) -> float | None:
    """None when API-Football didn't record the field (not the same as 0)."""
    if value is None or value == "":
        return None
    return float(str(value).rstrip("%"))


def parse(pages: list[dict], competition: str, season: int,
          league_id: int | None = None) -> dict[str, pd.DataFrame]:  # fmt: skip
    """/players responses -> source_players and long-form season totals.

    A player's statistics list has one entry per team-competition. Fetched
    per club, it also holds cups and European games, and a player who moved
    mid-season appears under both clubs; so entries are merged per player,
    de-duplicated by (team, league), and only `league_id`'s are summed."""
    merged: dict[str, tuple[dict, dict]] = {}
    for body in pages:
        for item in body.get("response", []):
            p = item["player"]
            _, entries = merged.setdefault(str(p["id"]), (p, {}))
            for s in item.get("statistics", []):
                lid = (s.get("league") or {}).get("id")
                if league_id is not None and lid != league_id:
                    continue
                entries[((s.get("team") or {}).get("id"), lid)] = s

    players, stats = [], []
    for pid, (p, by_key) in merged.items():
        entries = [s for s in by_key.values() if (s.get("games") or {}).get("minutes")]
        if not entries:
            continue
        minutes = sum(int(s["games"]["minutes"]) for s in entries)
        latest = entries[-1]
        players.append(
            {
                "source": SOURCE,
                "source_player_id": pid,
                "name": p.get("name") or f"{p.get('firstname', '')} {p.get('lastname', '')}",
                "team_name": latest["team"]["name"],
                "position": POSITIONS.get(latest["games"].get("position")),
                "birth_date": (p.get("birth") or {}).get("date"),
                "nationality": p.get("nationality"),
            }
        )
        team = ",".join(dict.fromkeys(s["team"]["name"] for s in entries))
        for (section, field), metric in METRIC_FIELDS.items():
            values = [_num((s.get(section) or {}).get(field)) for s in entries]
            recorded = [v for v in values if v is not None]
            if not recorded:  # not tracked for this player: missing, not zero
                continue
            stats.append(
                {
                    "source": SOURCE,
                    "source_player_id": pid,
                    "season": season,
                    "competition_id": competition,
                    "team_name": team,
                    "minutes": minutes,
                    "metric": metric,
                    "value": sum(recorded),
                }
            )
    players_df = pd.DataFrame(players)
    if not players_df.empty:
        players_df["birth_date"] = pd.to_datetime(players_df["birth_date"], errors="coerce").dt.date
    return {"source_players": players_df, "player_season_stats": pd.DataFrame(stats)}


def ingest_league(client: CachedClient, country: str, season: int) -> dict[str, pd.DataFrame]:
    """Every club's squad pages for one league-season (the free plan caps
    league-wide paging at 3 pages, so clubs are fetched one by one). Raises
    QuotaExhausted part-way; pages fetched so far stay cached, so the next
    run continues from there."""
    league_id, competition = LEAGUES[country]
    pages = []
    for team in fetch_teams(client, league_id, season):
        pages.extend(fetch_team_players(client, team["id"], season))
    return parse(pages, competition, season, league_id=league_id)
