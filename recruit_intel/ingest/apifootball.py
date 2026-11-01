"""API-Football adapter (api-sports.io): defensive stats for every league.

The only free, permitted source we found with tackles, interceptions, blocks,
duels and dribbled-past for players outside the Premier League. Needs a key
(free plan: 100 requests a day) in `.env` as API_FOOTBALL_KEY.

/players returns 20 players per page, so a league-season is ~30-40 pages and
the five leagues take about two days of the free quota. Every page is cached
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


def fetch_page(client: CachedClient, league_id: int, season: int, page: int) -> dict:
    url, params = f"{BASE}/players", {"league": league_id, "season": season, "page": page}
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
        fetch_page(client, league_id, wanted, 1)
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


def parse(pages: list[dict], competition: str, season: int) -> dict[str, pd.DataFrame]:
    """/players responses -> source_players and long-form season totals.

    A player's statistics list has one entry per team-competition; only
    entries for this league count, summed over teams (mid-season moves)."""
    players, stats = [], []
    for body in pages:
        for item in body.get("response", []):
            p = item["player"]
            pid = str(p["id"])
            entries = [s for s in item.get("statistics", [])
                       if (s.get("games") or {}).get("minutes")]  # fmt: skip
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
                total = sum(recorded)
                stats.append(
                    {
                        "source": SOURCE,
                        "source_player_id": pid,
                        "season": season,
                        "competition_id": competition,
                        "team_name": team,
                        "minutes": minutes,
                        "metric": metric,
                        "value": total,
                    }
                )
    players_df = pd.DataFrame(players).drop_duplicates("source_player_id", keep="last")
    if not players_df.empty:
        players_df["birth_date"] = pd.to_datetime(players_df["birth_date"], errors="coerce").dt.date
    return {"source_players": players_df, "player_season_stats": pd.DataFrame(stats)}


def ingest_league(client: CachedClient, country: str, season: int) -> dict[str, pd.DataFrame]:
    """All pages of one league-season. Raises QuotaExhausted part-way; pages
    fetched so far stay cached, so the next run continues from there."""
    league_id, competition = LEAGUES[country]
    first = fetch_page(client, league_id, season, 1)
    pages = [first]
    total = int((first.get("paging") or {}).get("total") or 1)
    for page in range(2, total + 1):
        pages.append(fetch_page(client, league_id, season, page))
    return parse(pages, competition, season)
