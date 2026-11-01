"""Understat adapter: xG, xA, npxG, xGChain, xGBuildup per player-season.

Covers the top 5 leagues plus the Russian Premier League. Uses the JSON
endpoint the site's own pages call (needs an XHR header).
"""

import pandas as pd

from recruit_intel.ingest.http import DAY, CachedClient

SOURCE = "understat"
BASE = "https://understat.com"

# Understat league slug -> (country code used in club configs, competition_id)
LEAGUES = {
    "EPL": ("ENG", "ENG-Premier League"),
    "La_liga": ("ESP", "ESP-La Liga"),
    "Bundesliga": ("GER", "GER-Bundesliga"),
    "Serie_A": ("ITA", "ITA-Serie A"),
    "Ligue_1": ("FRA", "FRA-Ligue 1"),
    "RFPL": ("RUS", "RUS-Premier League"),
}

LEAGUES_BY_COUNTRY = {country: slug for slug, (country, _) in LEAGUES.items() if country != "RUS"}

POSITIONS = {"GK": "GK", "D": "DEF", "M": "MID", "F": "FWD"}

METRIC_FIELDS = {
    "goals": "goals",
    "npg": "npg",
    "assists": "assists",
    "xG": "xg",
    "npxG": "npxg",
    "xA": "xa",
    "shots": "shots",
    "key_passes": "key_passes",
    "xGChain": "xg_chain",
    "xGBuildup": "xg_buildup",
    "games": "games",
}


def leagues_for_pool(pool: list[str]) -> list[str]:
    """Understat leagues covering a club config's recruitment pool."""
    return [slug for slug, (country, _) in LEAGUES.items() if country in pool]


def broad_position(raw: str) -> str | None:
    """'F M S' -> 'FWD'. Understat lists positions played, 'S' meaning sub."""
    for token in raw.split():
        if token in POSITIONS:
            return POSITIONS[token]
    return None


def make_client(**kw) -> CachedClient:
    return CachedClient(SOURCE, headers={"X-Requested-With": "XMLHttpRequest"}, **kw)


def fetch_league(client: CachedClient, league: str, season: int) -> dict:
    return client.get_json(
        f"{BASE}/getLeagueData/{league}/{season}",
        headers={"Referer": f"{BASE}/league/{league}/{season}"},
    )


def parse(data: dict, league: str, season: int) -> dict[str, pd.DataFrame]:
    _, competition = LEAGUES[league]
    players, stats = [], []
    for p in data["players"]:
        pid = str(p["id"])
        # Players who moved mid-season are listed as "Old Club,New Club".
        team = p["team_title"].split(",")[-1]
        players.append(
            {
                "source": SOURCE,
                "source_player_id": pid,
                "name": p["player_name"],
                "team_name": team,
                "position": broad_position(p["position"]),
            }
        )
        for field, metric in METRIC_FIELDS.items():
            stats.append(
                {
                    "source": SOURCE,
                    "source_player_id": pid,
                    "season": season,
                    "competition_id": competition,
                    "team_name": p["team_title"],
                    "minutes": int(p["time"]),
                    "metric": metric,
                    "value": float(p[field]),
                }
            )
    out = {
        "source_players": pd.DataFrame(players),
        "player_season_stats": pd.DataFrame(stats),
    }
    if data.get("teams"):
        out["team_matches"] = parse_teams(data["teams"], competition, season)
    return out


def parse_teams(teams: dict, competition: str, season: int) -> pd.DataFrame:
    """Per-match team stats: the raw material for team style profiles."""
    rows = []
    for team in teams.values():
        for m in team["history"]:
            rows.append(
                {
                    "source": SOURCE,
                    "team_id": str(team["id"]),
                    "team_name": team["title"],
                    "season": season,
                    "competition_id": competition,
                    "match_date": pd.Timestamp(m["date"]),
                    "home": m["h_a"] == "h",
                    "xg": float(m["xG"]),
                    "xga": float(m["xGA"]),
                    "npxg": float(m["npxG"]),
                    "npxga": float(m["npxGA"]),
                    "ppda_passes": int(m["ppda"]["att"]),
                    "ppda_actions": int(m["ppda"]["def"]),
                    "oppda_passes": int(m["ppda_allowed"]["att"]),
                    "oppda_actions": int(m["ppda_allowed"]["def"]),
                    "deep": int(m["deep"]),
                    "deep_allowed": int(m["deep_allowed"]),
                    "scored": int(m["scored"]),
                    "conceded": int(m["missed"]),
                    "xpts": float(m["xpts"]),
                    "pts": int(m["pts"]),
                }
            )
    return pd.DataFrame(rows)


def ingest(league: str, season: int, current_season: int, client=None) -> dict[str, pd.DataFrame]:
    """Fetch and parse one league-season. Past seasons are cached for 30 days."""
    ttl = DAY if season >= current_season else 30 * DAY
    own = client is None
    client = client or make_client(ttl=ttl)
    try:
        return parse(fetch_league(client, league, season), league, season)
    finally:
        if own:
            client.close()
