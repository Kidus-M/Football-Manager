"""Fantasy Premier League adapter (unofficial public API, Premier League only).

Gives every current PL player with birth date, position, minutes and the
defensive counts (tackles, CBI, recoveries) that no other free source has
since FBref lost its Opta stats.
"""

import pandas as pd

from recruit_intel.ingest.http import CachedClient

SOURCE = "fpl"
COMPETITION = "ENG-Premier League"
BOOTSTRAP = "https://fantasy.premierleague.com/api/bootstrap-static/"

POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# FPL field -> canonical metric name (see features/metrics.py)
METRIC_FIELDS = {
    "goals_scored": "goals",
    "assists": "assists",
    "expected_goals": "xg",
    "expected_assists": "xa",
    "tackles": "tackles",
    "clearances_blocks_interceptions": "cbi",
    "recoveries": "recoveries",
    "defensive_contribution": "defensive_contribution",
    "saves": "saves",
    "goals_conceded": "goals_conceded",
    "expected_goals_conceded": "xg_conceded",
    "clean_sheets": "clean_sheets",
    "starts": "starts",
}


def fetch_bootstrap(client: CachedClient) -> dict:
    return client.get_json(BOOTSTRAP)


def season_of(bootstrap: dict) -> int:
    """Start year of the season the API is serving, from the first deadline."""
    return int(bootstrap["events"][0]["deadline_time"][:4])


def parse(bootstrap: dict) -> dict[str, pd.DataFrame]:
    """Turn bootstrap-static into rows for teams, source_players, player_season_stats."""
    season = season_of(bootstrap)
    team_names = {t["id"]: t["name"] for t in bootstrap["teams"]}

    teams = pd.DataFrame(
        {
            "source": SOURCE,
            "source_team_id": str(t["id"]),
            "name": t["name"],
            "competition_id": COMPETITION,
            "season": season,
        }
        for t in bootstrap["teams"]
    )

    players, stats = [], []
    for e in bootstrap["elements"]:
        if e.get("removed"):
            continue
        pid = str(e["id"])
        team = team_names[e["team"]]
        players.append(
            {
                "source": SOURCE,
                "source_player_id": pid,
                "name": f"{e['first_name']} {e['second_name']}".strip(),
                "team_name": team,
                "position": POSITIONS[e["element_type"]],
                "birth_date": e.get("birth_date"),
            }
        )
        for field, metric in METRIC_FIELDS.items():
            stats.append(
                {
                    "source": SOURCE,
                    "source_player_id": pid,
                    "season": season,
                    "competition_id": COMPETITION,
                    "team_name": team,
                    "minutes": e["minutes"],
                    "metric": metric,
                    "value": float(e[field] or 0),
                }
            )

    players_df = pd.DataFrame(players)
    players_df["birth_date"] = pd.to_datetime(players_df["birth_date"]).dt.date
    return {
        "teams": teams,
        "source_players": players_df,
        "player_season_stats": pd.DataFrame(stats),
    }
