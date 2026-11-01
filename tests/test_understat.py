import pytest

from recruit_intel.ingest import understat

ROW = {
    "id": "8260",
    "player_name": "Erling Haaland",
    "games": "9",
    "time": "801",
    "goals": "9",
    "xG": "7.7",
    "assists": "0",
    "xA": "1.78",
    "shots": "34",
    "key_passes": "5",
    "position": "F S",
    "team_title": "Manchester City",
    "npg": "9",
    "npxG": "7.7",
    "xGChain": "9.8",
    "xGBuildup": "1.9",
}


@pytest.mark.parametrize(
    "raw,expected",
    [("F S", "FWD"), ("GK", "GK"), ("D M S", "DEF"), ("S", None), ("M", "MID")],
)
def test_broad_position(raw, expected):
    assert understat.broad_position(raw) == expected


def test_parse_long_form_stats():
    out = understat.parse({"players": [ROW]}, "EPL", 2026)
    stats = out["player_season_stats"]
    assert len(stats) == len(understat.METRIC_FIELDS)
    xg = stats.set_index("metric").loc["xg"]
    assert xg.value == 7.7 and xg.minutes == 801
    assert xg.competition_id == "ENG-Premier League"


def test_mid_season_move_keeps_latest_club():
    row = {**ROW, "team_title": "Brentford,Manchester United"}
    out = understat.parse({"players": [row]}, "EPL", 2026)
    assert out["source_players"].team_name.iloc[0] == "Manchester United"


TEAMS = {
    "71": {
        "id": "71",
        "title": "Aston Villa",
        "history": [
            {"h_a": "h", "xG": 0.32, "xGA": 1.4, "npxG": 0.32, "npxGA": 1.4,
             "ppda": {"att": 227, "def": 12}, "ppda_allowed": {"att": 146, "def": 24},
             "deep": 2, "deep_allowed": 6, "scored": 0, "missed": 0, "xpts": 0.43,
             "result": "d", "date": "2025-08-16 11:30:00", "pts": 1},
        ],
    }
}  # fmt: skip


def test_parse_teams_and_store():
    from recruit_intel.db import connect, upsert

    out = understat.parse({"players": [ROW], "teams": TEAMS}, "EPL", 2025)
    m = out["team_matches"].iloc[0]
    assert (m.team_name, m.home, m.ppda_passes, m.ppda_actions) == ("Aston Villa", True, 227, 12)
    assert (m.oppda_passes, m.deep_allowed, m.conceded) == (146, 6, 0)
    assert upsert(connect(":memory:"), "team_matches", out["team_matches"]) == 1


def test_leagues_for_pool():
    assert understat.leagues_for_pool(["ENG", "NED", "ITA"]) == ["EPL", "Serie_A"]
