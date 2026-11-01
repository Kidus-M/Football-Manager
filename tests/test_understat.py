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


def test_leagues_for_pool():
    assert understat.leagues_for_pool(["ENG", "NED", "ITA"]) == ["EPL", "Serie_A"]
