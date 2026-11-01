from datetime import date

import pandas as pd
import pytest

from recruit_intel.db import connect, upsert
from recruit_intel.features.per90 import per90_table
from recruit_intel.features.positions import group_of

EPL = "ENG-Premier League"


@pytest.mark.parametrize(
    "detailed,broad,expected",
    [("Right-Back", "DEF", "FB"), ("Left Winger", "FWD", "W"), (None, "DEF", "CB"),
     (None, "MID", "CM"), (None, None, None)],
)  # fmt: skip
def test_group_of(detailed, broad, expected):
    assert group_of(detailed, broad) == expected


def _row(source, pid, metric, value, minutes, team="X"):
    return {"source": source, "source_player_id": pid, "season": 2026, "competition_id": EPL,
            "team_name": team, "minutes": minutes, "metric": metric, "value": value}  # fmt: skip


def test_each_source_uses_its_own_minutes_and_clubs_are_summed():
    con = connect(":memory:")
    upsert(con, "players", pd.DataFrame([
        {"player_id": "tm-1", "name": "A", "birth_date": date(2000, 1, 1), "position": "DEF",
         "detailed_position": "Right-Back"},
    ]))  # fmt: skip
    upsert(con, "player_source_ids", pd.DataFrame([
        {"player_id": "tm-1", "source": "fpl", "source_player_id": "1"},
        {"player_id": "tm-1", "source": "understat", "source_player_id": "2"},
        {"player_id": "tm-1", "source": "understat", "source_player_id": "3"},  # second club
    ]))  # fmt: skip
    upsert(con, "player_season_stats", pd.DataFrame([
        _row("fpl", "1", "tackles", 20, 900),
        _row("understat", "2", "xa", 1.0, 450, team="Old"),
        _row("understat", "3", "xa", 2.0, 450, team="New"),
    ]))  # fmt: skip
    row = per90_table(con, 2026).iloc[0]
    assert row["group"] == "FB"
    assert row["tackles"] == pytest.approx(2.0)  # 20 / 900 * 90
    assert row["xa"] == pytest.approx(0.3)  # (1 + 2) / (450 + 450) * 90
    assert row["minutes"] == 900


def test_tackles_prefer_apifootball_and_fall_back_to_fpl():
    con = connect(":memory:")
    upsert(con, "players", pd.DataFrame([
        {"player_id": "tm-1", "name": "A", "position": "DEF", "detailed_position": "Centre-Back"},
        {"player_id": "tm-2", "name": "B", "position": "DEF", "detailed_position": "Centre-Back"},
    ]))  # fmt: skip
    upsert(con, "player_source_ids", pd.DataFrame([
        {"player_id": "tm-1", "source": "fpl", "source_player_id": "1"},
        {"player_id": "tm-1", "source": "apifootball", "source_player_id": "9"},
        {"player_id": "tm-2", "source": "fpl", "source_player_id": "2"},
    ]))  # fmt: skip
    upsert(con, "player_season_stats", pd.DataFrame([
        _row("fpl", "1", "tackles", 10, 900),
        _row("apifootball", "9", "tackles", 30, 900),
        _row("fpl", "2", "tackles", 20, 900),
    ]))  # fmt: skip
    t = per90_table(con, 2026).set_index("player_id").tackles
    assert t["tm-1"] == pytest.approx(3.0)  # API-Football wins
    assert t["tm-2"] == pytest.approx(2.0)  # FPL fills the gap


def test_older_apifootball_season_fills_only_gaps_and_is_labelled():
    con = connect(":memory:")
    upsert(con, "players", pd.DataFrame([
        {"player_id": "tm-1", "name": "PL", "position": "DEF", "detailed_position": "Centre-Back"},
        {"player_id": "tm-2", "name": "L", "position": "DEF", "detailed_position": "Centre-Back"},
    ]))  # fmt: skip
    upsert(con, "player_source_ids", pd.DataFrame([
        {"player_id": "tm-1", "source": "fpl", "source_player_id": "1"},
        {"player_id": "tm-1", "source": "apifootball", "source_player_id": "8"},
        {"player_id": "tm-2", "source": "understat", "source_player_id": "2"},
        {"player_id": "tm-2", "source": "apifootball", "source_player_id": "9"},
    ]))  # fmt: skip
    old = {"season": 2024}
    upsert(con, "player_season_stats", pd.DataFrame([
        _row("fpl", "1", "tackles", 20, 900),                      # current FPL tackles
        {**_row("apifootball", "8", "tackles", 90, 2700), **old},  # must not override them
        {**_row("apifootball", "8", "interceptions", 30, 2700), **old},
        _row("understat", "2", "xa", 1.0, 900),
        {**_row("apifootball", "9", "tackles", 60, 1800), **old},
        {**_row("apifootball", "9", "interceptions", 5, 300), **old},  # too few minutes
    ]))  # fmt: skip
    t = per90_table(con, [2025, 2026]).set_index("player_id")
    assert t.loc["tm-1", "tackles"] == pytest.approx(2.0)  # window data wins
    assert t.loc["tm-1", "interceptions"] == pytest.approx(1.0)  # gap filled from 2024
    assert t.loc["tm-1", "defence_season"] == 2024
    assert t.loc["tm-2", "tackles"] == pytest.approx(3.0)
    assert pd.isna(t.loc["tm-2", "interceptions"])  # 300 minutes: not trusted


def test_multiple_seasons_pool_totals_and_minutes():
    con = connect(":memory:")
    upsert(con, "players", pd.DataFrame([
        {"player_id": "tm-1", "name": "A", "position": "FWD",
         "detailed_position": "Centre-Forward"},
    ]))  # fmt: skip
    upsert(con, "player_source_ids", pd.DataFrame([
        {"player_id": "tm-1", "source": "understat", "source_player_id": "2"},
    ]))  # fmt: skip
    old = {**_row("understat", "2", "npxg", 9.0, 2700), "season": 2025}
    new = _row("understat", "2", "npxg", 1.0, 900)
    upsert(con, "player_season_stats", pd.DataFrame([old, new]))
    row = per90_table(con, [2025, 2026]).iloc[0]
    assert row["minutes"] == 3600
    assert row["npxg"] == pytest.approx(0.25)  # (9 + 1) / 3600 * 90


def test_goals_prevented_is_xg_conceded_minus_goals():
    con = connect(":memory:")
    upsert(con, "players", pd.DataFrame([
        {"player_id": "tm-1", "name": "K", "position": "GK", "detailed_position": "Goalkeeper"},
    ]))  # fmt: skip
    upsert(con, "player_source_ids", pd.DataFrame([
        {"player_id": "tm-1", "source": "fpl", "source_player_id": "1"},
    ]))  # fmt: skip
    upsert(con, "player_season_stats", pd.DataFrame([
        _row("fpl", "1", "xg_conceded", 12.0, 900),
        _row("fpl", "1", "goals_conceded", 9.0, 900),
    ]))  # fmt: skip
    row = per90_table(con, 2026).iloc[0]
    assert row["goals_prevented"] == pytest.approx(0.3)  # (12 - 9) / 900 * 90
