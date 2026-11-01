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
