from datetime import date

import pandas as pd

from recruit_intel.db import connect, upsert
from recruit_intel.quality import audit, fix


def _db():
    con = connect(":memory:")
    upsert(con, "source_players", pd.DataFrame([
        {"source": "fpl", "source_player_id": "1", "name": "Real", "birth_date": date(2000, 1, 1)},
        {"source": "fpl", "source_player_id": "2", "name": "Baby", "birth_date": date(2024, 1, 1)},
    ]))  # fmt: skip
    upsert(con, "players", pd.DataFrame([
        {"player_id": "fpl-1", "name": "Real"},
        {"player_id": "ghost", "name": "Nobody links here"},
    ]))  # fmt: skip
    upsert(con, "player_source_ids", pd.DataFrame([
        {"player_id": "fpl-1", "source": "fpl", "source_player_id": "1"},
        {"player_id": "fpl-9", "source": "fpl", "source_player_id": "9"},  # no source row
    ]))  # fmt: skip
    row = {"source": "fpl", "source_player_id": "1", "season": 2026,
           "competition_id": "ENG-Premier League", "minutes": 900}  # fmt: skip
    upsert(con, "player_season_stats", pd.DataFrame([
        {**row, "metric": "tackles", "value": 20},
        {**row, "metric": "saves", "value": -3},
        {**row, "metric": "cbi", "minutes": 9000, "value": 5},
        {**row, "source_player_id": "7", "metric": "tackles", "value": 1},  # no player
    ]))  # fmt: skip
    upsert(con, "market", pd.DataFrame([
        {"source": "transfermarkt", "source_player_id": "1", "value_eur": 0,
         "snapshot_date": date(2026, 11, 1)},
    ]))  # fmt: skip
    return con


def counts(con):
    return dict(zip(*audit(con)[["check", "rows"]].T.values, strict=True))


def test_audit_finds_each_planted_problem():
    c = counts(_db())
    assert c["orphan source ids"] == 1
    assert c["unlinked canonical players"] == 1
    assert c["stats without a player"] == 1
    assert c["negative stat values"] == 1
    assert c["impossible minutes"] == 1
    assert c["placeholder market values"] == 1
    assert c["impossible birth dates"] == 1


def test_fix_repairs_only_the_fixable_and_keeps_good_rows():
    con = _db()
    fix(con)
    c = counts(con)
    fixable = audit(con).query("fixable").set_index("check").rows
    assert (fixable == 0).all(), fixable[fixable > 0]
    assert c["team strength ratings"] == 1  # report-only check is untouched
    kept = con.execute("SELECT metric, value FROM player_season_stats").fetchall()
    assert kept == [("tackles", 20.0)]
    assert con.execute("SELECT value_eur FROM market").fetchone() == (None,)
