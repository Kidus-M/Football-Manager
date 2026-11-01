from datetime import date

import pandas as pd

from recruit_intel.analysis.squad import current_squad
from recruit_intel.config import load_club
from recruit_intel.db import connect, upsert
from recruit_intel.identity.run import rebuild

EPL = "ENG-Premier League"


def _stats(source, pid, minutes, competition, values):
    return [
        {
            "source": source,
            "source_player_id": pid,
            "season": 2026,
            "competition_id": competition,
            "minutes": minutes,
            "metric": metric,
            "value": value,
        }
        for metric, value in values.items()
    ]


def test_squad_joins_sources_through_canonical_ids():
    con = connect(":memory:")
    upsert(con, "teams", pd.DataFrame([
        {"source": "transfermarkt", "source_team_id": "985", "name": "Manchester United",
         "competition_id": EPL, "season": 2026},
    ]))  # fmt: skip
    upsert(con, "source_players", pd.DataFrame([
        {"source": "fpl", "source_player_id": "10", "name": "Bruno Borges Fernandes",
         "team_name": "Man Utd", "position": "MID", "birth_date": date(1994, 9, 8)},
        {"source": "transfermarkt", "source_player_id": "240306", "name": "Bruno Fernandes",
         "team_name": "Manchester United", "team_id": "985", "position": "MID",
         "birth_date": date(1994, 9, 8), "detailed_position": "Attacking Midfield"},
        {"source": "understat", "source_player_id": "1228", "name": "Bruno Fernandes",
         "team_name": "Manchester United", "position": "MID"},
    ]))  # fmt: skip
    upsert(con, "market", pd.DataFrame([
        {"source": "transfermarkt", "source_player_id": "240306", "value_eur": 35_000_000,
         "contract_end": date(2027, 6, 30), "snapshot_date": date(2026, 11, 1)},
    ]))  # fmt: skip
    upsert(con, "player_season_stats", pd.DataFrame(
        _stats("fpl", "10", 900, EPL, {"tackles": 10.0, "cbi": 5.0, "recoveries": 40.0,
                                       "saves": 0.0})
        + _stats("understat", "1228", 900, EPL, {"xg": 3.0, "npxg": 3.0, "xa": 2.0})
    ))  # fmt: skip

    rebuild(con)
    row = current_squad(con, load_club("man_utd"), 2026).iloc[0]

    assert row["name"] == "Bruno Fernandes"  # canonical (Transfermarkt) name
    assert row["detailed_position"] == "Attacking Midfield"
    assert (row["value_m_eur"], row["contract"], row["minutes"]) == (35.0, 2027, 900)
    assert row["npxg"] == 0.3 and row["tackles"] == 1.0
