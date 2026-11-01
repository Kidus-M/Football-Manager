import pandas as pd

from recruit_intel.db import connect, upsert


def test_upsert_replaces_on_primary_key():
    con = connect(":memory:")
    row = {"source": "fpl", "source_player_id": "1", "name": "A", "position": "MID"}
    upsert(con, "source_players", pd.DataFrame([row]))
    upsert(con, "source_players", pd.DataFrame([{**row, "position": "FWD"}]))
    assert con.execute("SELECT position FROM source_players").fetchall() == [("FWD",)]
