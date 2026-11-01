from datetime import date

import pytest

from recruit_intel.ingest import clubelo

CSV = """Rank,Club,Country,Level,Elo,From,To
1,Liverpool,ENG,1,2010.5,2026-10-26,2026-11-02
None,ManUnited,ENG,1,1850.2,2026-10-26,2026-11-02
"""


def test_parse_snapshot():
    df = clubelo.parse(CSV, date(2026, 11, 1))
    row = df.set_index("clubelo_name").loc["ManUnited"]
    assert row.elo == 1850.2 and row.level == 1
    assert (df.snapshot_date == date(2026, 11, 1)).all()


def test_empty_snapshot_raises():
    with pytest.raises(ValueError):
        clubelo.parse("Rank,Club,Country,Level,Elo,From,To\n", date(2026, 11, 1))
