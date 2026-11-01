from datetime import date
from pathlib import Path

import pytest

from recruit_intel.db import connect, upsert
from recruit_intel.ingest import transfermarkt as tm

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("€35M", 35_000_000),
        ("€8.0M", 8_000_000),
        ("€200K", 200_000),
        ("€1.20bn", 1_200_000_000),
        ("-", None),
    ],
)
def test_parse_value(text, expected):
    assert tm.parse_value(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [("07/07/2002 (24)", date(2002, 7, 7)), ("Jul 7, 2002 (24)", date(2002, 7, 7)), ("-", None)],
)
def test_parse_date(text, expected):
    assert tm.parse_date(text) == expected


def test_parse_squad_fixture():
    html = (FIXTURES / "tm_squad.html").read_text(encoding="utf-8")
    out = tm.parse_squad(html, date(2026, 11, 1))
    p = out["source_players"].iloc[0].to_dict()
    assert p == {
        "source": "transfermarkt",
        "source_player_id": "503883",
        "name": "Senne Lammens",
        "team_name": "Manchester United",
        "position": "GK",
        "detailed_position": "Goalkeeper",
        "birth_date": date(2002, 7, 7),
        "nationality": "Belgium",
        "height_cm": 193,
        "foot": "right",
    }
    m = out["market"].iloc[0]
    assert m.value_eur == 35_000_000
    assert (m.contract_end, m.joined_date) == (date(2030, 6, 30), date(2025, 9, 1))
    assert out["market"].value_eur.isna().sum() == 1  # Dermot Mee has no value yet ('-')


def test_parse_league_fixture():
    html = (FIXTURES / "tm_league.html").read_text(encoding="utf-8")
    clubs = tm.parse_league(html, "ENG-Premier League", 2026)
    assert list(clubs.source_team_id) == ["281", "11", "631"]
    assert clubs.name.iloc[0] == "Manchester City"


def test_squad_rows_fit_schema():
    html = (FIXTURES / "tm_squad.html").read_text(encoding="utf-8")
    con = connect(":memory:")
    for table, df in tm.parse_squad(html, date(2026, 11, 1)).items():
        assert upsert(con, table, df) == 4


def test_parse_managers_fixture():
    html = (FIXTURES / "tm_managers.html").read_text(encoding="utf-8")
    spells = tm.parse_managers(html, 985, today=date(2026, 11, 1))
    current = spells.iloc[0]
    assert (current["name"], current.manager_id) == ("Michael Carrick", "65647")
    assert current.appointed == date(2026, 5, 22) and current.ended is None  # contract end hidden
    interim = spells.iloc[1]
    assert (interim.appointed, interim.ended, interim.matches) == (
        date(2026, 1, 13),
        date(2026, 5, 22),
        16,
    )
    assert upsert(connect(":memory:"), "manager_spells", spells) == len(spells)


def test_block_page_raises():
    with pytest.raises(ValueError, match="table.items"):
        tm.parse_squad("<html><h1>Just a moment...</h1></html>", date(2026, 11, 1))
