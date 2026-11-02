from datetime import date

import pandas as pd
import pytest

from recruit_intel.identity.matcher import build_identity, name_score, surnames_agree

EPL, LIGA = "ENG-Premier League", "ESP-La Liga"

TM = pd.DataFrame(
    [
        ("1", "Bruno Fernandes", date(1994, 9, 8), "Portugal", "MID", "Manchester United", EPL),
        ("2", "Matheus Cunha", date(1999, 5, 27), "Brazil", "FWD", "Manchester United", EPL),
        ("3", "Bukayo Saka", date(2001, 9, 5), "England", "FWD", "Arsenal FC", EPL),
        ("4", "Kylian Mbappé", date(1998, 12, 20), "France", "FWD", "Real Madrid", LIGA),
    ],
    columns=[
        "source_player_id", "name", "birth_date", "nationality", "position", "team_name",
        "competition_id",
    ],
)  # fmt: skip
FPL = pd.DataFrame(
    [
        ("10", "Bruno Borges Fernandes", date(1994, 9, 8), "MID", "Man Utd"),
        ("11", "Matheus Santos Carneiro da Cunha", date(1999, 5, 27), "MID", "Man Utd"),
        ("12", "Chido Obi", date(2007, 11, 29), "FWD", "Man Utd"),
    ],
    columns=["source_player_id", "name", "birth_date", "position", "team_name"],
)
US = pd.DataFrame(
    [
        ("100", "Bruno Fernandes", "MID", "Manchester United", EPL),
        ("101", "Bukayo Saka", "FWD", "Arsenal", EPL),
        ("102", "Kylian Mbappe-Lottin", "FWD", "Real Madrid", LIGA),
        ("103", "Chido Obi", "FWD", "Manchester United", EPL),
        ("104", "Someone Unknown", "MID", "Arsenal", EPL),
    ],
    columns=["source_player_id", "name", "position", "team_name", "competition_id"],
)


def ids_by_source(out):
    ids = out["player_source_ids"]
    return {(r.source, r.source_player_id): r.player_id for r in ids.itertuples()}


def test_exact_name_beats_subset_name():
    exact = name_score("Bruno Fernandes", "Bruno Fernandes")
    assert exact > name_score("Bruno Fernandes", "Bruno Borges Fernandes")


def test_all_sources_link_to_one_canonical_player():
    m = ids_by_source(build_identity(TM, FPL, US))
    assert m[("fpl", "10")] == m[("understat", "100")] == "tm-1"
    assert m[("fpl", "11")] == "tm-2"
    assert m[("understat", "101")] == "tm-3"  # 'Arsenal' -> 'Arsenal FC'
    assert m[("understat", "102")] == "tm-4"  # accents + extra surname


def test_players_missing_from_transfermarkt_link_to_each_other():
    m = ids_by_source(build_identity(TM, FPL, US))
    assert m[("fpl", "12")] == m[("understat", "103")] == "fpl-12"


def _squad(*names):
    return pd.DataFrame(
        [(str(i), n, None, None, "DEF", "VfB Stuttgart", "GER-Bundesliga")
         for i, n in enumerate(names)],
        columns=TM.columns,
    )  # fmt: skip


def _understat(name):
    return pd.DataFrame(
        [("900", name, "DEF", "VfB Stuttgart", "GER-Bundesliga")], columns=US.columns
    )


def test_unique_surname_in_club_matches_different_first_name():
    out = build_identity(
        _squad("Jeff Chabot", "Atakan Karazor"), FPL.iloc[:0], _understat("Julian Chabot")
    )
    m = ids_by_source(out)
    assert m[("understat", "900")] == "tm-0"


def test_shared_surname_in_club_does_not_match():
    out = build_identity(
        _squad("Jeff Chabot", "Tom Chabot"), FPL.iloc[:0], _understat("Julian Chabot")
    )
    assert ids_by_source(out)[("understat", "900")] == "us-900"


@pytest.mark.parametrize(
    "a,b,allow_single,expected",
    [
        ("Alberto Moreno", "Alberto Dossena", True, False),  # first name only
        ("Diego Morcillo", "Diego Murillo", False, False),  # similar but different surname
        ("Fernando Niño", "Fer Niño", False, True),
        ("Ísak Bergmann Jóhannesson", "Ísak Jóhannesson", False, True),
        ("Kepa", "Kepa Arrizabalaga", True, True),
        ("Aarón", "Aarón Ochoa", False, False),  # one word: only trusted within a club
    ],
)
def test_surnames_agree(a, b, allow_single, expected):
    assert surnames_agree(a, b, allow_single) is expected


def test_unmatched_player_keeps_source_id_and_canonical_row():
    out = build_identity(TM, FPL, US)
    assert ids_by_source(out)[("understat", "104")] == "us-104"
    players = out["players"].set_index("player_id")
    assert players.loc["tm-1", "name"] == "Bruno Fernandes"  # Transfermarkt name preferred
    assert players.loc["us-104", "name"] == "Someone Unknown"
    assert out["player_source_ids"].groupby(["source", "source_player_id"]).size().max() == 1


def test_apifootball_links_to_understat_through_club_and_season():
    af = pd.DataFrame(
        [("500", "R. Fraser", date(1994, 2, 24), "MID", "Southampton"),  # left the top 5
         ("501", "B. Saka", None, "FWD", "Arsenal"),  # no birth date: no TM link
         ("502", "J. Nobody", None, "MID", "Arsenal")],  # fmt: skip
        columns=["source_player_id", "name", "birth_date", "position", "team_name"],
    )
    us = pd.concat([US, pd.DataFrame([("105", "Ryan Fraser", "MID", "Southampton", EPL)],
                                     columns=US.columns)])  # fmt: skip
    seasons = {
        "af": pd.DataFrame(
            [("500", "R. Fraser", 2024, EPL, "Southampton"),
             ("501", "B. Saka", 2024, EPL, "Arsenal"),
             ("502", "J. Nobody", 2024, EPL, "Arsenal")],
            columns=["source_player_id", "name", "season", "competition_id", "team_name"],
        ),
        "us": pd.DataFrame(
            [("105", "Ryan Fraser", 2024, EPL, "Southampton"),
             ("101", "Bukayo Saka", 2024, EPL, "Arsenal")],
            columns=["source_player_id", "name", "season", "competition_id", "team_name"],
        ),
    }  # fmt: skip
    m = ids_by_source(build_identity(TM, FPL, us, af, seasons))
    assert m[("apifootball", "500")] == m[("understat", "105")] == "us-105"
    assert m[("apifootball", "501")] == "tm-3"  # through Understat's Transfermarkt link
    assert m[("apifootball", "502")] == "af-502"  # nobody at the club by that name
