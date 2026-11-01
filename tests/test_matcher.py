from datetime import date

import pandas as pd

from recruit_intel.identity.matcher import build_identity, name_score

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


def test_unmatched_player_keeps_source_id_and_canonical_row():
    out = build_identity(TM, FPL, US)
    assert ids_by_source(out)[("understat", "104")] == "us-104"
    players = out["players"].set_index("player_id")
    assert players.loc["tm-1", "name"] == "Bruno Fernandes"  # Transfermarkt name preferred
    assert players.loc["us-104", "name"] == "Someone Unknown"
    assert out["player_source_ids"].groupby(["source", "source_player_id"]).size().max() == 1
