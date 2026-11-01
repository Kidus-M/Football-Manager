import pandas as pd
import pytest

from recruit_intel.config import Role
from recruit_intel.search.rank import (
    Filters,
    age_score,
    apply_filters,
    contract_score,
    fit_scores,
    rank,
    value_scores,
)

EPL, L1 = "ENG-Premier League", "FRA-Ligue 1"
ROLE = Role(key="st", label="ST", groups=["ST"], weights={"npxg": 1})
STRENGTH = {EPL: 1.0, L1: 0.8}


def _pool(*rows):
    cols = ["player_id", "name", "group", "competition_id", "minutes", "age", "value_eur",
            "contract_year", "npxg"]  # fmt: skip
    return pd.DataFrame(rows, columns=cols)


POOL = _pool(
    ("epl", "EPL striker", "ST", EPL, 3000, 25, 50e6, 2029, 90),
    ("l1", "Ligue 1 striker", "ST", L1, 3000, 25, 50e6, 2029, 90),
    ("old", "Old striker", "ST", EPL, 3000, 33, 5e6, 2027, 95),
    ("cameo", "Cameo", "ST", EPL, 300, 22, 5e6, 2030, 99),
    ("mine", "Own player", "ST", EPL, 3000, 24, 60e6, 2030, 80),
    ("cb", "Centre-back", "CB", EPL, 3000, 24, 10e6, 2030, 99),
)


def test_filters():
    f = Filters(groups=["ST"], max_age=30, min_minutes=900, exclude={"mine"})
    assert set(apply_filters(POOL, f).player_id) == {"epl", "l1"}
    f = Filters(groups=["ST"], contract_until=2027, min_minutes=0)
    assert set(apply_filters(POOL, f).player_id) == {"old"}


def test_league_strength_discounts_identical_output():
    ranked = rank(POOL, ROLE, Filters(groups=["ST"], max_age=30), STRENGTH, 2027)
    fits = ranked.set_index("player_id").fit
    assert fits["epl"] > fits["l1"]
    assert list(ranked.player_id[:2]) == ["epl", "l1"]


def test_partial_role_data_is_discounted():
    role = Role(key="fb", label="FB", groups=["FB"], weights={"xa": 0.8, "tackles": 0.2})
    pool = pd.DataFrame(
        [
            {"player_id": "full", "group": "FB", "competition_id": EPL, "minutes": 3000,
             "xa": 90, "tackles": 90},
            {"player_id": "partial", "group": "FB", "competition_id": EPL, "minutes": 3000,
             "xa": 90, "tackles": None},
        ]
    )  # fmt: skip
    fits = fit_scores(pool, role, ["FB"], STRENGTH).set_index("player_id").fit
    assert fits["full"] == pytest.approx(90)
    assert fits["partial"] == pytest.approx(82)  # 50 + 40 * 0.8 coverage


def test_why_explains_the_rank():
    ranked = rank(POOL, ROLE, Filters(groups=["ST"], max_age=30), STRENGTH, 2027)
    why = ranked.set_index("player_id").why["l1"]
    assert "Ligue 1" in why and "age 25" in why and "contract 2029" in why


def test_system_fit_breaks_a_tie_and_shows_in_why():
    pool = _pool(
        ("a", "A", "ST", EPL, 3000, 25, 50e6, 2029, 90),
        ("b", "B", "ST", EPL, 3000, 25, 50e6, 2029, 90),
    )
    fits = pd.Series({"a": -10.0, "b": 12.0})
    ranked = rank(pool, ROLE, Filters(groups=["ST"]), STRENGTH, 2027, system_fit=fits)
    assert list(ranked.player_id) == ["b", "a"]
    assert "style fit +12" in ranked.why.iloc[0]


@pytest.mark.parametrize("age,expected", [(24, 100), (29, 50), (34, 10), (18, 75)])
def test_age_score(age, expected):
    assert age_score(age) == expected


def test_contract_score_rewards_leverage():
    assert contract_score(2027, 2027) == 100  # free next summer
    assert contract_score(2028, 2027) > contract_score(2031, 2027)


def test_value_score_against_budget():
    scores = value_scores(pd.Series([10e6, 50e6, None]), budget=50_000_000)
    assert list(scores) == [80, 0, 50]
