import pandas as pd
import pytest

from recruit_intel.features.percentiles import min_minutes_for, percentiles

EPL = "ENG-Premier League"


def _table(rows):
    return pd.DataFrame(
        [
            {"player_id": pid, "name": pid, "group": group, "detailed_position": None,
             "competition_id": comp, "minutes": minutes, "npxg": npxg, "goals_conceded": gc}
            for pid, group, comp, minutes, npxg, gc in rows
        ]
    )  # fmt: skip


TABLE = _table(
    [
        ("a", "ST", EPL, 900, 0.1, 1.0),
        ("b", "ST", EPL, 900, 0.2, 2.0),
        ("c", "ST", EPL, 900, 0.3, 3.0),
        ("d", "ST", EPL, 900, 0.4, 4.0),
        ("cameo", "ST", EPL, 50, 0.9, 0.0),  # too few minutes to be a peer
        ("cb", "CB", EPL, 900, 0.9, 1.0),  # other group: not a peer
        ("other_league", "ST", "ESP-La Liga", 900, 5.0, 1.0),
    ]
)


def test_ranked_within_competition_and_group_only():
    pct = percentiles(TABLE, EPL, min_minutes=270).set_index("player_id")
    assert pct.loc["a", "npxg"] == pytest.approx(12.5)  # beats none, ties itself
    assert pct.loc["d", "npxg"] == pytest.approx(87.5)
    assert pct.loc["d", "peers"] == 4
    assert "other_league" not in pct.index


def test_low_minutes_player_is_ranked_but_not_qualified():
    pct = percentiles(TABLE, EPL, min_minutes=270).set_index("player_id")
    assert not pct.loc["cameo", "qualified"]
    assert pct.loc["cameo", "npxg"] == 100  # beats every qualified peer


def test_lower_is_better_metric_is_flipped():
    pct = percentiles(TABLE, EPL, min_minutes=270).set_index("player_id")
    assert pct.loc["a", "goals_conceded"] > pct.loc["d", "goals_conceded"]


def test_min_minutes_scales_with_season():
    early = _table([("x", "ST", EPL, 540, 0, 0)])
    late = _table([("x", "ST", EPL, 3000, 0, 0)])
    assert min_minutes_for(early, EPL) == 270
    assert min_minutes_for(late, EPL) == 1000
