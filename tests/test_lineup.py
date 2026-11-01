import pandas as pd
import pytest

from recruit_intel.analysis.lineup import Candidate, best_xi, depth_chart, shrink
from recruit_intel.config import Role, Slot

ROLES = {
    "fwd": Role(key="fwd", label="F", groups=["ST"], weights={"npxg": 1}),
    "wide": Role(key="wide", label="W", groups=["W"], weights={"xa": 1}),
}
SLOTS = [
    Slot(key="ST", groups=["ST"], role="fwd", x=50, y=85),
    Slot(key="RW", groups=["W", "ST"], role="wide", x=85, y=70),
]


def _squad(*rows):
    return pd.DataFrame(
        [{"player_id": pid, "name": pid, "group": g, "minutes": m, "npxg": npxg, "xa": xa}
         for pid, g, m, npxg, xa in rows]
    )  # fmt: skip


def test_shrink_pulls_small_samples_toward_average():
    assert shrink(90, 900, 270) == 90
    assert shrink(90, 135, 270) == 70


def test_best_xi_maximises_total_not_greedy():
    # Greedy would give ST to 'a' (best striker) and leave RW to weak 'w'.
    # Optimal: 'a' on the wing (he can play there), 'b' up front.
    cands = {
        "ST": [Candidate("a", "a", 90, 90, 900), Candidate("b", "b", 85, 85, 900)],
        "RW": [Candidate("a", "a", 80, 80, 900), Candidate("w", "w", 30, 30, 900)],
    }
    xi = best_xi(cands)
    assert (xi["ST"].player_id, xi["RW"].player_id) == ("b", "a")


def test_depth_chart_lists_backups_excluding_starters():
    squad = _squad(
        ("st1", "ST", 900, 90, None),
        ("st2", "ST", 900, 60, None),
        ("w1", "W", 900, None, 75),
        ("cb", "CB", 900, 99, 99),  # not eligible anywhere
    )
    chart = {d.slot.key: d for d in depth_chart(squad, SLOTS, ROLES, 270)}
    assert chart["ST"].starter.player_id == "st1"
    assert [c.player_id for c in chart["ST"].backups] == ["st2"]
    assert chart["RW"].starter.player_id == "w1"


def test_slot_with_no_eligible_player_has_no_starter():
    chart = depth_chart(_squad(("st1", "ST", 900, 90, None)), SLOTS[:1] + [
        Slot(key="GK", groups=["GK"], role="fwd", x=50, y=5)], ROLES, 270)  # fmt: skip
    assert chart[1].starter is None


@pytest.mark.parametrize("minutes,expected", [(0, 50), (270, 90)])
def test_shrink_bounds(minutes, expected):
    assert shrink(90, minutes, 270) == expected
