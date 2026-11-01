import pandas as pd
import pytest

from recruit_intel.analysis.profile import best_role, role_fit, strengths_weaknesses
from recruit_intel.config import Role, load_roles


def _row(**pcts):
    return pd.Series({"group": "ST", "position": "FWD", **pcts})


def test_strengths_and_weaknesses_use_group_metrics_only():
    row = _row(npxg=95, shots=85, xa=10, key_passes=50, tackles=1)  # tackles: not a ST metric
    strengths, weaknesses = strengths_weaknesses(row)
    assert strengths == ["npxg", "shots"]
    assert weaknesses == ["xa"]


def test_role_fit_is_weighted_mean_over_available_metrics():
    role = Role(key="r", label="R", positions=["FWD"], weights={"npxg": 3, "xa": 1})
    assert role_fit(_row(npxg=80, xa=40), role).score == pytest.approx(70)


def test_role_fit_needs_position_and_coverage():
    role = Role(key="r", label="R", positions=["FWD"], weights={"npxg": 1, "tackles": 3})
    assert role_fit(_row(npxg=90), role) is None  # only 25% of the weight has data
    assert role_fit(_row(npxg=90, tackles=50, position="DEF"), role) is None


def test_best_role_picks_highest_fit():
    fit = best_role(_row(npxg=95, shots=90, npg=90, xa=20, xg_chain=60), load_roles())
    assert fit.role == "poacher"


def test_negative_role_weight_rejected():
    with pytest.raises(ValueError, match="positive"):
        Role(key="r", label="R", positions=["GK"], weights={"goals_conceded": -1})
