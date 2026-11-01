import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from recruit_intel.models.managers import profiles, signature, similar_managers  # noqa: E402
from recruit_intel.models.team_style import STYLE_FEATURES  # noqa: E402


def _stint(manager_id, team_id, season, matches, **z):
    vec = {f: 0.0 for f in STYLE_FEATURES} | z
    return {"manager_id": manager_id, "manager": f"M{manager_id}", "team_id": team_id,
            "season": season, "matches": matches, **vec}  # fmt: skip


STINTS = pd.DataFrame(
    [
        _stint("1", "a", 2023, 38, ppda=-1.2),  # presses hard at two clubs
        _stint("1", "b", 2024, 19, ppda=-0.9),
        _stint("2", "c", 2024, 38, buildup_share=1.5),
        _stint("3", "d", 2024, 38, ppda=-1.0),
    ]
)


def test_profile_is_match_weighted_mean():
    prof = profiles(STINTS).set_index("manager_id").loc["1"]
    assert prof.matches == 57 and prof.clubs == 2
    assert prof.ppda == pytest.approx((-1.2 * 38 - 0.9 * 19) / 57)


def test_signature_names_the_consistent_trait():
    prof = profiles(STINTS).set_index("manager_id").loc["1"]
    assert signature(prof) == "presses more than its level"


def test_similar_managers_by_style():
    near = similar_managers(profiles(STINTS), "1", k=2)
    assert list(near.manager_id) == ["3", "2"]
    assert np.isfinite(near.distance).all()
