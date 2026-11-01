import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from recruit_intel.models.team_style import STYLE_FEATURES, fit, nearest_teams  # noqa: E402


def _seasons(n=300, seed=2):
    """Strength drives territory and npxG; a third of teams press extra hard
    for their level."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        q = rng.uniform(0.8, 2.2)  # expected points per match
        presser = i % 3 == 0
        rows.append({
            "team_id": str(i), "team_name": f"T{i}", "competition_id": "X", "season": 2025,
            "matches": 38, "xpts_pm": q,
            "ppda": 14 - 2 * q - (4 if presser else 0) + rng.normal(0, 0.5),
            "oppda": 8 + 2 * q + rng.normal(0, 0.8),
            "territory": 0.2 + 0.2 * q + rng.normal(0, 0.02),
            "xg_per_shot": 0.08 + 0.015 * q + rng.normal(0, 0.005),
            "buildup_share": 0.55 + 0.03 * q + rng.normal(0, 0.03),
            "npxg": 0.4 + 0.6 * q + rng.normal(0, 0.1),
            "npxga": 2.2 - 0.6 * q + rng.normal(0, 0.1),
        })  # fmt: skip
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def fitted():
    df = _seasons()
    model, metrics = fit(df, epochs=150)
    return df, model, metrics


def test_strength_is_removed_before_styling(fitted):
    df, model, metrics = fitted
    assert metrics["strength_r2"]["territory"] > 0.8
    adjusted = model.adjusted(df)
    territory = adjusted[:, STYLE_FEATURES.index("territory")]
    assert abs(np.corrcoef(territory, df.xpts_pm)[0, 1]) < 0.05


def test_pressing_trait_found_independent_of_strength(fitted):
    df, model, _ = fitted
    traits = model.traits(df)
    pressers = [t for i, t in enumerate(traits) if i % 3 == 0]
    share = np.mean(["presses more than its level" in t for t in pressers])
    assert share > 0.8


def test_nearest_teams_share_the_style(fitted):
    df, model, _ = fitted
    near = nearest_teams(df, model, row=0, k=10)  # team 0 is a presser
    assert (near.team_id.astype(int) % 3 == 0).mean() >= 0.8
