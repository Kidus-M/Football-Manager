import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from recruit_intel.models.dataset import EMBED_FEATURES  # noqa: E402
from recruit_intel.models.team_fit import FitModel, train, validate  # noqa: E402
from recruit_intel.models.team_style import STYLE_FEATURES  # noqa: E402


def _pairs(n=3000, seed=4):
    """Destination pressing (low adjusted ppda) lifts output 25%; strength
    alone does nothing."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        base = rng.uniform(0.5, 1.5)
        press = rng.normal()
        lift = 1.25 if press < -0.5 else 1.0
        row = {"source_player_id": str(i), "season": 2018 + i % 6, "age": 25.0,
               "ctx_strength": rng.uniform(1, 2), "dest_strength": rng.uniform(1, 2),
               "moved": bool(i % 2)}  # fmt: skip
        for f, scale in zip(EMBED_FEATURES, [0.3, 0.2, 2.0, 1.5, 0.6, 0.3], strict=True):
            row[f] = base * scale
            row[f"{f}_next"] = base * scale * lift * rng.normal(1, 0.05)
        for f in STYLE_FEATURES:
            row[f"ctx_{f}"] = rng.normal()
            row[f"dest_{f}"] = press if f == "ppda" else rng.normal()
        rows.append(row)
    return pd.DataFrame(rows)


def test_style_model_beats_ablation_when_style_matters():
    report = validate(_pairs())
    allp = report["all"]
    assert allp["with style"] < allp["strength only"] < allp["no change"] * 1.05
    assert allp["with style"] < 0.6 * allp["no change"]


def test_fit_score_recovers_the_planted_lift():
    pairs = _pairs()
    model = train(pairs)
    sample = pairs.sample(200, random_state=1)
    press = model.fit_score(sample.assign(dest_ppda=-1.5)).mean()
    sit = model.fit_score(sample.assign(dest_ppda=1.5)).mean()
    assert press == pytest.approx(25, abs=7)  # planted: +25% at pressing teams
    assert sit == pytest.approx(0, abs=5)
    assert isinstance(model, FitModel)
