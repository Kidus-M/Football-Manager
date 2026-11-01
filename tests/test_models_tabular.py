import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from recruit_intel.models.dataset import FEATURES  # noqa: E402
from recruit_intel.models.projection import make_pairs, train_and_validate  # noqa: E402
from recruit_intel.models.value import bargains, value_model  # noqa: E402

STRONG, WEAK = "ENG-Premier League", "FRA-Ligue 1"


def _seasons(n_players=600, seed=1):
    """Players with a stable talent level; moving into STRONG cuts output 20%."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_players):
        talent = rng.uniform(0.5, 1.5)
        comp = WEAK if p % 2 else STRONG
        for season in range(2019, 2026):
            if season > 2019 and rng.random() < 0.15:
                comp = STRONG if comp == WEAK else WEAK  # a league move
            level = talent * (0.8 if comp == STRONG else 1.0)
            vals = {
                f: level * base * rng.normal(1, 0.08)
                for f, base in zip(
                    FEATURES, [0.3, 0.2, 2.0, 1.5, 0.6, 0.3, 0.25, 0.15], strict=True
                )
            }
            rows.append({"player_id": f"p{p}", "name": f"p{p}", "group": "ST",
                         "competition_id": comp, "season": season, "minutes": 2000,
                         "age": 24 + season - 2019, **vals})  # fmt: skip
    return pd.DataFrame(rows)


def test_make_pairs_links_consecutive_seasons_only():
    s = _seasons(5)
    pairs = make_pairs(s, last_complete=2024)
    assert (pairs.season < 2024).all()
    assert {"from_comp", "to_comp", "npxg_next"} <= set(pairs.columns)


def test_projection_beats_persistence_and_learns_league_gap():
    _, report = train_and_validate(_seasons())
    m = report["metrics"]["npxg"]
    assert m["r2_model"] > m["r2_same_as_last_year"]
    # Output kept moving from the weak league to the strong one: about 0.8.
    assert report["league_factors"][WEAK] == pytest.approx(0.8, abs=0.08)
    assert report["league_factors"][STRONG] == pytest.approx(1.0, abs=0.05)


def test_value_model_flags_planted_bargain():
    rng = np.random.default_rng(3)
    rows = []
    for i in range(400):
        out = rng.uniform(0.5, 1.5)
        vals = {f: out * 0.3 for f in FEATURES}
        value = 20e6 * out * rng.normal(1, 0.1)
        rows.append({"player_id": f"p{i}", "name": f"p{i}", "group": "ST", "age": 25,
                     "contract_year": 2029, "minutes": 2500, "competition_id": STRONG,
                     "value_eur": value, **vals})  # fmt: skip
    rows[0].update(value_eur=rows[0]["value_eur"] * 0.25)  # priced at a quarter of peers
    valued, metrics = value_model(pd.DataFrame(rows), window_year=2027)
    assert metrics["r2_log_value"] > 0.5
    assert bargains(valued, limit=3).player_id.iloc[0] == "p0"
