from datetime import date

import numpy as np
import pandas as pd
import pytest

from recruit_intel.analysis.profile import role_fit
from recruit_intel.config import Role
from recruit_intel.db import connect, upsert
from recruit_intel.features.context import possession_factor
from recruit_intel.features.per90 import per90_table
from recruit_intel.features.percentiles import percentiles
from recruit_intel.models import defence
from recruit_intel.search.rank import Filters, rank

EPL, ITA = "ENG-Premier League", "ITA-Serie A"


def test_possession_factor_rescales_to_half_and_caps():
    f = possession_factor(pd.Series([0.5, 0.6, 0.9, None]))
    assert list(f.round(3)) == [1.0, 1.25, round(0.5 / 0.3, 3), 1.0]


def _stat(source, pid, metric, value, minutes, season=2026, team="X", comp=EPL):
    return {"source": source, "source_player_id": pid, "season": season,
            "competition_id": comp, "team_name": team, "minutes": minutes,
            "metric": metric, "value": value}  # fmt: skip


def _match(team, day, ours, theirs, season=2026):
    return {"source": "understat", "team_id": team, "team_name": team, "season": season,
            "competition_id": EPL, "match_date": pd.Timestamp(f"{season}-09-{day:02d}"),
            "home": True, "xg": 1.0, "xga": 1.0, "npxg": 1.0, "npxga": 1.0,
            "ppda_passes": theirs, "ppda_actions": 20, "oppda_passes": ours,
            "oppda_actions": 20, "deep": 5, "deep_allowed": 5, "scored": 1,
            "conceded": 1, "xpts": 1.5, "pts": 1}  # fmt: skip


def test_defensive_counts_are_possession_adjusted():
    con = connect(":memory:")
    upsert(con, "players", pd.DataFrame([
        {"player_id": "tm-1", "name": "A", "position": "DEF", "detailed_position": "Centre-Back"},
    ]))  # fmt: skip
    upsert(con, "player_source_ids", pd.DataFrame([
        {"player_id": "tm-1", "source": "fpl", "source_player_id": "1"},
        {"player_id": "tm-1", "source": "understat", "source_player_id": "2"},
    ]))  # fmt: skip
    upsert(con, "player_season_stats", pd.DataFrame([
        _stat("fpl", "1", "tackles", 20, 900),
        _stat("understat", "2", "xa", 1.0, 900, team="City"),
    ]))  # fmt: skip
    # City make 60% of the passes in the PPDA zone: 0.5 / (1 - 0.6) = x1.25.
    upsert(con, "team_matches", pd.DataFrame([_match("City", 1, 300, 200),
                                              _match("City", 8, 300, 200)]))  # fmt: skip
    row = per90_table(con, 2026).iloc[0]
    assert row["possession"] == pytest.approx(0.6)
    assert row["tackles"] == pytest.approx(2.0 * 1.25)
    assert row["xa"] == pytest.approx(0.1)  # attacking output isn't adjusted


def test_estimates_rank_against_measured_reference_with_a_range():
    table = pd.DataFrame([
        {"player_id": f"m{i}", "name": "", "group": "CB", "competition_id": ITA,
         "minutes": 900, "tackles": np.nan} for i in range(3)
    ] + [
        {"player_id": "est", "name": "", "group": "CB", "competition_id": ITA, "minutes": 900,
         "tackles": 2.0, "tackles_lo": 1.0, "tackles_hi": 3.0, "tackles_kind": "estimated"},
    ])  # fmt: skip
    # No measured peers in the window: rank against measured older seasons.
    reference = pd.DataFrame(
        {
            "competition_id": ITA,
            "group": "CB",
            "metric": "tackles",
            "value": np.linspace(0.1, 3.9, 39),
        }  # fmt: skip
    )
    pct = percentiles(table, ITA, min_minutes=0, reference=reference)
    row = pct.set_index("player_id").loc["est"]
    assert row["tackles"] == pytest.approx(50, abs=1.5)  # 2.0 is mid-distribution
    assert row["tackles_lo"] < row["tackles"] < row["tackles_hi"]
    assert row["tackles_kind"] == "estimated"


def test_lower_is_better_ranges_stay_ordered():
    table = pd.DataFrame([
        {"player_id": "est", "name": "", "group": "CB", "competition_id": ITA, "minutes": 900,
         "dribbled_past": 1.0, "dribbled_past_lo": 0.5, "dribbled_past_hi": 1.5,
         "dribbled_past_kind": "carried"},
    ])  # fmt: skip
    reference = pd.DataFrame(
        {
            "competition_id": ITA,
            "group": "CB",
            "metric": "dribbled_past",
            "value": np.linspace(0.1, 2.0, 30),
        }  # fmt: skip
    )
    row = percentiles(table, ITA, min_minutes=0, reference=reference).iloc[0]
    assert row["dribbled_past_lo"] < row["dribbled_past"] < row["dribbled_past_hi"]


ROLE = Role(key="cb", label="CB", groups=["CB"], weights={"xg_buildup": 0.5, "tackles": 0.5})


def _cb(pid, kind=None, **values):
    row = {"player_id": pid, "group": "CB", "competition_id": EPL, "minutes": 3000,
           "xg_buildup": 80, "tackles": 80, **values}  # fmt: skip
    if kind:
        row["tackles_kind"] = kind
    return row


def test_role_fit_range_confidence_and_own_data():
    measured = role_fit(pd.Series(_cb("m")), ROLE)
    assert (measured.lo, measured.hi, measured.confidence) == (80, 80, 1.0)
    assert measured.own_data

    est = role_fit(pd.Series(_cb("e", "estimated", tackles_lo=40, tackles_hi=95)), ROLE)
    assert est.score == 80
    assert (est.lo, est.hi) == (60, 87.5)
    reliable = defence.DEFAULT_RELIABILITY
    assert est.confidence == pytest.approx(0.5 + 0.5 * reliable["estimated"])
    assert not est.own_data

    carried = role_fit(pd.Series(_cb("c", "carried", tackles_lo=70, tackles_hi=90)), ROLE)
    assert carried.own_data
    assert est.confidence < carried.confidence < measured.confidence


def test_search_discounts_estimates_and_can_leave_them_out():
    pool = pd.DataFrame([
        _cb("measured"),
        _cb("estimated", "estimated", tackles_lo=40, tackles_hi=95),
    ]).assign(age=24, value_eur=1e6, contract_year=2028)  # fmt: skip
    ranked = rank(pool, ROLE, Filters(groups=["CB"]), {EPL: 1.0}, 2027).set_index("player_id")
    assert ranked.fit["measured"] > ranked.fit["estimated"]
    assert ranked.fit_lo["estimated"] < ranked.fit["estimated"] < ranked.fit_hi["estimated"]
    assert "defence estimated" in ranked.why["estimated"]
    only = rank(pool, ROLE, Filters(groups=["CB"], own_data_only=True), {EPL: 1.0}, 2027)
    assert list(only.player_id) == ["measured"]


def _synthetic_rows(n=400, seed=0):
    rng = np.random.default_rng(seed)
    rows = pd.DataFrame({
        "player_id": [f"p{i % (n // 2)}" for i in range(n)],
        "season": 2024, "minutes": 2000, "group": rng.choice(["CB", "CM"], n),
        "competition_id": rng.choice([EPL, ITA], n), "age": rng.uniform(19, 33, n),
        **{c: rng.uniform(0, 1, n) for c in defence.PROFILE},
        "possession": rng.uniform(0.35, 0.65, n), "ppda": 10.0, "xga_pm": 1.2,
        "deep_allowed_pm": 6.0,
    })  # fmt: skip
    skill = rng.normal(2, 0.6, n)
    rows["tackles"] = (skill + 0.1 * rng.normal(size=n)).clip(0)
    rows["lag_tackles"] = np.where(np.arange(n) % 2 == 0, skill, np.nan)
    rows["lag_gap"] = np.where(rows.lag_tackles.notna(), 2.0, np.nan)
    for m in defence.TARGETS:
        if m != "tackles":
            rows[m] = rows.xg_buildup + rng.normal(0, 0.1, n)
    return rows


@pytest.fixture
def two_targets(monkeypatch):
    """Fitting 3 quantile models for every target is slow; two show the logic."""
    monkeypatch.setattr(defence, "TARGETS", ["tackles", "interceptions"])


def test_model_uses_the_anchor_and_validates(two_targets):
    rows = _synthetic_rows()
    model = defence.DefenceModel.fit(rows)
    pred = model.predict(rows.iloc[:20])["tackles"]
    assert (pred.lo <= pred.mid).all() and (pred.mid <= pred.hi).all()
    report = defence.validate(rows, folds=2)
    by = {(r["scenario"], r["metric"]): r for r in report["validation"]}
    # An anchor that is the player's true level beats no anchor at all.
    assert by[("carried, anchor 2 seasons back", "tackles")]["model"] > 0.8
    assert report["reliability"]["carried"] > report["reliability"]["estimated"]


def test_fill_marks_carried_and_estimated_cells(two_targets):
    con = connect(":memory:")
    upsert(con, "players", pd.DataFrame([
        {"player_id": "tm-1", "name": "A", "position": "DEF", "detailed_position": "Centre-Back",
         "birth_date": date(1999, 1, 1)},
    ]))  # fmt: skip
    upsert(con, "player_source_ids", pd.DataFrame([
        {"player_id": "tm-1", "source": "apifootball", "source_player_id": "9"},
    ]))  # fmt: skip
    upsert(con, "player_season_stats", pd.DataFrame([
        _stat("apifootball", "9", "tackles", 60, 1800, season=2024),
    ]))  # fmt: skip
    model = defence.DefenceModel.fit(_synthetic_rows())
    out = pd.DataFrame([
        {"player_id": pid, "group": "CB", "competition_id": ITA, "minutes": 2000,
         "birth_date": date(1999, 1, 1), "defence_season": pd.NA,
         **{c: 0.5 for c in defence.PROFILE}}
        for pid in ("tm-1", "tm-2")
    ])  # fmt: skip
    filled = model.fill(con, out, [2025, 2026]).set_index("player_id")
    assert filled.loc["tm-1", "tackles_kind"] == "carried"
    assert filled.loc["tm-1", "defence_season"] == 2024
    assert filled.loc["tm-2", "tackles_kind"] == "estimated"
    assert filled.loc["tm-2", "tackles_lo"] <= filled.loc["tm-2", "tackles"]
