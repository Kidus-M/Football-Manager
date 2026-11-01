import pandas as pd
import pytest

from recruit_intel.analysis.tactics import (
    LeverReading,
    advise,
    compare,
    instructions,
    lever_readings,
    load_config,
    reweight,
)
from recruit_intel.config import Role, load_roles

CONFIG = {
    "added_weight": 0.2,
    "settings": {
        "press": {"high": {"groups": ["ST"], "boost": {"tackles": 2.0, "npxg": 0.5}}},
    },
}
ROLES = {
    "fwd": Role(key="fwd", label="F", groups=["ST"], weights={"npxg": 0.8, "xa": 0.2}),
    "cb": Role(key="cb", label="CB", groups=["CB"], weights={"cbi": 1.0}),
}


def test_reweight_scales_adds_and_skips_other_groups():
    out = reweight(ROLES, {"press": "high"}, CONFIG)
    assert out["fwd"].weights == {"npxg": 0.4, "xa": 0.2, "tackles": pytest.approx(0.2)}
    assert out["cb"].weights == {"cbi": 1.0}
    assert ROLES["fwd"].weights == {"npxg": 0.8, "xa": 0.2}  # originals untouched


def test_unknown_setting_lists_options():
    with pytest.raises(ValueError, match="press"):
        reweight(ROLES, {"press": "extreme"}, CONFIG)


def test_shipped_settings_produce_valid_roles():
    roles = load_roles()
    for lever, options in load_config()["settings"].items():
        for option in options:
            for role in reweight(roles, {lever: option}).values():
                assert all(w > 0 for w in role.weights.values())


STARTERS = pd.DataFrame(
    [
        {"name": "Keeper", "group": "GK", "xg_buildup": 20},
        {"name": "Winger", "group": "W", "tackles": 80, "recoveries": 70, "npxg": 85,
         "shots": 70, "xa": 30},
        {"name": "Mid", "group": "CM", "tackles": 75, "recoveries": 85, "xg_buildup": 50},
        {"name": "Nine", "group": "ST", "tackles": 40, "recoveries": None, "npxg": 30,
         "shots": 35, "xa": 70},
    ]
)  # fmt: skip
LEVER_CONFIG = {
    "levers": {
        "press": {"slots_groups": ["ST", "W", "CM"], "metrics": ["tackles", "recoveries"],
                  "high_at": 60, "low_at": 40},
        "finishing": {"slots_groups": ["ST"], "metrics": ["npxg", "shots"],
                      "high_at": 60, "low_at": 40},
    }
}  # fmt: skip


def test_lever_readings_average_relevant_starters():
    readings = {r.lever: r for r in lever_readings(STARTERS, LEVER_CONFIG)}
    press = readings["press"]
    assert press.squad == pytest.approx((75 + 80 + 40) / 3)  # missing recovery skipped
    assert press.verdict == "high" and press.best[0] == "Mid 80"
    assert readings["finishing"].verdict == "low"


@pytest.mark.parametrize(
    "verdict,style_z,expected",
    [("high", -0.8, "support a higher press"), ("low", 0.9, "beyond what these players"),
     ("mid", 0.1, "aligned"), ("high", None, "no style comparison")],
)  # fmt: skip
def test_advice(verdict, style_z, expected):
    r = LeverReading("press", 65.0, verdict, [], [], style_z=style_z)
    assert expected in advise(r)


def test_instructions_fire_on_percentile_rules():
    said = instructions(STARTERS)

    def lines(who):
        return " | ".join(text for name, text in said if name == who)

    assert "build through the centre-backs" in lines("Keeper")
    assert "inside foot" in lines("Winger") and "trigger the press" in lines("Winger")
    assert "drop" in lines("Nine") and "few chances" in lines("Nine")
    assert "Mid" in lines("Mid")  # ball-winner rule (recoveries 85)


def test_compare_lists_only_changed_slots():
    base = pd.DataFrame([
        {"slot": "ST", "starter": "A", "fit": 70, "severity": None},
        {"slot": "RW", "starter": "B", "fit": 60, "severity": "medium"},
    ])  # fmt: skip
    what_if = pd.DataFrame([
        {"slot": "ST", "starter": "A", "fit": 70, "severity": None},
        {"slot": "RW", "starter": "C", "fit": 52, "severity": "high"},
    ])  # fmt: skip
    changes = compare(base, what_if)
    assert [(c.slot, c.before, c.after, c.severity_after) for c in changes] == [
        ("RW", "B", "C", "high")
    ]
