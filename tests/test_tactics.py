import pandas as pd
import pytest

from recruit_intel.analysis.tactics import compare, load_config, reweight
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
