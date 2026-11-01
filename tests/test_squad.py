from datetime import date

import pandas as pd

from recruit_intel.analysis.squad import _match


def test_match_long_fpl_names_to_short_understat_names():
    fpl = pd.DataFrame(
        {"name": ["Matheus Santos Carneiro da Cunha", "Bruno Borges Fernandes", "Chido Obi"]}
    )
    us = pd.DataFrame({"name": ["Bruno Fernandes", "Matheus Cunha", "Benjamin Sesko"]})
    assert _match(fpl, us) == {0: 1, 1: 0}


def test_birth_date_disambiguates_shared_names():
    fpl = pd.DataFrame(
        {"name": ["Jack Fletcher", "Tyler Fletcher"], "birth_date": [date(2007, 3, 19)] * 2}
    )
    tm = pd.DataFrame(
        {
            "name": ["Tyler Fletcher", "Jack Fletcher"],
            "birth_date": [date(2007, 3, 19), date(2007, 3, 19)],
        }
    )
    assert _match(fpl, tm) == {0: 1, 1: 0}


def test_conflicting_birth_dates_never_match():
    fpl = pd.DataFrame({"name": ["Harry Amass"], "birth_date": [date(2007, 3, 16)]})
    tm = pd.DataFrame({"name": ["Harry Amass"], "birth_date": [date(2000, 1, 1)]})
    assert _match(fpl, tm) == {}
