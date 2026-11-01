import pandas as pd

from recruit_intel.analysis.squad import _match


def test_match_long_fpl_names_to_short_understat_names():
    fpl = pd.Series(["Matheus Santos Carneiro da Cunha", "Bruno Borges Fernandes", "Chido Obi"])
    us = pd.Series(["Bruno Fernandes", "Matheus Cunha", "Benjamin Sesko"])
    assert _match(fpl, us) == {0: 1, 1: 0}
