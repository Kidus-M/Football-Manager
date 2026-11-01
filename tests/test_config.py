import pytest

from recruit_intel.config import load_club, load_roles


def test_man_utd_loads():
    club = load_club("man_utd")
    assert club.club == "Manchester United"
    assert club.season_start_year == 2026
    assert club.formation is None  # still TBD
    assert club.recruitment.max_value_eur is None
    assert club.source_name("fpl") == "Man Utd"


def test_every_role_is_valid():
    roles = load_roles()
    assert "complete_forward" in roles
    for role in roles.values():
        assert role.weights, role.key


def test_unknown_club_lists_available():
    with pytest.raises(FileNotFoundError, match="man_utd"):
        load_club("nope")
