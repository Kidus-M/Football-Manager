import pytest

from recruit_intel.config import ClubConfig, lineup, load_club, load_formations, load_roles


def test_man_utd_loads():
    club = load_club("man_utd")
    assert club.club == "Manchester United"
    assert club.season_start_year == 2026
    assert club.formation == "4-2-3-1"
    assert club.recruitment.max_value_eur is None  # still TBD
    assert club.source_name("fpl") == "Man Utd"


def test_every_role_is_valid():
    roles = load_roles()
    assert "complete_forward" in roles
    for role in roles.values():
        assert role.weights, role.key


def test_every_formation_has_eleven_valid_slots():
    formations = load_formations()
    assert {"4-2-3-1", "3-4-2-1", "4-3-3"} <= set(formations)
    for slots in formations.values():
        assert sum(s.groups == ["GK"] for s in slots) == 1


def test_lineup_applies_club_role_overrides():
    slots = {s.key: s for s in lineup(load_club("man_utd"))}
    assert slots["LCB"].role == "stopper_cb"  # formation default
    assert slots["LB"].role == "attacking_wingback"  # club override
    assert slots["RCB"].role == "ball_playing_cb"


def test_lineup_with_other_formation_ignores_missing_slots():
    slots = {s.key for s in lineup(load_club("man_utd"), "3-4-2-1")}
    assert "LWB" in slots and "LB" not in slots


def _club(**overrides):
    base = load_club("man_utd").model_dump()
    return ClubConfig(**{**base, **overrides})


def test_stray_role_override_is_rejected(tmp_path):
    clubs = tmp_path / "clubs"
    clubs.mkdir()
    (clubs / "x.yaml").write_text(
        "club: X\ncompetition: c\nseason: 2026-27\nformation: 4-2-3-1\n"
        "roles: {LWB: attacking_wingback}\nrecruitment: {pool: [ENG], max_age: 27}\n"
        "peer_group: c\n"
    )
    with pytest.raises(ValueError, match="LWB"):
        load_club("x", clubs_dir=clubs)


def test_unknown_formation_lists_available():
    with pytest.raises(ValueError, match="4-2-3-1"):
        lineup(_club(), "9-0-1")


def test_unknown_club_lists_available():
    with pytest.raises(FileNotFoundError, match="man_utd"):
        load_club("nope")
