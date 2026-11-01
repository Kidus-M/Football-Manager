from recruit_intel.identity.teams import match_teams, team_key

TM_PL = [
    "Manchester City", "Arsenal FC", "Chelsea FC", "Liverpool FC", "Tottenham Hotspur",
    "Newcastle United", "Brighton & Hove Albion", "Brentford FC", "Aston Villa",
    "AFC Bournemouth", "Crystal Palace", "Nottingham Forest", "Sunderland AFC", "Leeds United",
    "Everton FC", "Fulham FC", "Ipswich Town", "Coventry City", "Hull City", "Manchester United",
]  # fmt: skip
FPL = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton", "Chelsea", "Coventry City",
    "Crystal Palace", "Everton", "Fulham", "Hull City", "Ipswich Town", "Leeds", "Liverpool",
    "Man City", "Man Utd", "Newcastle", "Nott'm Forest", "Spurs", "Sunderland",
]  # fmt: skip


def test_team_key_strips_club_tokens():
    assert team_key("1. FC Union Berlin") == "union berlin"
    assert team_key("AFC Bournemouth") == "bournemouth"


def test_every_fpl_team_maps_one_to_one():
    mapping = match_teams(FPL, TM_PL)
    assert len(mapping) == 20 and len(set(mapping.values())) == 20
    assert mapping["Man Utd"] == "Manchester United"
    assert mapping["Man City"] == "Manchester City"
    assert mapping["Spurs"] == "Tottenham Hotspur"
    assert mapping["Nott'm Forest"] == "Nottingham Forest"


def test_unknown_team_is_left_unmapped():
    assert match_teams(["Real Madrid"], TM_PL) == {}
