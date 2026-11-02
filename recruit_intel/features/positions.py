"""Position groups: who gets compared with whom, and on which metrics.

Percentiles are only meaningful against players doing the same job, so peers
are grouped by Transfermarkt's detailed position (a right-back is compared
with full-backs, not with every defender). Each group also lists the metrics
its radar shows and its strengths/weaknesses are judged on. Defensive
metrics are the ones every league has (API-Football); recoveries come from
FPL and so only exist for Premier League players.
"""

GROUPS: dict[str, dict] = {
    "GK": {
        "label": "Goalkeepers",
        "detailed": ["Goalkeeper"],
        "broad": "GK",
        "metrics": ["goals_prevented", "saves", "goals_conceded", "xg_buildup", "recoveries"],
    },
    "CB": {
        "label": "Centre-backs",
        "detailed": ["Centre-Back"],
        "broad": "DEF",
        "metrics": ["tackles", "interceptions", "blocks", "duels_won", "recoveries", "xg_buildup"],
    },
    "FB": {
        "label": "Full-backs",
        "detailed": ["Right-Back", "Left-Back"],
        "broad": None,
        "metrics": [
            "tackles",
            "interceptions",
            "duels_won",
            "xa",
            "key_passes",
            "xg_chain",
            "xg_buildup",
        ],
    },  # fmt: skip
    "DM": {
        "label": "Defensive midfielders",
        "detailed": ["Defensive Midfield"],
        "broad": None,
        "metrics": [
            "tackles",
            "interceptions",
            "duels_won",
            "recoveries",
            "xg_buildup",
            "xg_chain",
        ],
    },
    "CM": {
        "label": "Central midfielders",
        "detailed": ["Central Midfield", "Midfielder", "Right Midfield", "Left Midfield"],
        "broad": "MID",
        "metrics": [
            "tackles",
            "interceptions",
            "xg_buildup",
            "xg_chain",
            "key_passes",
            "xa",
            "npxg",
        ],
    },  # fmt: skip
    "AM": {
        "label": "Attacking midfielders",
        "detailed": ["Attacking Midfield"],
        "broad": None,
        "metrics": ["npxg", "shots", "xa", "key_passes", "xg_chain", "dribbles_completed"],
    },
    "W": {
        "label": "Wingers",
        "detailed": ["Left Winger", "Right Winger"],
        "broad": None,
        "metrics": ["npxg", "shots", "xa", "key_passes", "dribbles_completed", "tackles"],
    },
    "ST": {
        "label": "Strikers",
        "detailed": ["Centre-Forward", "Second Striker"],
        "broad": "FWD",
        "metrics": ["npxg", "npg", "shots", "xa", "key_passes", "xg_chain", "duels_won"],
    },
}

_BY_DETAILED = {d: g for g, spec in GROUPS.items() for d in spec["detailed"]}
_BY_BROAD = {spec["broad"]: g for g, spec in GROUPS.items() if spec["broad"]}


def group_of(detailed_position: str | None, broad_position: str | None) -> str | None:
    """Detailed position decides; the broad one is the fallback for players
    no Transfermarkt squad lists."""
    if detailed_position in _BY_DETAILED:
        return _BY_DETAILED[detailed_position]
    return _BY_BROAD.get(broad_position)
