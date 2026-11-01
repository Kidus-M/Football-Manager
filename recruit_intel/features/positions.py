"""Position groups: who gets compared with whom, and on which metrics.

Percentiles are only meaningful against players doing the same job, so peers
are grouped by Transfermarkt's detailed position (a right-back is compared
with full-backs, not with every defender). Each group also lists the metrics
its radar shows and its strengths/weaknesses are judged on.
"""

GROUPS: dict[str, dict] = {
    "GK": {
        "label": "Goalkeepers",
        "detailed": ["Goalkeeper"],
        "broad": "GK",
        "metrics": ["saves", "goals_conceded", "xg_conceded", "xg_buildup", "recoveries"],
    },
    "CB": {
        "label": "Centre-backs",
        "detailed": ["Centre-Back"],
        "broad": "DEF",
        "metrics": ["cbi", "tackles", "recoveries", "xg_buildup", "xg_chain", "npxg"],
    },
    "FB": {
        "label": "Full-backs",
        "detailed": ["Right-Back", "Left-Back"],
        "broad": None,
        "metrics": [
            "tackles",
            "cbi",
            "recoveries",
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
        "metrics": ["tackles", "recoveries", "cbi", "xg_buildup", "xg_chain", "key_passes"],
    },
    "CM": {
        "label": "Central midfielders",
        "detailed": ["Central Midfield", "Midfielder", "Right Midfield", "Left Midfield"],
        "broad": "MID",
        "metrics": [
            "tackles",
            "recoveries",
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
        "metrics": ["npxg", "shots", "xa", "key_passes", "xg_chain", "recoveries"],
    },
    "W": {
        "label": "Wingers",
        "detailed": ["Left Winger", "Right Winger"],
        "broad": None,
        "metrics": ["npxg", "shots", "xa", "key_passes", "xg_chain", "tackles"],
    },
    "ST": {
        "label": "Strikers",
        "detailed": ["Centre-Forward", "Second Striker"],
        "broad": "FWD",
        "metrics": ["npxg", "npg", "shots", "xa", "key_passes", "xg_chain"],
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
