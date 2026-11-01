"""Canonical metric catalogue.

Every metric the platform scores on is listed here once, with the source that
provides it. Ingestion adapters map their raw fields onto these names, and role
weights in config/roles.yaml may only reference these names.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Metric:
    name: str
    label: str
    source: str
    higher_is_better: bool = True


METRICS: dict[str, Metric] = {
    m.name: m
    for m in [
        # Understat (top 5 leagues + RPL)
        Metric("goals", "Goals", "understat"),
        Metric("npg", "Non-penalty goals", "understat"),
        Metric("assists", "Assists", "understat"),
        Metric("xg", "xG", "understat"),
        Metric("npxg", "Non-penalty xG", "understat"),
        Metric("xa", "xA", "understat"),
        Metric("shots", "Shots", "understat"),
        Metric("key_passes", "Key passes", "understat"),
        Metric("xg_chain", "xGChain", "understat"),
        Metric("xg_buildup", "xGBuildup", "understat"),
        # Fantasy Premier League (Premier League only)
        Metric("tackles", "Tackles", "fpl"),
        Metric("cbi", "Clearances, blocks, interceptions", "fpl"),
        Metric("recoveries", "Ball recoveries", "fpl"),
        Metric("saves", "Saves", "fpl"),
        Metric("goals_conceded", "Goals conceded", "fpl", higher_is_better=False),
        Metric("xg_conceded", "xG conceded", "fpl", higher_is_better=False),
        Metric("defensive_contribution", "Defensive contribution", "fpl"),
        Metric("clean_sheets", "Clean sheets", "fpl"),
        Metric("starts", "Starts", "fpl"),
        # Derived in features/per90.py from the metrics above
        Metric("goals_prevented", "Goals prevented (xG conceded - goals)", "derived"),
    ]
}
