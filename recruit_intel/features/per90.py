"""League-wide per-90 table, one row per canonical player and competition.

Each source's totals are divided by that source's own minutes (Understat
and FPL count minutes independently). Per-90 rates from tiny samples are
noise, so callers filter on `minutes` before ranking.
"""

import duckdb
import pandas as pd

from recruit_intel.features.metrics import METRICS
from recruit_intel.features.positions import group_of


def per90_table(con: duckdb.DuckDBPyConnection, season: int) -> pd.DataFrame:
    """Columns: player_id, name, birth_date, position, detailed_position, group,
    competition_id, minutes, then one column per canonical metric."""
    rows = con.execute(
        """
        SELECT ids.player_id, s.source, s.source_player_id, s.competition_id, s.minutes,
               s.metric, s.value
        FROM player_season_stats s
        JOIN player_source_ids ids USING (source, source_player_id)
        WHERE s.season = ? AND s.metric IN (SELECT unnest(?))
        """,
        [season, list(METRICS)],
    ).df()
    if rows.empty:
        raise LookupError(f"no stats for season {season}; run `recruit ingest`")

    # One player can appear twice in a source (two clubs, same league):
    # sum totals and minutes per (player, competition, source) first.
    minutes = (
        rows.drop_duplicates(["player_id", "source", "source_player_id", "competition_id"])
        .groupby(["player_id", "competition_id", "source"])
        .minutes.sum()
    )
    totals = rows.groupby(["player_id", "competition_id", "source", "metric"]).value.sum().unstack()
    per90 = totals.div(minutes.where(minutes > 0), axis=0) * 90
    # Several sources publish some metrics (FPL and Understat both have xG);
    # each metric is taken only from the source the catalogue assigns it.
    long = per90.stack().rename("value").reset_index()
    long = long[long.source == long.metric.map(lambda m: METRICS[m].source)]
    wide = long.pivot_table(index=["player_id", "competition_id"], columns="metric", values="value")
    wide["minutes"] = minutes.groupby(level=["player_id", "competition_id"]).max()
    if {"xg_conceded", "goals_conceded"} <= set(wide.columns):
        # A keeper's own contribution, unlike saves (which mostly count shots
        # faced). Pre-shot xG, so it's a rough proxy for goals prevented.
        wide["goals_prevented"] = wide["xg_conceded"] - wide["goals_conceded"]
    wide = wide.reset_index()

    players = con.execute(
        "SELECT player_id, name, birth_date, position, detailed_position FROM players"
    ).df()
    out = players.merge(wide, on="player_id")
    out["group"] = [
        group_of(d, b) for d, b in zip(out.detailed_position, out.position, strict=True)
    ]
    lead = ["player_id", "name", "birth_date", "position", "detailed_position", "group",
            "competition_id", "minutes"]  # fmt: skip
    metric_cols = [m for m in METRICS if m in out]
    return out[lead + metric_cols]
