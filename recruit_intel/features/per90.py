"""League-wide per-90 table, one row per canonical player and competition.

Each source's totals are divided by that source's own minutes (Understat
and FPL count minutes independently). Per-90 rates from tiny samples are
noise, so callers filter on `minutes` before ranking.
"""

import duckdb
import pandas as pd

from recruit_intel.features.metrics import METRICS
from recruit_intel.features.positions import group_of


def per90_table(con: duckdb.DuckDBPyConnection, season: int | list[int]) -> pd.DataFrame:
    """Columns: player_id, name, birth_date, position, detailed_position, group,
    competition_id, minutes, then one column per canonical metric.

    Several seasons pool their totals and minutes into one rate per player and
    competition: a bigger sample, at the cost of weighting old form equally.
    """
    seasons = [season] if isinstance(season, int) else list(season)
    rows = con.execute(
        """
        SELECT ids.player_id, s.source, s.source_player_id, s.season, s.competition_id,
               s.minutes, s.metric, s.value
        FROM player_season_stats s
        JOIN player_source_ids ids USING (source, source_player_id)
        WHERE s.season IN (SELECT unnest(?)) AND s.metric IN (SELECT unnest(?))
        """,
        [seasons, list(METRICS)],
    ).df()
    if rows.empty:
        raise LookupError(f"no stats for seasons {seasons}; run `recruit ingest`")

    # A player can appear more than once per source and competition (two clubs,
    # several seasons): sum totals and minutes per (player, competition, source).
    minutes = (
        rows.drop_duplicates(
            ["player_id", "source", "source_player_id", "season", "competition_id"]
        )
        .groupby(["player_id", "competition_id", "source"])
        .minutes.sum()
    )
    totals = rows.groupby(["player_id", "competition_id", "source", "metric"]).value.sum().unstack()
    per90 = totals.div(minutes.where(minutes > 0), axis=0) * 90
    # Several sources publish some metrics (FPL and Understat both have xG);
    # each metric is taken only from the source the catalogue assigns it.
    long = per90.stack().rename("value").reset_index()
    # Its fallback sources fill in only for players the main source lacks.
    rank = [
        METRICS[m].sources.index(s) if s in METRICS[m].sources else -1
        for m, s in zip(long.metric, long.source, strict=True)
    ]
    long = long.assign(_rank=rank)
    long = (
        long[long._rank >= 0]
        .sort_values("_rank")
        .drop_duplicates(["player_id", "competition_id", "metric"])
    )
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
    out = _fill_from_older_seasons(con, out, min(seasons))
    lead = ["player_id", "name", "birth_date", "position", "detailed_position", "group",
            "competition_id", "minutes", "defence_season"]  # fmt: skip
    metric_cols = [m for m in METRICS if m in out]
    return out[lead + metric_cols]


# Sources whose older seasons may stand in for the window. API-Football's free
# plan stops at 2024-25, so its defensive stats are older than the window;
# they fill only what the window lacks, and `defence_season` says from when.
OLDER_FILL_SOURCES = ("apifootball",)
OLDER_MIN_MINUTES = 450


def _fill_from_older_seasons(
    con: duckdb.DuckDBPyConnection, out: pd.DataFrame, before: int
) -> pd.DataFrame:
    out = out.assign(defence_season=pd.NA)
    metrics = [m for m, spec in METRICS.items() if spec.source in OLDER_FILL_SOURCES]
    rows = con.execute(
        """
        SELECT ids.player_id, s.season, s.competition_id, s.minutes, s.metric, s.value
        FROM player_season_stats s
        JOIN player_source_ids ids USING (source, source_player_id)
        WHERE s.source IN (SELECT unnest(?)) AND s.season < ?
          AND s.metric IN (SELECT unnest(?)) AND s.minutes >= ?
          AND ids.player_id IN (SELECT unnest(?))
        """,
        [list(OLDER_FILL_SOURCES), before, metrics, OLDER_MIN_MINUTES,
         out.player_id.unique().tolist()],
    ).df()  # fmt: skip
    if rows.empty:
        return out
    # Each player's latest older season, summed over competitions he played in.
    rows = rows[rows.season == rows.groupby("player_id").season.transform("max")]
    minutes = (
        rows.drop_duplicates(["player_id", "competition_id"]).groupby("player_id").minutes.sum()
    )
    per90 = rows.groupby(["player_id", "metric"]).value.sum().unstack().div(minutes, axis=0) * 90
    used_season = rows.groupby("player_id").season.max()

    filled_any = pd.Series(False, index=out.index)
    for metric in per90.columns:
        older = out.player_id.map(per90[metric])
        current = out[metric] if metric in out else pd.Series(float("nan"), index=out.index)
        gap = current.isna() & older.notna()
        out[metric] = current.where(~gap, older)
        filled_any |= gap
    out.loc[filled_any, "defence_season"] = out.loc[filled_any, "player_id"].map(used_season)
    return out
