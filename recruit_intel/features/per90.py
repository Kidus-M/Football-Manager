"""League-wide per-90 table, one row per canonical player and competition.

Each source's totals are divided by that source's own minutes (Understat
and FPL count minutes independently). Per-90 rates from tiny samples are
noise, so callers filter on `minutes` before ranking. Defensive volume counts
are possession-adjusted (features/context.py).
"""

import duckdb
import pandas as pd

from recruit_intel.features import history
from recruit_intel.features.context import CONTEXT, adjust, window_context
from recruit_intel.features.metrics import METRICS
from recruit_intel.features.positions import group_of


def per90_table(
    con: duckdb.DuckDBPyConnection, season: int | list[int], fill: bool = True
) -> pd.DataFrame:
    """Columns: player_id, name, birth_date, position, detailed_position, group,
    competition_id, minutes, defence_season, team CONTEXT, then one column per
    canonical metric.

    Several seasons pool their totals and minutes into one rate per player and
    competition: a bigger sample, at the cost of weighting old form equally.

    `fill`: API-Football metrics the window lacks are estimated by the defensive
    model (models/defence.py), or taken from the player's latest older season
    when no model is trained. Filled cells get `<metric>_kind` ("carried",
    "estimated" or "older") and, for estimates, `<metric>_lo`/`<metric>_hi`
    (10th/90th percentile of the prediction). Percentiles rank filled values
    against measured ones: pass them `history.reference(con)`.
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
    out = out.merge(window_context(con, seasons), on=["player_id", "competition_id"], how="left")
    out = adjust(out, out.possession)
    out["defence_season"] = pd.NA
    if fill:
        from recruit_intel.models import defence  # the model trains on this table

        model = defence.load()
        if model is not None:
            out = model.fill(con, out, seasons)
        else:
            out = history.fill_from_older(con, out, min(seasons))
    lead = ["player_id", "name", "birth_date", "position", "detailed_position", "group",
            "competition_id", "minutes", "defence_season", *CONTEXT]  # fmt: skip
    metric_cols = [m for m in METRICS if m in out]
    extra = [f"{m}_{s}" for m in metric_cols for s in ("kind", "lo", "hi") if f"{m}_{s}" in out]
    return out[lead + metric_cols + extra]


