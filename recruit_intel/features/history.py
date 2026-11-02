"""Measured defensive seasons from API-Football, possession-adjusted.

API-Football's free plan stops at 2024-25, so its seasons usually sit before
the analysis window. They serve three purposes: anchoring the defensive
estimator (models/defence.py), filling the window directly when no estimator
is trained, and giving percentiles a measured reference distribution where the
window itself has no measured peers.
"""

import duckdb
import pandas as pd

from recruit_intel.features.context import adjust, team_context
from recruit_intel.features.metrics import METRICS
from recruit_intel.features.positions import group_of

SOURCE = "apifootball"
METRICS_MEASURED = [m for m, spec in METRICS.items() if spec.source == SOURCE]
MIN_MINUTES = 450
REFERENCE_MIN_MINUTES = 900


def measured_seasons(
    con: duckdb.DuckDBPyConnection,
    before: int | None = None,
    min_minutes: int = MIN_MINUTES,
    player_ids: list[str] | None = None,
) -> pd.DataFrame:
    """One row per player, season and competition: player_id, season,
    competition_id, minutes, group, possession and METRICS_MEASURED per 90
    (volume counts rescaled to 50% possession)."""
    rows = con.execute(
        f"""
        SELECT ids.player_id, s.source_player_id, s.season, s.competition_id, s.minutes,
               s.metric, s.value
        FROM player_season_stats s
        JOIN player_source_ids ids USING (source, source_player_id)
        WHERE s.source = ? AND s.metric IN (SELECT unnest(?)) AND s.minutes >= ?
          {"AND s.season < ?" if before is not None else ""}
          {"AND ids.player_id IN (SELECT unnest(?))" if player_ids is not None else ""}
        """,
        [SOURCE, METRICS_MEASURED, min_minutes]
        + ([before] if before is not None else [])
        + ([player_ids] if player_ids is not None else []),
    ).df()
    cols = ["player_id", "season", "competition_id", "minutes", "group", "possession"]
    if rows.empty:
        return pd.DataFrame(columns=cols + METRICS_MEASURED)
    key = ["player_id", "season", "competition_id"]
    minutes = rows.drop_duplicates(key + ["source_player_id"]).groupby(key).minutes.sum()
    totals = rows.groupby(key + ["metric"]).value.sum().unstack()
    out = (totals.div(minutes, axis=0) * 90).assign(minutes=minutes).reset_index()

    ctx = team_context(con, sorted(out.season.unique().tolist()))
    out = out.merge(ctx[key + ["possession"]], on=key, how="left")
    out = adjust(out, out.possession)
    players = con.execute(
        "SELECT player_id, position, detailed_position FROM players "
        "WHERE player_id IN (SELECT unnest(?))",
        [out.player_id.unique().tolist()],
    ).df()
    players["group"] = [
        group_of(d, p) for d, p in zip(players.detailed_position, players.position, strict=True)
    ]
    out = out.merge(players[["player_id", "group"]], on="player_id", how="left")
    for m in METRICS_MEASURED:
        if m not in out:
            out[m] = float("nan")
    return out[cols + METRICS_MEASURED]


def latest_before(seasons: pd.DataFrame) -> pd.DataFrame:
    """Each player's latest measured season, competitions pooled by minutes:
    player_id, season, minutes, METRICS_MEASURED."""
    if seasons.empty:
        return pd.DataFrame(columns=["player_id", "season", "minutes", *METRICS_MEASURED])
    latest = seasons[seasons.season == seasons.groupby("player_id").season.transform("max")]
    w = latest.minutes.astype(float)
    values = latest[METRICS_MEASURED].astype(float)
    by = latest.player_id
    # Minutes-weighted mean per metric, over the competitions that recorded it.
    num = values.mul(w, axis=0).groupby(by).sum(min_count=1)
    den = values.notna().mul(w, axis=0).groupby(by).sum()
    out = num / den.where(den > 0)
    out["season"] = latest.groupby("player_id").season.max()
    out["minutes"] = w.groupby(by).sum()
    return out.reset_index()


def reference(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Measured per-90 distributions to rank against where the window has
    none: competition_id, group, metric, value (players with 900+ minutes)."""
    seasons = measured_seasons(con, min_minutes=REFERENCE_MIN_MINUTES)
    return seasons.melt(
        id_vars=["competition_id", "group"], value_vars=METRICS_MEASURED, var_name="metric"
    ).dropna(subset=["value", "group"])


def fill_from_older(con: duckdb.DuckDBPyConnection, out: pd.DataFrame, before: int) -> pd.DataFrame:
    """No estimator trained: fill missing API-Football metrics with each
    player's latest older season, and say which (`defence_season`)."""
    out = out.assign(defence_season=pd.NA)
    latest = latest_before(
        measured_seasons(con, before=before, player_ids=out.player_id.unique().tolist())
    ).set_index("player_id")
    if latest.empty:
        return out
    filled_any = pd.Series(False, index=out.index)
    for metric in METRICS_MEASURED:
        older = out.player_id.map(latest[metric])
        current = out[metric] if metric in out else pd.Series(float("nan"), index=out.index)
        gap = current.isna() & older.notna()
        out[metric] = current.where(~gap, older)
        out.loc[gap, f"{metric}_kind"] = "older"
        filled_any |= gap
    out.loc[filled_any, "defence_season"] = out.loc[filled_any, "player_id"].map(latest.season)
    return out
