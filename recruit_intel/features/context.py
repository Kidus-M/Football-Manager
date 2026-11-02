"""Team context for a player's numbers: how much defending his team does.

A centre-back at a side that keeps the ball has fewer tackles to make, so raw
defensive counts punish players at dominant teams. Defensive volume metrics are
rescaled to what they would be at 50% possession (the usual possession
adjustment), with possession estimated from Understat's per-match PPDA counts:
our passes and the opponent's passes in the zone PPDA covers. That proxy
exists for every top-five-league team and season back to 2016.
"""

import duckdb
import numpy as np
import pandas as pd

# Counts that scale with how long the opponent has the ball.
POSSESSION_ADJUSTED = ("tackles", "interceptions", "blocks", "dribbled_past", "cbi",
                       "recoveries", "defensive_contribution")  # fmt: skip
CONTEXT = ["possession", "ppda", "xga_pm", "deep_allowed_pm"]
# Caps the correction at about x0.7-x1.7: beyond that the proxy is mostly noise.
POSSESSION_RANGE = (0.3, 0.7)


def possession_factor(possession: pd.Series) -> pd.Series:
    """Multiplier to a 50%-possession equivalent; 1 where possession is unknown."""
    p = possession.astype(float).clip(*POSSESSION_RANGE)
    return (0.5 / (1 - p)).fillna(1.0)


def team_context(con: duckdb.DuckDBPyConnection, seasons: list[int]) -> pd.DataFrame:
    """One row per player, season and competition (Understat players):
    player_id, season, competition_id, minutes and CONTEXT columns, averaged
    over his teams when he changed club mid-season."""
    return con.execute(
        """
        WITH t AS (
            SELECT team_name, season, competition_id,
                   sum(oppda_passes)::DOUBLE
                       / nullif(sum(oppda_passes) + sum(ppda_passes), 0) AS possession,
                   sum(ppda_passes)::DOUBLE / nullif(sum(ppda_actions), 0) AS ppda,
                   avg(xga) AS xga_pm,
                   avg(deep_allowed) AS deep_allowed_pm
            FROM team_matches
            WHERE source = 'understat' AND season IN (SELECT unnest(?))
            GROUP BY ALL
        ), split AS (
            SELECT DISTINCT ids.player_id, s.season, s.competition_id, s.minutes,
                   unnest(string_split(s.team_name, ',')) AS team
            FROM player_season_stats s
            JOIN player_source_ids ids USING (source, source_player_id)
            WHERE s.source = 'understat' AND s.season IN (SELECT unnest(?))
        ), p AS (
            SELECT player_id, season, competition_id, minutes, trim(team) AS team_name
            FROM split
        )
        SELECT p.player_id, p.season, p.competition_id, any_value(p.minutes) AS minutes,
               avg(t.possession) AS possession, avg(t.ppda) AS ppda,
               avg(t.xga_pm) AS xga_pm, avg(t.deep_allowed_pm) AS deep_allowed_pm
        FROM p JOIN t USING (team_name, season, competition_id)
        GROUP BY ALL
        """,
        [seasons, seasons],
    ).df()


def window_context(con: duckdb.DuckDBPyConnection, seasons: list[int]) -> pd.DataFrame:
    """CONTEXT per player and competition over a window of seasons, weighted by
    the minutes he played in each."""
    ctx = team_context(con, seasons)
    if ctx.empty:
        return pd.DataFrame(columns=["player_id", "competition_id", *CONTEXT])
    w = ctx.minutes.clip(lower=1).astype(float)
    weighted = ctx[CONTEXT].mul(w, axis=0).assign(_w=w)
    weighted[["player_id", "competition_id"]] = ctx[["player_id", "competition_id"]]
    sums = weighted.groupby(["player_id", "competition_id"]).sum(min_count=1)
    out = sums[CONTEXT].div(sums._w, axis=0)
    return out.replace([np.inf, -np.inf], np.nan).reset_index()


def adjust(table: pd.DataFrame, possession: pd.Series) -> pd.DataFrame:
    """POSSESSION_ADJUSTED columns of `table` rescaled to 50% possession."""
    factor = possession_factor(possession)
    out = table.copy()
    for m in POSSESSION_ADJUSTED:
        if m in out:
            out[m] = out[m] * factor.to_numpy()
    return out
