"""Positional percentiles against a peer group.

A player's percentile on a metric is the share of qualified peers (same
competition, same position group, enough minutes) he beats, counting ties
as half. Metrics where lower is better are flipped, so 100 is always best.

Players under the minutes threshold are still ranked against the qualified
peers, but flagged `qualified=False`: their numbers are a preview, not a
verdict.
"""

import numpy as np
import pandas as pd

from recruit_intel.features.metrics import METRICS

MIN_MINUTES_FLOOR = 270  # three full matches
MIN_MINUTES_SHARE = 1 / 3  # of the most minutes anyone in the competition has played


def min_minutes_for(table: pd.DataFrame, competition: str) -> int:
    """Scale the threshold with the season: early on nobody has 900 minutes."""
    top = table.loc[table.competition_id == competition, "minutes"].max()
    return int(max(MIN_MINUTES_FLOOR, MIN_MINUTES_SHARE * top))


def _rank_against(values: pd.Series, peers: pd.Series) -> pd.Series:
    peers = np.sort(peers.dropna().to_numpy())
    if len(peers) == 0:
        return pd.Series(np.nan, index=values.index)
    below = np.searchsorted(peers, values.to_numpy(), side="left")
    at_or_below = np.searchsorted(peers, values.to_numpy(), side="right")
    pct = (below + 0.5 * (at_or_below - below)) / len(peers) * 100
    return pd.Series(pct, index=values.index).where(values.notna())


def percentiles(
    table: pd.DataFrame, competition: str, min_minutes: int | None = None
) -> pd.DataFrame:
    """Per-player percentiles for every metric in `table`, within `competition`.

    Returns the identity columns, `qualified`, `peers` (qualified players in the
    group) and one 0-100 column per metric.
    """
    min_minutes = min_minutes_for(table, competition) if min_minutes is None else min_minutes
    comp = table[(table.competition_id == competition) & table.group.notna()].copy()
    comp["qualified"] = comp.minutes >= min_minutes
    metrics = [m for m in METRICS if m in comp]

    parts = []
    for _group, players in comp.groupby("group"):
        qualified = players[players.qualified]
        ident = ["player_id", "name", "position", "group", "detailed_position", "minutes",
                 "qualified"]  # fmt: skip
        out = players[[c for c in ident if c in players]].copy()
        out["peers"] = len(qualified)
        for m in metrics:
            pct = _rank_against(players[m], qualified[m])
            out[m] = pct if METRICS[m].higher_is_better else 100 - pct
        parts.append(out)
    result = pd.concat(parts, ignore_index=True)
    result.attrs["min_minutes"] = min_minutes
    return result
