"""Positional percentiles against a peer group.

A player's percentile on a metric is the share of qualified peers (same
competition, same position group, enough minutes) he beats, counting ties
as half. Metrics where lower is better are flipped, so 100 is always best.

Players under the minutes threshold are still ranked against the qualified
peers, but flagged `qualified=False`: their numbers are a preview, not a
verdict.

Peers are always measured values. Estimated or older-season values (see
per90_table's `fill`) are ranked against the measured peers in the window, or,
when the window has too few, against measured API-Football seasons of the same
position group in the same league (else in every league). Estimates keep
their range: `<metric>_lo`/`<metric>_hi` become the percentiles of the
prediction's 10th and 90th percentiles.
"""

import numpy as np
import pandas as pd

from recruit_intel.features.metrics import METRICS

MIN_MINUTES_FLOOR = 270  # three full matches
MIN_MINUTES_SHARE = 1 / 3  # of the most minutes anyone in the competition has played
MIN_REFERENCE = 20  # measured peers needed to rank against


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


def _measured_peers(
    qualified: pd.DataFrame, metric: str, reference: pd.DataFrame | None,
    competition: str, group: str,
) -> pd.Series:  # fmt: skip
    kind = f"{metric}_kind"
    peers = qualified[metric] if kind not in qualified else qualified.loc[
        qualified[kind].isna(), metric]  # fmt: skip
    peers = peers.dropna()
    if len(peers) >= MIN_REFERENCE or reference is None:
        return peers
    ref = reference[(reference.metric == metric) & (reference.group == group)]
    same_league = ref[ref.competition_id == competition]
    if len(same_league) >= MIN_REFERENCE:
        return same_league.value
    return ref.value if len(ref) >= MIN_REFERENCE else peers


def percentiles(
    table: pd.DataFrame,
    competition: str,
    min_minutes: int | None = None,
    reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-player percentiles for every metric in `table`, within `competition`.

    Returns the identity columns, `qualified`, `peers` (qualified players in the
    group) and one 0-100 column per metric. `reference`: measured
    distributions for filled values (features/history.py `reference`).
    """
    min_minutes = min_minutes_for(table, competition) if min_minutes is None else min_minutes
    comp = table[(table.competition_id == competition) & table.group.notna()].copy()
    comp["qualified"] = comp.minutes >= min_minutes
    metrics = [m for m in METRICS if m in comp]

    parts = []
    for group, players in comp.groupby("group"):
        qualified = players[players.qualified]
        ident = ["player_id", "name", "position", "group", "detailed_position",
                 "competition_id", "minutes", "qualified", "defence_season",
                 "possession"]  # fmt: skip
        out = players[[c for c in ident if c in players]].copy()
        out["peers"] = len(qualified)
        for m in metrics:
            peers = _measured_peers(qualified, m, reference, competition, group)

            def pct(values: pd.Series, m=m, peers=peers) -> pd.Series:
                p = _rank_against(values, peers)
                return p if METRICS[m].higher_is_better else 100 - p

            out[m] = pct(players[m])
            if f"{m}_lo" in players:
                lo, hi = pct(players[f"{m}_lo"]), pct(players[f"{m}_hi"])
                out[f"{m}_lo"], out[f"{m}_hi"] = np.fmin(lo, hi), np.fmax(lo, hi)
            if f"{m}_kind" in players:
                out[f"{m}_kind"] = players[f"{m}_kind"]
        parts.append(out)
    result = pd.concat(parts, ignore_index=True)
    result.attrs["min_minutes"] = min_minutes
    return result
