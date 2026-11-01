"""Manager profiles: the style a manager brings, separated from his squads.

Every Understat match is assigned to the manager in charge on its date
(Transfermarkt spells, via the Understat-team -> Transfermarkt-club name
map). Each stint (manager x team x season, 8+ matches) gets a style vector
from its own matches, adjusted for that team's strength as in team_style.
A manager's profile is the match-weighted mean over his stints; a trait
that holds across several clubs is his, not something one squad forced.

Shot quality and build-up share come from season-level player totals, so
for a stint that's only part of a season they are the season's values.
"""

import duckdb
import numpy as np
import pandas as pd

from recruit_intel.identity.teams import match_teams
from recruit_intel.models.team_style import (
    MIN_MATCHES,
    STYLE_FEATURES,
    StyleModel,
    describe,
    team_seasons,
)

PROFILE_THRESHOLD = 0.6  # averaged z-scores shrink, so traits need less distance


def club_map(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Understat team_id -> Transfermarkt club_id, matched within competition."""
    us = con.execute("SELECT DISTINCT team_id, team_name, competition_id FROM team_matches").df()
    tm = con.execute(
        """
        SELECT DISTINCT source_team_id AS club_id, name, competition_id
        FROM teams WHERE source = 'transfermarkt'
        """
    ).df()
    rows = []
    for comp, group in us.groupby("competition_id"):
        tm_comp = tm[tm.competition_id == comp]
        by_name = dict(zip(tm_comp.name, tm_comp.club_id, strict=True))
        mapping = match_teams(group.team_name.unique().tolist(), list(by_name))
        for r in group.itertuples():
            if r.team_name in mapping:
                rows.append((r.team_id, by_name[mapping[r.team_name]]))
    return pd.DataFrame(rows, columns=["team_id", "club_id"]).drop_duplicates("team_id")


def stints(con: duckdb.DuckDBPyConnection, model: StyleModel) -> pd.DataFrame:
    """One row per manager x team x season with 8+ matches: identity, matches,
    strength and the adjusted style vector (STYLE_FEATURES columns, z-scores)."""
    con.register("_club_map", club_map(con))
    try:
        matches = con.execute(
            """
            SELECT m.*, s.manager_id, s.name AS manager
            FROM team_matches m
            JOIN _club_map c USING (team_id)
            JOIN manager_spells s
              ON s.club_id = c.club_id
             AND CAST(m.match_date AS DATE) >= s.appointed
             AND (s.ended IS NULL OR CAST(m.match_date AS DATE) <= s.ended)
            -- On a handover day keep the newer appointment.
            QUALIFY row_number() OVER (PARTITION BY m.team_id, m.match_date
                                       ORDER BY s.appointed DESC) = 1
            """
        ).df()
    finally:
        con.unregister("_club_map")
    if matches.empty:
        raise LookupError("no manager data; run `recruit ingest --managers`")

    g = matches.groupby(["manager_id", "team_id", "season", "competition_id"])
    st = g.agg(
        manager=("manager", "first"),
        team_name=("team_name", "first"),
        matches=("match_date", "count"),
        first_match=("match_date", "min"),
        ppda_p=("ppda_passes", "sum"), ppda_a=("ppda_actions", "sum"),
        oppda_p=("oppda_passes", "sum"), oppda_a=("oppda_actions", "sum"),
        deep=("deep", "sum"), deep_allowed=("deep_allowed", "sum"),
        npxg=("npxg", "mean"), npxga=("npxga", "mean"), xpts_pm=("xpts", "mean"),
    ).reset_index()  # fmt: skip
    st = st[st.matches >= MIN_MATCHES]
    st["ppda"] = st.ppda_p / st.ppda_a
    st["oppda"] = st.oppda_p / st.oppda_a
    st["territory"] = st.deep / (st.deep + st.deep_allowed)
    season_level = team_seasons(con)[["team_id", "season", "xg_per_shot", "buildup_share"]]
    st = st.merge(season_level, on=["team_id", "season"]).dropna(subset=STYLE_FEATURES)

    adjusted = model.adjusted(st)
    out = st[["manager_id", "manager", "team_id", "team_name", "competition_id", "season",
              "first_match", "matches", "xpts_pm"]].copy()  # fmt: skip
    out[STYLE_FEATURES] = adjusted
    return out.sort_values("first_match").reset_index(drop=True)


def profiles(stint_rows: pd.DataFrame) -> pd.DataFrame:
    """One row per manager: matches, clubs, seasons, weighted style z-scores."""

    def agg(group: pd.DataFrame) -> pd.Series:
        w = group.matches.to_numpy(float)
        z = (group[STYLE_FEATURES].to_numpy() * w[:, None]).sum(axis=0) / w.sum()
        return pd.Series(
            {
                "manager": group.manager.iloc[-1],
                "matches": int(w.sum()),
                "clubs": group.team_id.nunique(),
                "seasons": group.season.nunique(),
                **dict(zip(STYLE_FEATURES, z, strict=True)),
            }  # fmt: skip
        )

    return (
        stint_rows.groupby("manager_id")
        .apply(agg)
        .reset_index()
        .sort_values("matches", ascending=False)
    )


def signature(profile: pd.Series) -> str:
    return describe(profile[STYLE_FEATURES].to_numpy(float), 3, PROFILE_THRESHOLD)


def similar_managers(profs: pd.DataFrame, manager_id: str, k: int = 6,
                     min_matches: int = 38) -> pd.DataFrame:  # fmt: skip
    pool = profs[(profs.matches >= min_matches) | (profs.manager_id == manager_id)]
    v = pool[STYLE_FEATURES].to_numpy(float)
    target = pool.manager_id.to_numpy() == manager_id
    d = np.linalg.norm(v - v[target][0], axis=1)
    return pool.assign(distance=d)[~target].sort_values("distance").head(k)
