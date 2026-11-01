"""Current-squad table for the configured club.

Joins the club's FPL players (positions, birth dates, defensive counts) with
its Understat players (xG family) and Transfermarkt players (market value,
contract). Until the identity module lands, the join is a within-club
name-token match, confirmed by birth date where both sides have one. That is
reliable for one squad of ~30 but not for league-wide matching.
"""

from datetime import date

import duckdb
import pandas as pd

from recruit_intel.config import ClubConfig
from recruit_intel.identity.names import tokens

PER90_METRICS = {
    "fpl": ["tackles", "cbi", "recoveries", "saves"],
    "understat": ["npxg", "xa", "xg_chain", "xg_buildup", "shots", "key_passes"],
}


def _source_squad(con, source: str, team: str, season: int) -> pd.DataFrame:
    """One row per player: identity columns, minutes, and per-90 metrics."""
    metrics = PER90_METRICS[source]
    rows = con.execute(
        """
        SELECT p.source_player_id, p.name, p.position, p.birth_date,
               s.minutes, s.metric, s.value
        FROM player_season_stats s
        JOIN source_players p USING (source, source_player_id)
        WHERE s.source = ? AND s.season = ? AND p.team_name = ?
          AND s.metric IN (SELECT unnest(?))
        """,
        [source, season, team, metrics],
    ).df()
    if rows.empty:
        return rows
    ident = ["source_player_id", "name", "position", "birth_date", "minutes"]
    wide = (
        rows[ident]
        .drop_duplicates("source_player_id")
        .merge(
            rows.pivot(index="source_player_id", columns="metric", values="value").reset_index(),
            on="source_player_id",
        )
    )
    for m in metrics:
        if m in wide:
            wide[m] = (wide[m] / wide["minutes"] * 90).where(wide["minutes"] > 0)
    return wide


def _transfermarkt_squad(con, club_id: str) -> pd.DataFrame:
    """Latest market snapshot for each of the club's Transfermarkt players."""
    return con.execute(
        """
        SELECT p.name, p.birth_date, p.detailed_position, m.value_eur, m.contract_end
        FROM source_players p
        JOIN market m USING (source, source_player_id)
        WHERE p.source = 'transfermarkt' AND p.team_id = ?
        QUALIFY row_number() OVER (PARTITION BY p.source_player_id
                                   ORDER BY m.snapshot_date DESC) = 1
        """,
        [club_id],
    ).df()


def _match(left: pd.DataFrame, right: pd.DataFrame) -> dict[int, int]:
    """Left row index -> right row index, by name-token overlap.

    When both rows carry a birth date, it must agree, and agreement breaks
    ties between players who share name tokens.
    """
    has_dob = "birth_date" in left and "birth_date" in right
    out: dict[int, int] = {}
    taken: set[int] = set()
    for i, lrow in left.iterrows():
        a = tokens(lrow["name"])
        best, best_score = None, 0.0
        for j, rrow in right.iterrows():
            if j in taken:
                continue
            b = tokens(rrow["name"])
            score = len(a & b) / min(len(a), len(b))
            if has_dob and pd.notna(lrow["birth_date"]) and pd.notna(rrow["birth_date"]):
                if pd.Timestamp(lrow["birth_date"]) != pd.Timestamp(rrow["birth_date"]):
                    continue
                score += 1
            if score > best_score:
                best, best_score = j, score
        if best is not None and best_score >= 0.5:
            out[i] = best
            taken.add(best)
    return out


def current_squad(con: duckdb.DuckDBPyConnection, club: ClubConfig, season: int) -> pd.DataFrame:
    fpl = _source_squad(con, "fpl", str(club.source_name("fpl")), season)
    if fpl.empty:
        raise LookupError(f"no FPL rows for {club.club} in {season}; run `recruit ingest` first")
    us = _source_squad(con, "understat", str(club.source_name("understat")), season)

    us_cols = [m for m in PER90_METRICS["understat"] if m in us]
    joined = fpl.copy()
    for col in us_cols:
        joined[col] = float("nan")
    if not us.empty:
        for i, j in _match(fpl, us).items():
            joined.loc[i, us_cols] = us.loc[j, us_cols].to_numpy()

    tm = _transfermarkt_squad(con, str(club.source_name("transfermarkt_id")))
    tm_cols = ["detailed_position", "value_eur", "contract_end"]
    for col in tm_cols:
        joined[col] = None
    for i, j in _match(fpl, tm).items():
        joined.loc[i, tm_cols] = tm.loc[j, tm_cols].to_numpy()
    joined["value_m_eur"] = pd.to_numeric(joined["value_eur"]) / 1e6
    joined["contract"] = pd.to_datetime(joined["contract_end"]).dt.year.astype("Int64")

    born = pd.to_datetime(joined["birth_date"])
    joined["age"] = ((pd.Timestamp(date.today()) - born).dt.days // 365.25).astype("Int64")
    order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    joined = joined.sort_values(
        ["position", "minutes"], key=lambda s: s.map(order) if s.name == "position" else -s
    )
    cols = [
        "name",
        "position",
        "detailed_position",
        "age",
        "value_m_eur",
        "contract",
        "minutes",
        *us_cols,
        *PER90_METRICS["fpl"],
    ]
    return joined[[c for c in cols if c in joined]].reset_index(drop=True)
