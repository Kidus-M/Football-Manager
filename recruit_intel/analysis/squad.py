"""Current-squad table for the configured club.

The squad is the club's FPL roster (it lists every registered player). Each
player's Understat stats and Transfermarkt market data are joined through
the canonical player_id from identity matching, so they follow the player
rather than the club name.
"""

from datetime import date

import duckdb
import pandas as pd

from recruit_intel.config import ClubConfig

PER90_METRICS = {
    "fpl": ["tackles", "cbi", "recoveries", "saves"],
    "understat": ["npxg", "xa", "xg_chain", "xg_buildup", "shots", "key_passes"],
}


def _per90(con, source: str, season: int, player_ids: list[str]) -> pd.DataFrame:
    """Per-90 metrics by canonical player, summed across competitions played."""
    rows = con.execute(
        """
        SELECT ids.player_id, s.competition_id, s.minutes, s.metric, s.value
        FROM player_season_stats s
        JOIN player_source_ids ids USING (source, source_player_id)
        WHERE s.source = ? AND s.season = ? AND s.metric IN (SELECT unnest(?))
          AND ids.player_id IN (SELECT unnest(?))
        """,
        [source, season, PER90_METRICS[source], player_ids],
    ).df()
    if rows.empty:
        return pd.DataFrame(columns=["player_id", "minutes", *PER90_METRICS[source]])
    minutes = (
        rows.drop_duplicates(["player_id", "competition_id"]).groupby("player_id").minutes.sum()
    )
    totals = rows.groupby(["player_id", "metric"]).value.sum().unstack()
    per90 = totals.div(minutes, axis=0).mul(90).where(minutes > 0, axis=0)
    return per90.assign(minutes=minutes).reset_index()


def market_snapshot(con, player_ids: list[str]) -> pd.DataFrame:
    """Latest Transfermarkt snapshot per canonical player."""
    return con.execute(
        """
        SELECT ids.player_id, p.detailed_position, m.value_eur, m.contract_end
        FROM market m
        JOIN source_players p USING (source, source_player_id)
        JOIN player_source_ids ids USING (source, source_player_id)
        WHERE m.source = 'transfermarkt' AND ids.player_id IN (SELECT unnest(?))
        QUALIFY row_number() OVER (PARTITION BY ids.player_id
                                   ORDER BY m.snapshot_date DESC) = 1
        """,
        [player_ids],
    ).df()


def current_squad(con: duckdb.DuckDBPyConnection, club: ClubConfig, season: int) -> pd.DataFrame:
    roster = con.execute(
        """
        SELECT ids.player_id, pl.name, pl.position, pl.birth_date
        FROM source_players p
        JOIN player_source_ids ids USING (source, source_player_id)
        JOIN players pl USING (player_id)
        WHERE p.source = 'fpl' AND p.team_name = ?
        """,
        [str(club.source_name("fpl"))],
    ).df()
    if roster.empty:
        raise LookupError(
            f"no matched FPL players for {club.club}; run `recruit ingest` (or `recruit match`)"
        )
    ids = roster.player_id.tolist()

    fpl = _per90(con, "fpl", season, ids)
    us = _per90(con, "understat", season, ids).drop(columns="minutes")
    squad = (
        roster.merge(fpl, on="player_id", how="left")
        .merge(us, on="player_id", how="left")
        .merge(market_snapshot(con, ids), on="player_id", how="left")
    )

    born = pd.to_datetime(squad["birth_date"])
    squad["age"] = ((pd.Timestamp(date.today()) - born).dt.days // 365.25).astype("Int64")
    squad["value_m_eur"] = pd.to_numeric(squad["value_eur"]) / 1e6
    squad["contract"] = pd.to_datetime(squad["contract_end"]).dt.year.astype("Int64")
    squad["minutes"] = squad["minutes"].fillna(0).astype(int)

    order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    squad = squad.assign(_pos=squad.position.map(order)).sort_values(
        ["_pos", "minutes"], ascending=[True, False]
    )
    cols = [
        "name",
        "position",
        "detailed_position",
        "age",
        "value_m_eur",
        "contract",
        "minutes",
        *PER90_METRICS["understat"],
        *PER90_METRICS["fpl"],
    ]
    return squad.reindex(columns=cols).reset_index(drop=True)
