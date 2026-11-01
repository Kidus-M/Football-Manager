"""The recruitment pool: every rated player with market context.

Each player is ranked against his own league's position-group peers (a
90th-percentile Ligue 1 winger is judged against Ligue 1 wingers; league
strength is applied later, in ranking). A player who played in several
leagues over the window keeps the league where he played most.
"""

from datetime import date
from functools import cache
from pathlib import Path

import duckdb
import pandas as pd
import yaml

from recruit_intel import CONFIG_DIR
from recruit_intel.analysis.squad import market_snapshot
from recruit_intel.features.per90 import per90_table
from recruit_intel.features.percentiles import percentiles


@cache
def league_strength(path: Path = CONFIG_DIR / "leagues.yaml") -> dict[str, float]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _main_league(df: pd.DataFrame) -> pd.DataFrame:
    """Keep each player's row from the league he played most in."""
    return df.sort_values("minutes", ascending=False).drop_duplicates("player_id")


def with_market(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> pd.DataFrame:
    """Add club, age, value_eur and contract_year (Transfermarkt) by player_id."""
    ids = df.player_id.tolist()
    market = market_snapshot(con, ids).drop(columns="detailed_position")
    info = con.execute(
        """
        SELECT pl.player_id, pl.birth_date AS _born, any_value(p.team_name) AS club
        FROM players pl
        LEFT JOIN player_source_ids ids
          ON ids.player_id = pl.player_id AND ids.source = 'transfermarkt'
        LEFT JOIN source_players p USING (source, source_player_id)
        WHERE pl.player_id IN (SELECT unnest(?))
        GROUP BY ALL
        """,
        [ids],
    ).df()
    out = df.merge(info, on="player_id", how="left").merge(market, on="player_id", how="left")
    born = pd.to_datetime(out.pop("_born"))
    out["age"] = ((pd.Timestamp(date.today()) - born).dt.days // 365.25).astype("Int64")
    out["contract_year"] = pd.to_datetime(out.pop("contract_end")).dt.year.astype("Int64")
    out["value_eur"] = pd.to_numeric(out["value_eur"])
    return out.reset_index(drop=True)


def candidate_pool(
    con: duckdb.DuckDBPyConnection, seasons: list[int], competitions: list[str] | None = None
) -> pd.DataFrame:
    """Percentile rows plus club, age, value_eur, contract_year per player."""
    table = per90_table(con, seasons)
    comps = competitions or sorted(table.competition_id.unique())
    parts = [percentiles(table, comp) for comp in comps if (table.competition_id == comp).any()]
    return with_market(con, _main_league(pd.concat(parts, ignore_index=True)))


def per90_pool(con: duckdb.DuckDBPyConnection, seasons: list[int]) -> pd.DataFrame:
    """Raw per-90 rows (not percentiles) plus market context, one per player."""
    table = per90_table(con, seasons).drop(columns="birth_date")
    return with_market(con, _main_league(table))
