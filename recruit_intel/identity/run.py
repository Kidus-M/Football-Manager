"""Rebuild the canonical player tables from whatever the sources have loaded."""

import duckdb
import pandas as pd

from recruit_intel import db
from recruit_intel.identity.matcher import build_identity

TOP5 = ["ENG-Premier League", "ESP-La Liga", "GER-Bundesliga", "ITA-Serie A", "FRA-Ligue 1"]


def load_inputs(con: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    tm = con.execute(
        """
        SELECT p.source_player_id, p.name, p.birth_date, p.nationality, p.position,
               p.detailed_position, p.team_name, t.competition_id
        FROM source_players p
        LEFT JOIN (SELECT source_team_id, arg_max(competition_id, season) AS competition_id
                   FROM teams WHERE source = 'transfermarkt' GROUP BY 1) t
          ON t.source_team_id = p.team_id
        WHERE p.source = 'transfermarkt'
        """
    ).df()
    fpl = con.execute(
        """
        SELECT source_player_id, name, birth_date, position, team_name
        FROM source_players WHERE source = 'fpl'
        """
    ).df()
    # Understat: club and competition both come from the player's latest
    # season row (most minutes on ties), so the two always agree.
    us = con.execute(
        """
        SELECT p.source_player_id, p.name, p.position,
               s.team_name, s.competition_id, s.season, s.minutes
        FROM source_players p
        JOIN (SELECT source_player_id, team_name, competition_id, season, minutes
              FROM player_season_stats
              WHERE source = 'understat' AND metric = 'xg'
              QUALIFY row_number() OVER (PARTITION BY source_player_id
                                         ORDER BY season DESC, minutes DESC) = 1
             ) s USING (source_player_id)
        WHERE p.source = 'understat'
        """
    ).df()
    # Players who moved mid-season are listed as "Old Club,New Club".
    us["team_name"] = us["team_name"].str.split(",").str[-1]
    for df in (tm, fpl):
        df["birth_date"] = pd.to_datetime(df["birth_date"]).dt.date
    af = con.execute(
        """
        SELECT source_player_id, name, birth_date, position, team_name
        FROM source_players WHERE source = 'apifootball'
        """
    ).df()
    af["birth_date"] = pd.to_datetime(af["birth_date"]).dt.date
    return {"tm": tm, "fpl": fpl, "us": us, "af": af}


def report(ids: pd.DataFrame, inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Match rate per source, plus the roadmap's target population: Understat
    players in the top 5 leagues who played this season, linked to Transfermarkt."""
    linked = ids[ids.method != "unmatched"]
    to_tm = ids[ids.player_id.str.startswith("tm-")]
    us = inputs["us"]
    target = us[(us.season == us.season.max()) & us.competition_id.isin(TOP5) & (us.minutes > 0)]
    rows = []
    for label, pop, source, pool in (
        ("FPL players", inputs["fpl"], "fpl", linked),
        ("API-Football players", inputs["af"], "apifootball", linked),
        ("Understat players, all seasons", us, "understat", linked),
        ("Understat top-5 this season -> Transfermarkt", target, "understat", to_tm),
    ):
        hit = pool[pool.source == source].source_player_id.isin(pop.source_player_id).sum()
        rows.append({"population": label, "players": len(pop), "matched": int(hit)})
    out = pd.DataFrame(rows)
    out["rate_%"] = (out.matched / out.players * 100).round(1)
    return out


def rebuild(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    inputs = load_inputs(con)
    tables = build_identity(inputs["tm"], inputs["fpl"], inputs["us"], inputs["af"])
    for table in ("player_source_ids", "players"):
        con.execute(f"DELETE FROM {table}")
        db.upsert(con, table, tables[table])
    return report(tables["player_source_ids"], inputs)
