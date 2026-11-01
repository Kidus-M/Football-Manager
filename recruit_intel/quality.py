"""Data-quality audit: find problems in the database, and fix the safe ones.

Each check counts offending rows and shows a few examples. Checks with a
`fix` repair only what is unambiguous: rows pointing at nothing, values
that can't be real, placeholders that would otherwise read as data.
Problems that need judgement (two canonical players that may be one
person, contracts that may have run out) are reported, never changed.

`recruit audit` reports; `recruit audit --fix` applies the fixes and then
reports again.
"""

from dataclasses import dataclass

import duckdb
import pandas as pd

MAX_SEASON_MINUTES = 4200  # 38 matches x 90 + stoppage time, with room to spare


@dataclass(frozen=True)
class Check:
    name: str
    what: str
    count_sql: str  # SELECT ... returning the offending rows
    fix_sql: str | None = None  # statement repairing them; None = report only


CHECKS = [
    Check(
        "orphan source ids",
        "identity links to a source row that no longer exists",
        """SELECT i.* FROM player_source_ids i
           ANTI JOIN source_players p USING (source, source_player_id)""",
        """DELETE FROM player_source_ids WHERE (source, source_player_id) NOT IN
           (SELECT source, source_player_id FROM source_players)""",
    ),
    Check(
        "unlinked canonical players",
        "canonical players no source row links to",
        "SELECT pl.* FROM players pl ANTI JOIN player_source_ids i USING (player_id)",
        "DELETE FROM players WHERE player_id NOT IN (SELECT player_id FROM player_source_ids)",
    ),
    Check(
        "stats without a player",
        "season stats for a source player that isn't stored",
        """SELECT s.* FROM player_season_stats s
           ANTI JOIN source_players p USING (source, source_player_id)""",
        """DELETE FROM player_season_stats WHERE (source, source_player_id) NOT IN
           (SELECT source, source_player_id FROM source_players)""",
    ),
    Check(
        "negative stat values",
        "season totals below zero (impossible for counts)",
        "SELECT * FROM player_season_stats WHERE value < 0",
        "DELETE FROM player_season_stats WHERE value < 0",
    ),
    Check(
        "impossible minutes",
        f"minutes below zero or above {MAX_SEASON_MINUTES} in one league-season",
        f"SELECT * FROM player_season_stats WHERE minutes < 0 OR minutes > {MAX_SEASON_MINUTES}",
        f"""DELETE FROM player_season_stats
            WHERE minutes < 0 OR minutes > {MAX_SEASON_MINUTES}""",
    ),
    Check(
        "output without minutes",
        "non-zero totals for a player who played no minutes",
        """SELECT * FROM player_season_stats
           WHERE minutes = 0 AND value > 0 AND metric <> 'games'""",
        """DELETE FROM player_season_stats
           WHERE minutes = 0 AND value > 0 AND metric <> 'games'""",
    ),
    Check(
        "placeholder market values",
        "a market value of 0 is Transfermarkt's 'not valued', not a price",
        "SELECT * FROM market WHERE value_eur = 0",
        "UPDATE market SET value_eur = NULL WHERE value_eur = 0",
    ),
    Check(
        "impossible birth dates",
        "birth dates in the future or implying an age under 14 or over 45",
        """SELECT * FROM source_players WHERE birth_date IS NOT NULL AND
           (birth_date > current_date - INTERVAL 14 YEAR
            OR birth_date < current_date - INTERVAL 45 YEAR)""",
        """UPDATE source_players SET birth_date = NULL WHERE birth_date IS NOT NULL AND
           (birth_date > current_date - INTERVAL 14 YEAR
            OR birth_date < current_date - INTERVAL 45 YEAR)""",
    ),
    Check(
        "matches without pressing data",
        "team matches with no defensive actions recorded (PPDA undefined)",
        "SELECT * FROM team_matches WHERE ppda_actions = 0 OR oppda_actions = 0",
        """UPDATE team_matches SET
             ppda_passes = CASE WHEN ppda_actions = 0 THEN NULL ELSE ppda_passes END,
             ppda_actions = CASE WHEN ppda_actions = 0 THEN NULL ELSE ppda_actions END,
             oppda_passes = CASE WHEN oppda_actions = 0 THEN NULL ELSE oppda_passes END,
             oppda_actions = CASE WHEN oppda_actions = 0 THEN NULL ELSE oppda_actions END
           WHERE ppda_actions = 0 OR oppda_actions = 0""",
    ),
    Check(
        "expired contracts",
        "contract end before the snapshot it came from (report only: may be a renewal "
        "Transfermarkt hasn't recorded yet)",
        """SELECT m.*, p.name FROM market m JOIN source_players p USING (source, source_player_id)
           WHERE m.contract_end < m.snapshot_date""",
    ),
    Check(
        "possible duplicate players",
        "two canonical players with the same name and birth date (report only: needs a "
        "matching fix, not a delete)",
        """SELECT name, birth_date, list(player_id) AS player_ids FROM players
           WHERE birth_date IS NOT NULL GROUP BY name, birth_date HAVING count(*) > 1""",
    ),
    Check(
        "team strength ratings",
        "no ClubElo snapshot stored (report only: the ClubElo API has been down)",
        "SELECT 1 AS missing WHERE (SELECT count(*) FROM team_elo) = 0",
    ),
]


def audit(con: duckdb.DuckDBPyConnection, examples: int = 3) -> pd.DataFrame:
    rows = []
    for check in CHECKS:
        found = con.execute(check.count_sql).df()
        sample = found.head(examples).to_dict("records") if len(found) else []
        rows.append({
            "check": check.name, "rows": len(found), "fixable": check.fix_sql is not None,
            "what": check.what, "examples": sample,
        })  # fmt: skip
    return pd.DataFrame(rows)


def fix(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Apply every safe fix; returns rows affected per check. Runs in one
    transaction, in CHECKS order (orphans first, so later checks see them gone)."""
    affected = {}
    con.execute("BEGIN TRANSACTION")
    try:
        for check in CHECKS:
            if check.fix_sql is None:
                continue
            n = len(con.execute(check.count_sql).df())
            if n:
                con.execute(check.fix_sql)
            affected[check.name] = n
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return affected
