"""DuckDB storage.

Ingestion writes raw, source-keyed rows (source_players, player_season_stats,
market). Identity matching later links source IDs to one canonical `players`
row through player_source_ids. Stats are stored as season totals in long form
(one row per metric) so a new metric or source needs no migration; per-90
values are derived in features/.
"""

from pathlib import Path

import duckdb
import pandas as pd

from recruit_intel import DATA_DIR

DEFAULT_DB = DATA_DIR / "recruit.duckdb"

SCHEMA = """
CREATE TABLE IF NOT EXISTS competitions (
    competition_id TEXT PRIMARY KEY,   -- e.g. 'ENG-Premier League'
    country TEXT,
    name TEXT
);

CREATE TABLE IF NOT EXISTS teams (
    source TEXT NOT NULL,
    source_team_id TEXT NOT NULL,
    name TEXT NOT NULL,
    competition_id TEXT,
    season INTEGER,
    PRIMARY KEY (source, source_team_id, season)
);

CREATE TABLE IF NOT EXISTS team_elo (
    clubelo_name TEXT NOT NULL,
    country TEXT,
    level INTEGER,
    elo DOUBLE,
    snapshot_date DATE NOT NULL,
    PRIMARY KEY (clubelo_name, snapshot_date)
);

CREATE TABLE IF NOT EXISTS source_players (
    source TEXT NOT NULL,
    source_player_id TEXT NOT NULL,
    name TEXT NOT NULL,
    team_name TEXT,
    position TEXT,                     -- GK / DEF / MID / FWD
    birth_date DATE,
    nationality TEXT,
    fetched_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (source, source_player_id)
);

CREATE TABLE IF NOT EXISTS player_season_stats (
    source TEXT NOT NULL,
    source_player_id TEXT NOT NULL,
    season INTEGER NOT NULL,           -- start year: 2025 = 2025-26
    competition_id TEXT NOT NULL,
    team_name TEXT,
    minutes INTEGER,
    metric TEXT NOT NULL,
    value DOUBLE,                      -- season total
    PRIMARY KEY (source, source_player_id, season, competition_id, metric)
);

CREATE TABLE IF NOT EXISTS market (
    source TEXT NOT NULL,
    source_player_id TEXT NOT NULL,
    value_eur BIGINT,
    contract_end DATE,
    wage_estimate_eur BIGINT,
    snapshot_date DATE NOT NULL,
    PRIMARY KEY (source, source_player_id, snapshot_date)
);

CREATE TABLE IF NOT EXISTS team_matches (
    source TEXT NOT NULL,
    team_id TEXT NOT NULL,
    team_name TEXT NOT NULL,
    season INTEGER NOT NULL,
    competition_id TEXT NOT NULL,
    match_date TIMESTAMP NOT NULL,
    home BOOLEAN,
    xg DOUBLE, xga DOUBLE, npxg DOUBLE, npxga DOUBLE,
    ppda_passes INTEGER, ppda_actions INTEGER,      -- opponent passes / our defensive actions
    oppda_passes INTEGER, oppda_actions INTEGER,    -- the same, when the opponent presses us
    deep INTEGER, deep_allowed INTEGER,             -- completions within ~20m of goal
    scored INTEGER, conceded INTEGER,
    xpts DOUBLE, pts INTEGER,
    PRIMARY KEY (source, team_id, match_date)
);

CREATE TABLE IF NOT EXISTS manager_spells (
    source TEXT NOT NULL,
    club_id TEXT NOT NULL,                 -- Transfermarkt club id
    manager_id TEXT NOT NULL,
    name TEXT NOT NULL,
    appointed DATE NOT NULL,
    ended DATE,                            -- NULL while the spell is ongoing
    matches INTEGER,
    ppg DOUBLE,
    PRIMARY KEY (source, club_id, manager_id, appointed)
);

CREATE TABLE IF NOT EXISTS players (
    player_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    birth_date DATE,
    nationality TEXT,
    position TEXT
);

CREATE TABLE IF NOT EXISTS player_source_ids (
    player_id TEXT NOT NULL,
    source TEXT NOT NULL,
    source_player_id TEXT NOT NULL,
    match_score DOUBLE,
    PRIMARY KEY (source, source_player_id)
);
"""

# Columns added after the first release. ADD COLUMN IF NOT EXISTS keeps old
# database files working without a separate migration tool.
MIGRATIONS = """
ALTER TABLE source_players ADD COLUMN IF NOT EXISTS detailed_position TEXT;
ALTER TABLE source_players ADD COLUMN IF NOT EXISTS height_cm INTEGER;
ALTER TABLE source_players ADD COLUMN IF NOT EXISTS foot TEXT;
ALTER TABLE source_players ADD COLUMN IF NOT EXISTS team_id TEXT;
ALTER TABLE market ADD COLUMN IF NOT EXISTS joined_date DATE;
ALTER TABLE player_source_ids ADD COLUMN IF NOT EXISTS method TEXT;
ALTER TABLE players ADD COLUMN IF NOT EXISTS detailed_position TEXT;
"""


def connect(path: Path | str = DEFAULT_DB) -> duckdb.DuckDBPyConnection:
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute(SCHEMA)
    con.execute(MIGRATIONS)
    return con


def upsert(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame) -> int:
    """Insert rows, replacing any with the same primary key."""
    if df.empty:
        return 0
    cols = ", ".join(df.columns)
    con.register("_upsert_df", df)
    try:
        con.execute(f"INSERT OR REPLACE INTO {table} ({cols}) SELECT {cols} FROM _upsert_df")
    finally:
        con.unregister("_upsert_df")
    return len(df)
