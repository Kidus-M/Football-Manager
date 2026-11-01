"""Training data shared by the models.

All models use the Understat metrics only: they exist for every player in
the top five leagues, whereas FPL's defensive counts exist only for the
Premier League and would make the leagues incomparable.
"""

from datetime import date

import duckdb
import numpy as np
import pandas as pd

from recruit_intel.features.per90 import per90_table

FEATURES = ["npxg", "xa", "shots", "key_passes", "xg_chain", "xg_buildup", "npg", "assists"]
MIN_MINUTES = 900


def seasons_available(con: duckdb.DuckDBPyConnection) -> list[int]:
    rows = con.execute(
        "SELECT DISTINCT season FROM player_season_stats WHERE source = 'understat' ORDER BY 1"
    ).fetchall()
    return [r[0] for r in rows]


def player_seasons(con: duckdb.DuckDBPyConnection, min_minutes: int = MIN_MINUTES) -> pd.DataFrame:
    """One row per player, season and competition with enough minutes:
    identity, group, minutes, age at the season's start, FEATURES per 90."""
    parts = []
    for season in seasons_available(con):
        t = per90_table(con, season)
        t = t[t.minutes >= min_minutes].dropna(subset=FEATURES)
        parts.append(t.assign(season=season))
    df = pd.concat(parts, ignore_index=True)
    born = pd.to_datetime(df["birth_date"])
    season_start = pd.to_datetime(df["season"].astype(str) + "-08-01")
    df["age"] = (season_start - born).dt.days / 365.25
    cols = ["player_id", "name", "group", "competition_id", "season", "minutes", "age"]
    return df[cols + FEATURES].reset_index(drop=True)


def transform(x: pd.DataFrame) -> np.ndarray:
    """Per-90 counts are right-skewed; log1p evens them out before scaling."""
    return np.log1p(x[FEATURES].clip(lower=0).to_numpy(dtype=np.float32))


def age_today(birth_date) -> float | None:
    if pd.isna(birth_date):
        return None
    return (pd.Timestamp(date.today()) - pd.Timestamp(birth_date)).days / 365.25
