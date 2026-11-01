"""ClubElo adapter: team strength ratings, used for league/opponent adjustment.

The API returns a CSV snapshot of every rated club for a date:
Rank,Club,Country,Level,Elo,From,To
"""

import io
from datetime import date

import pandas as pd

from recruit_intel.ingest.http import CachedClient

SOURCE = "clubelo"
BASE = "http://api.clubelo.com"


def parse(csv_text: str, snapshot: date) -> pd.DataFrame:
    raw = pd.read_csv(io.StringIO(csv_text))
    if raw.empty:
        raise ValueError("ClubElo returned an empty snapshot")
    return pd.DataFrame(
        {
            "clubelo_name": raw["Club"],
            "country": raw["Country"],
            "level": raw["Level"].astype("Int64"),
            "elo": raw["Elo"].astype(float),
            "snapshot_date": snapshot,
        }
    )


def ingest(snapshot: date | None = None, client: CachedClient | None = None) -> pd.DataFrame:
    snapshot = snapshot or date.today()
    own = client is None
    client = client or CachedClient(SOURCE)
    try:
        return parse(client.get_text(f"{BASE}/{snapshot.isoformat()}"), snapshot)
    finally:
        if own:
            client.close()
