"""'Players like X': nearest neighbours in embedding space, with filters."""

import pandas as pd

from recruit_intel.models.dataset import EMBED_FEATURES
from recruit_intel.models.embeddings import Embedder, nearest


def similar_players(
    pool: pd.DataFrame,
    target_id: str,
    embedder: Embedder,
    k: int = 10,
    same_group: bool = False,
    younger: bool = False,
    cheaper: bool = False,
    exclude: set[str] | None = None,
    min_minutes: int = 900,
) -> pd.DataFrame:
    """pool: one row per player with EMBED_FEATURES (per 90), group, age, value_eur.
    Returns the k most similar players with a `distance` column."""
    pool = pool.dropna(subset=EMBED_FEATURES).reset_index(drop=True)
    hits = pool.index[pool.player_id == target_id]
    if len(hits) == 0:
        raise LookupError("target has no qualifying per-90 row in the pool")
    t = int(hits[0])
    target = pool.loc[t]

    mask = pool.minutes >= min_minutes  # tiny samples make noisy neighbours
    if same_group:
        mask &= pool.group == target.group
    if younger and pd.notna(target.age):
        mask &= pool.age < target.age
    if cheaper and pd.notna(target.value_eur):
        mask &= pool.value_eur < target.value_eur
    if exclude:
        mask &= ~pool.player_id.isin(exclude)

    vectors = embedder.embed(pool)
    # Unknown age or value fails a "younger"/"cheaper" filter rather than passing it.
    found = nearest(vectors, t, k, mask.fillna(False).to_numpy(dtype=bool))
    out = pool.loc[[i for i, _ in found]].copy()
    out["distance"] = [d for _, d in found]
    return out
