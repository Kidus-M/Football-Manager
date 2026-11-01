import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from recruit_intel.models.dataset import EMBED_FEATURES  # noqa: E402
from recruit_intel.models.embeddings import Embedder, nearest, train  # noqa: E402
from recruit_intel.models.similarity import similar_players  # noqa: E402


def _synthetic(n_per_type=150, seed=0):
    """Three player types with distinct output shapes."""
    rng = np.random.default_rng(seed)
    types = {
        "shooter": [0.45, 0.08, 3.2, 0.9, 0.6, 0.1],
        "creator": [0.10, 0.30, 1.2, 2.6, 0.7, 0.3],
        "builder": [0.03, 0.05, 0.4, 0.8, 0.5, 0.45],
    }
    rows = []
    for kind, centre in types.items():
        for i in range(n_per_type):
            vals = np.clip(np.array(centre) * rng.normal(1, 0.12, len(centre)), 0, None)
            rows.append(
                {
                    "player_id": f"{kind}{i}",
                    "kind": kind,
                    **dict(zip(EMBED_FEATURES, vals, strict=True)),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def trained():
    df = _synthetic()
    emb, metrics = train(df, epochs=60)
    return df, emb, metrics


def test_autoencoder_beats_mean_baseline(trained):
    _, _, metrics = trained
    assert metrics["test_mse"] < 0.5 * metrics["baseline_mse"]


def test_neighbours_share_the_player_type(trained):
    df, emb, _ = trained
    vectors = emb.embed(df)
    hits = nearest(vectors, 0, k=10)
    assert all(df.kind[i] == "shooter" for i, _ in hits)


def test_save_and_load_roundtrip(trained, tmp_path):
    df, emb, _ = trained
    emb.save(tmp_path)
    again = Embedder.load(tmp_path)
    assert np.allclose(emb.embed(df.head(5)), again.embed(df.head(5)), atol=1e-6)


def test_similar_players_filters(trained):
    df, emb, _ = trained
    pool = df.assign(group="ST", minutes=2000,
                     age=pd.array(np.arange(len(df)) % 30 + 18, dtype="Int64"),
                     value_eur=np.arange(len(df)) * 1e5)  # fmt: skip
    pool.loc[41, "age"] = pd.NA  # unknown age must not pass a "younger" filter
    target = pool.player_id[40]
    out = similar_players(pool, target, emb, k=5, younger=True, cheaper=True)
    t = pool.set_index("player_id").loc[target]
    assert (out.age < t.age).all() and (out.value_eur < t.value_eur).all()
    assert target not in set(out.player_id) and pool.player_id[41] not in set(out.player_id)
