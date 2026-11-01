"""Role discovery: clusters of players who contribute alike.

A Gaussian mixture over the player embeddings, with the number of clusters
chosen by BIC rather than by hand. Each cluster is described by the per-90
features that most set it apart from the average player and by the listed
positions of its members, so the clusters read as playing roles ("high-
volume shooter", "deep build-up player") that can cut across positions.
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from recruit_intel.features.metrics import METRICS
from recruit_intel.models.dataset import FEATURES, transform

K_RANGE = range(4, 13)


def fit_mixture(vectors: np.ndarray, k_range=K_RANGE, seed: int = 7) -> GaussianMixture:
    best, best_bic = None, np.inf
    for k in k_range:
        gm = GaussianMixture(k, covariance_type="full", n_init=3, random_state=seed)
        gm.fit(vectors)
        bic = gm.bic(vectors)
        if bic < best_bic:
            best, best_bic = gm, bic
    return best


def describe(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """One row per cluster: size, top positions, defining features (z-scores)."""
    x = transform(df)
    z = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)
    rows = []
    for c in sorted(set(labels)):
        members = labels == c
        mean_z = z[members].mean(axis=0)
        order = np.argsort(mean_z)
        high = [f"{METRICS[FEATURES[i]].label} +{mean_z[i]:.1f}" for i in order[::-1][:2]
                if mean_z[i] > 0.3]  # fmt: skip
        low = [f"{METRICS[FEATURES[i]].label} {mean_z[i]:.1f}" for i in order[:1]
               if mean_z[i] < -0.3]  # fmt: skip
        groups = df.group[members].value_counts(normalize=True).head(3)
        rows.append(
            {
                "cluster": int(c),
                "players": int(members.sum()),
                "positions": ", ".join(f"{g} {p:.0%}" for g, p in groups.items()),
                "defined_by": ", ".join(high + low) or "average profile",
                "examples": ", ".join(
                    df.name[members].iloc[np.argsort(-df.minutes[members].to_numpy())][:4]
                ),
            }
        )
    return pd.DataFrame(rows)
