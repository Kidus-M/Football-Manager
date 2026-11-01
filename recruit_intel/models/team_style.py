"""Team style: how a team plays, as one vector per team-season.

Built from Understat's per-match team stats plus the team's players' season
totals. Each feature is a rate or share, so it says how a team plays rather
than only how good it is (xG for and against are kept: chance creation and
defensive risk are part of a style, as the spec asks):

- ppda: opponent passes per defensive action. Lower = more pressing.
- oppda: the same when opponents defend against us. Lower = we get pressed
  hard, typically because we try to play out.
- territory: our share of the match's deep completions.
- xg_per_shot: shot quality: patient chance-making vs speculative shooting.
- buildup_share: share of our xGChain that comes from build-up involvement
  (players before the last two actions): possession-heavy sides score high.
- npxg / npxga per match: chance creation and defensive risk.

Possession and formations aren't in the free data we can reach reliably
(FotMob blocks scrapers), so the vector describes intent and outcomes, not
shape.

An autoencoder compresses the vectors to 2 dimensions (a style map), and a
Gaussian mixture groups them into named styles.
"""

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd
import torch
from sklearn.mixture import GaussianMixture

from recruit_intel.models.embeddings import AutoEncoder, fit_autoencoder

STYLE_FEATURES = ["ppda", "oppda", "territory", "xg_per_shot", "buildup_share", "npxg", "npxga"]
LABELS = {
    "ppda": ("low press", "high press"),  # (high value, low value): lower ppda = more press
    "oppda": ("rarely pressed", "pressed high"),
    "territory": ("dominant territory", "deep-lying"),
    "xg_per_shot": ("high-quality chances", "speculative shooting"),
    "buildup_share": ("patient build-up", "direct"),
    "npxg": ("high chance creation", "low chance creation"),
    "npxga": ("open defensively", "tight defensively"),
}
MIN_MATCHES = 10


def team_seasons(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """One row per team-season: team_id, team_name, competition_id, season,
    matches, xpts_pm, then STYLE_FEATURES."""
    teams = con.execute(
        """
        SELECT team_id, any_value(team_name) AS team_name, competition_id, season,
               count(*) AS matches,
               sum(ppda_passes) / nullif(sum(ppda_actions), 0) AS ppda,
               sum(oppda_passes) / nullif(sum(oppda_actions), 0) AS oppda,
               sum(deep) / nullif(sum(deep) + sum(deep_allowed), 0) AS territory,
               avg(npxg) AS npxg, avg(npxga) AS npxga, avg(xpts) AS xpts_pm
        FROM team_matches
        GROUP BY team_id, competition_id, season
        """
    ).df()
    # Shots and xGChain/xGBuildup come from the team's players. Players who
    # moved mid-season ("Old,New" team names) are left out of the sums.
    from_players = con.execute(
        """
        SELECT team_name, competition_id, season,
               sum(value) FILTER (WHERE metric = 'shots') AS shots,
               sum(value) FILTER (WHERE metric = 'npxg') AS player_npxg,
               sum(value) FILTER (WHERE metric = 'xg_buildup') AS buildup,
               sum(value) FILTER (WHERE metric = 'xg_chain') AS chain
        FROM player_season_stats
        WHERE source = 'understat' AND team_name NOT LIKE '%,%'
        GROUP BY ALL
        """
    ).df()
    df = teams.merge(from_players, on=["team_name", "competition_id", "season"], how="left")
    df["xg_per_shot"] = df.player_npxg / df.shots
    df["buildup_share"] = df.buildup / df.chain
    df = df[df.matches >= MIN_MATCHES].dropna(subset=STYLE_FEATURES)
    cols = ["team_id", "team_name", "competition_id", "season", "matches", "xpts_pm"]
    return df[cols + STYLE_FEATURES].reset_index(drop=True)


@dataclass
class StyleModel:
    mean: np.ndarray
    std: np.ndarray
    encoder: AutoEncoder
    mixture: GaussianMixture
    names: dict[int, str]

    def scale(self, df: pd.DataFrame) -> np.ndarray:
        return ((df[STYLE_FEATURES].to_numpy(np.float32) - self.mean) / self.std).astype(
            np.float32
        )

    def embed(self, df: pd.DataFrame) -> np.ndarray:
        with torch.no_grad():
            return self.encoder.encoder(torch.from_numpy(self.scale(df))).numpy()

    def style(self, df: pd.DataFrame) -> list[str]:
        return [self.names[c] for c in self.mixture.predict(self.scale(df))]


def name_cluster(mean_z: np.ndarray) -> str:
    """Two strongest traits, e.g. 'high press, dominant territory'."""
    order = np.argsort(-np.abs(mean_z))
    traits = []
    for i in order[:2]:
        if abs(mean_z[i]) < 0.4:
            break
        high, low = LABELS[STYLE_FEATURES[i]]
        traits.append(high if mean_z[i] > 0 else low)
    return ", ".join(traits) or "balanced"


def fit(df: pd.DataFrame, epochs: int = 400, k_range=range(3, 9)) -> tuple[StyleModel, dict]:
    x = df[STYLE_FEATURES].to_numpy(np.float32)
    mean, std = x.mean(axis=0), x.std(axis=0) + 1e-6
    z = ((x - mean) / std).astype(np.float32)

    ae = AutoEncoder(len(STYLE_FEATURES), latent=2)
    rng = np.random.default_rng(7)
    idx = rng.permutation(len(z))
    cut = int(len(z) * 0.85)
    fit_autoencoder(ae, torch.from_numpy(z[idx[:cut]]), epochs, batch=64)
    with torch.no_grad():
        held = torch.from_numpy(z[idx[cut:]])
        test_mse = float(torch.nn.functional.mse_loss(ae(held), held))

    best, best_bic = None, np.inf
    for k in k_range:
        gm = GaussianMixture(k, covariance_type="full", n_init=3, random_state=7).fit(z)
        if (bic := gm.bic(z)) < best_bic:
            best, best_bic = gm, bic
    labels = best.predict(z)
    names = {c: name_cluster(z[labels == c].mean(axis=0)) for c in range(best.n_components)}
    model = StyleModel(mean, std, ae, best, names)
    return model, {"team_seasons": len(df), "ae_test_mse": test_mse, "styles": len(names)}
