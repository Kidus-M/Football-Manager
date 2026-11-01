"""Team style: how a team plays, separated from how good it is.

One vector per team-season, from Understat's per-match team stats plus the
team's players' season totals:

- ppda: opponent passes per defensive action. Lower = more pressing.
- oppda: the same when opponents defend against us. Lower = we get pressed
  hard, typically because we try to play out.
- territory: our share of the match's deep completions.
- xg_per_shot: shot quality: patient chance-making vs speculative shooting.
- buildup_share: share of our xGChain from build-up involvement.
- npxg / npxga per match: chance creation and defensive risk.

Raw, these mostly measure strength: expected points per match explains 81%
of the variance in territory and 77% in npxG, because good teams have the
ball. So each feature is regressed on strength (xpts per match) and the
residual is the style: "presses more than a side of its level does".

Styles turned out to be a continuum, not types: a Gaussian mixture's BIC
is lowest at k=2 on both raw and adjusted features. So instead of forcing
named clusters, each team-season is described by its strongest
strength-adjusted traits, and an autoencoder maps the adjusted features to
2 dimensions (a style map where nearby teams play alike regardless of
quality).

Possession and formations aren't in the free data we can reach reliably
(FotMob blocks scrapers): the vector describes intent and outcomes, not shape.
"""

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd
import torch

from recruit_intel.models.embeddings import AutoEncoder, fit_autoencoder

STYLE_FEATURES = ["ppda", "oppda", "territory", "xg_per_shot", "buildup_share", "npxg", "npxga"]
# (label when the strength-adjusted value is high, label when low)
TRAITS = {
    "ppda": ("presses less than its level", "presses more than its level"),
    "oppda": ("rarely pressed", "invites the press"),
    "territory": ("more territory than its level", "cedes territory"),
    "xg_per_shot": ("patient, high-quality shots", "shoots from distance"),
    "buildup_share": ("patient build-up", "direct"),
    "npxg": ("creates more than its level", "creates less than its level"),
    "npxga": ("open defensively", "tight defensively"),
}
TRAIT_Z = 0.8  # how far from a typical side of its level a trait must be
MIN_MATCHES = 8  # low enough to include the current season after ~2 months


def team_seasons(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """One row per team-season: team_id, team_name, competition_id, season,
    matches, xpts_pm (strength), then STYLE_FEATURES."""
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
    slope: np.ndarray  # per feature, vs strength
    intercept: np.ndarray
    resid_std: np.ndarray
    encoder: AutoEncoder

    def adjusted(self, df: pd.DataFrame) -> np.ndarray:
        """Strength-adjusted style z-scores, one column per STYLE_FEATURE."""
        x = df[STYLE_FEATURES].to_numpy(np.float64)
        expected = self.intercept + np.outer(df["xpts_pm"].to_numpy(), self.slope)
        return ((x - expected) / self.resid_std).astype(np.float32)

    def embed(self, df: pd.DataFrame) -> np.ndarray:
        with torch.no_grad():
            return self.encoder.encoder(torch.from_numpy(self.adjusted(df))).numpy()

    def traits(self, df: pd.DataFrame, max_traits: int = 3) -> list[str]:
        return [describe(z, max_traits) for z in self.adjusted(df)]


def describe(z: np.ndarray, max_traits: int = 3, threshold: float = TRAIT_Z) -> str:
    """Strongest strength-adjusted traits of one style vector, as words."""
    order = np.argsort(-np.abs(z))[:max_traits]
    labels = [TRAITS[STYLE_FEATURES[i]][0 if z[i] > 0 else 1]
              for i in order if abs(z[i]) >= threshold]  # fmt: skip
    return ", ".join(labels) or "typical for its level"


def fit(df: pd.DataFrame, epochs: int = 400) -> tuple[StyleModel, dict]:
    x = df[STYLE_FEATURES].to_numpy(np.float64)
    q = df["xpts_pm"].to_numpy()
    # Per-feature least squares on strength.
    qc = q - q.mean()
    slope = (qc[:, None] * (x - x.mean(axis=0))).sum(axis=0) / (qc**2).sum()
    intercept = x.mean(axis=0) - slope * q.mean()
    resid = x - (intercept + np.outer(q, slope))
    r2 = 1 - resid.var(axis=0) / x.var(axis=0)

    model = StyleModel(slope, intercept, resid.std(axis=0), AutoEncoder(len(STYLE_FEATURES), 2))
    z = model.adjusted(df)
    rng = np.random.default_rng(7)
    idx = rng.permutation(len(z))
    cut = int(len(z) * 0.85)
    fit_autoencoder(model.encoder, torch.from_numpy(z[idx[:cut]]), epochs, batch=64)
    with torch.no_grad():
        held = torch.from_numpy(z[idx[cut:]])
        test_mse = float(torch.nn.functional.mse_loss(model.encoder(held), held))
    metrics = {
        "team_seasons": len(df),
        "strength_r2": dict(zip(STYLE_FEATURES, np.round(r2, 2).tolist(), strict=True)),
        "map_test_mse": round(test_mse, 3),  # 1.0 = no better than the mean
    }
    return model, metrics


MODEL_PATH = "team_style.joblib"


def save(model: StyleModel) -> None:
    import joblib

    from recruit_intel.models.embeddings import MODEL_DIR

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / MODEL_PATH)


def load() -> StyleModel:
    import joblib

    from recruit_intel.models.embeddings import MODEL_DIR

    path = MODEL_DIR / MODEL_PATH
    if not path.exists():
        raise FileNotFoundError("no trained style model; run `recruit train`")
    # Safe to unpickle: only ever written by `save` above into the local,
    # gitignored data/ directory.
    return joblib.load(path)


def nearest_teams(df: pd.DataFrame, model: StyleModel, row: int, k: int = 8) -> pd.DataFrame:
    """Team-seasons that play most alike (style map distance), other teams only."""
    v = model.embed(df)
    d = np.linalg.norm(v - v[row], axis=1)
    out = df.assign(distance=d)
    out = out[out.team_id != df.team_id.iloc[row]]
    return out.sort_values("distance").head(k)
