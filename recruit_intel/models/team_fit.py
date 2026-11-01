"""Player-team fit: does a team's style change what a player produces?

A two-tower network (the spec's design). The player tower reads a player's
per-90 profile, age and current team context (strength-adjusted style +
strength); the team tower reads a destination team's style and strength.
The head predicts the change in the player's six per-90 features next season
(log scale), from consecutive player-seasons: stays and moves both, since a
club's style changes too (new manager, new players).

The question it must answer honestly is whether *style* adds anything
beyond strength. So an ablation with the destination's style blanked
(strength kept) is trained identically, and both are scored on held-out
next seasons. The fit score is only meaningful if the full model beats the
ablation there; `validate` reports both.

Fit score for a player at a team: predicted change in his attacking
involvement (mean of npxG, xA and xGChain, log scale), as a percentage.
"""

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd
import torch
from torch import nn

from recruit_intel.models.dataset import EMBED_FEATURES, transform
from recruit_intel.models.team_style import STYLE_FEATURES, StyleModel, team_seasons

FIT_FEATURES = ["npxg", "xa", "xg_chain"]  # what the fit score summarises
MIN_MINUTES = 900
SEED = 7


def player_team_seasons(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Understat player-seasons at a single team (no mid-season movers):
    source_player_id, season, team_id, minutes, age, EMBED_FEATURES per 90."""
    rows = con.execute(
        """
        WITH s AS (
            SELECT source_player_id, season, competition_id, team_name,
                   any_value(minutes) AS minutes,
                   map_from_entries(list(struct_pack(k := metric, v := value))) AS m
            FROM player_season_stats
            WHERE source = 'understat' AND team_name NOT LIKE '%,%'
            GROUP BY ALL
        ), teams AS (
            SELECT DISTINCT team_id, team_name, competition_id, season FROM team_matches
        )
        SELECT s.source_player_id, s.season, t.team_id, s.minutes,
               s.m['npxg'] AS npxg, s.m['xa'] AS xa, s.m['shots'] AS shots,
               s.m['key_passes'] AS key_passes, s.m['xg_chain'] AS xg_chain,
               s.m['xg_buildup'] AS xg_buildup, pl.birth_date
        FROM s
        JOIN teams t USING (team_name, competition_id, season)
        LEFT JOIN player_source_ids ids
          ON ids.source = 'understat' AND ids.source_player_id = s.source_player_id
        LEFT JOIN players pl ON pl.player_id = ids.player_id
        WHERE s.minutes >= ?
        """,
        [MIN_MINUTES],
    ).df()
    for f in EMBED_FEATURES:
        rows[f] = rows[f] / rows.minutes * 90
    season_start = pd.to_datetime(rows.season.astype(str) + "-08-01")
    rows["age"] = (season_start - pd.to_datetime(rows.pop("birth_date"))).dt.days / 365.25
    return rows


def team_context(con: duckdb.DuckDBPyConnection, style: StyleModel) -> pd.DataFrame:
    """team_id, season, strength and the strength-adjusted style vector."""
    ts = team_seasons(con)
    out = ts[["team_id", "season", "xpts_pm"]].copy()
    out[STYLE_FEATURES] = style.adjusted(ts)
    return out


def _prefixed(teams: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Team context columns renamed with a prefix: team style has npxg/npxga
    too, which must never collide with the player's own per-90 columns."""
    cols = {c: f"{prefix}{c}" for c in STYLE_FEATURES} | {"xpts_pm": f"{prefix}strength"}
    return teams.rename(columns=cols)


def make_pairs(players: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Season t at one team joined to season t+1, with both teams' context."""
    nxt = players.assign(season=players.season - 1)
    pairs = players.merge(nxt, on=["source_player_id", "season"], suffixes=("", "_next"))
    pairs = pairs.merge(_prefixed(teams, "ctx_"), on=["team_id", "season"])
    dest = _prefixed(teams, "dest_").rename(columns={"team_id": "team_id_next"})
    pairs = pairs.merge(dest.assign(season=dest.season - 1), on=["team_id_next", "season"])
    pairs["moved"] = pairs.team_id != pairs.team_id_next
    return pairs.reset_index(drop=True)


class TwoTower(nn.Module):
    def __init__(self, n_player: int, n_team: int, hidden: int = 16, n_out: int = 6) -> None:
        super().__init__()
        self.player = nn.Sequential(nn.Linear(n_player, 32), nn.GELU(), nn.Linear(32, hidden))
        self.team = nn.Sequential(nn.Linear(n_team, 16), nn.GELU(), nn.Linear(16, hidden))
        self.head = nn.Sequential(nn.Linear(3 * hidden, 32), nn.GELU(), nn.Linear(32, n_out))

    def forward(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        a, b = self.player(p), self.team(t)
        return self.head(torch.cat([a, b, a * b], dim=1))


CTX_COLS = ["ctx_strength", *[f"ctx_{f}" for f in STYLE_FEATURES]]
TEAM_COLS = ["dest_strength", *[f"dest_{f}" for f in STYLE_FEATURES]]


@dataclass
class FitModel:
    net: TwoTower
    p_mean: np.ndarray
    p_std: np.ndarray
    t_mean: np.ndarray
    t_std: np.ndarray
    use_style: bool = True

    def _player_x(self, df: pd.DataFrame) -> np.ndarray:
        age = df["age"].fillna(26.0).to_numpy(float)[:, None]
        ctx = df[CTX_COLS].to_numpy(float)
        return np.hstack([transform(df), age, ctx])

    def _team_x(self, df: pd.DataFrame) -> np.ndarray:
        t = df[TEAM_COLS].to_numpy(float).copy()
        if not self.use_style:
            t[:, 1:] = 0.0  # keep strength, blank the style
        return t

    def _tensors(self, df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        p = (self._player_x(df) - self.p_mean) / self.p_std
        t = (self._team_x(df) - self.t_mean) / self.t_std
        return torch.tensor(p, dtype=torch.float32), torch.tensor(t, dtype=torch.float32)

    def predict_delta(self, df: pd.DataFrame) -> np.ndarray:
        with torch.no_grad():
            return self.net(*self._tensors(df)).numpy()

    def fit_score(self, df: pd.DataFrame) -> np.ndarray:
        """Predicted % change in attacking involvement at the destination.

        Deltas are in log1p space, so they're mapped back to per-90 values
        before comparing (exp(delta) - 1 would understate small rates).
        """
        idx = [EMBED_FEATURES.index(f) for f in FIT_FEATURES]
        now = transform(df)[:, idx]
        nxt = np.expm1(now + self.predict_delta(df)[:, idx])
        return (nxt.sum(axis=1) / np.expm1(now).sum(axis=1) - 1) * 100


def _targets(df: pd.DataFrame) -> np.ndarray:
    nxt = df[[f"{f}_next" for f in EMBED_FEATURES]].set_axis(EMBED_FEATURES, axis=1)
    return transform(nxt) - transform(df)


def train(pairs: pd.DataFrame, use_style: bool = True, epochs: int = 150) -> FitModel:
    torch.manual_seed(SEED)
    probe = FitModel(None, 0, 1, 0, 1, use_style)  # type: ignore[arg-type]
    px, tx = probe._player_x(pairs), probe._team_x(pairs)
    model = FitModel(
        TwoTower(px.shape[1], tx.shape[1]),
        px.mean(axis=0), px.std(axis=0) + 1e-6, tx.mean(axis=0), tx.std(axis=0) + 1e-6,
        use_style,
    )  # fmt: skip
    p, t = model._tensors(pairs)
    y = torch.tensor(_targets(pairs), dtype=torch.float32)
    opt = torch.optim.Adam(model.net.parameters(), lr=2e-3, weight_decay=1e-4)
    model.net.train()
    for _ in range(epochs):
        for idx in torch.randperm(len(y)).split(256):
            opt.zero_grad()
            loss = nn.functional.mse_loss(model.net(p[idx], t[idx]), y[idx])
            loss.backward()
            opt.step()
    model.net.eval()
    return model


def save(model: FitModel, report: dict) -> None:
    import json

    import joblib

    from recruit_intel.models.embeddings import MODEL_DIR

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "team_fit.joblib")
    (MODEL_DIR / "team_fit_validation.json").write_text(json.dumps(report, indent=2))


def load() -> tuple[FitModel, dict]:
    import json

    import joblib

    from recruit_intel.models.embeddings import MODEL_DIR

    path = MODEL_DIR / "team_fit.joblib"
    if not path.exists():
        raise FileNotFoundError("no trained fit model; run `recruit train`")
    # Safe to unpickle: only ever written by `save` above into the local,
    # gitignored data/ directory.
    report = json.loads((MODEL_DIR / "team_fit_validation.json").read_text())
    return joblib.load(path), report


def style_adds_skill(report: dict) -> bool:
    """Did the style model beat the strength-only ablation on held-out movers?"""
    movers = report["movers"]
    return movers["with style"] < movers["strength only"]


def fit_for_destination(
    con: duckdb.DuckDBPyConnection,
    model: FitModel,
    style: StyleModel,
    player_ids: list[str],
    dest_team: str,
) -> pd.Series:
    """Fit score (% change in attacking involvement) of canonical players
    moving to `dest_team` (Understat name) as it plays this season. Each
    player is described by his latest qualifying season at a single club."""
    teams = team_context(con, style)
    names = con.execute("SELECT DISTINCT team_id, team_name, season FROM team_matches").df()
    dest_rows = names[names.team_name == dest_team].merge(teams, on=["team_id", "season"])
    if dest_rows.empty:
        raise LookupError(f"no style data for {dest_team}")
    dest = _prefixed(dest_rows.sort_values("season").tail(1), "dest_")

    ids = con.execute(
        """
        SELECT player_id, source_player_id FROM player_source_ids
        WHERE source = 'understat' AND player_id IN (SELECT unnest(?))
        """,
        [player_ids],
    ).df()
    seasons = player_team_seasons(con).merge(ids, on="source_player_id")
    latest = seasons.sort_values("season").drop_duplicates("player_id", keep="last")
    rows = latest.merge(_prefixed(teams, "ctx_"), on=["team_id", "season"])
    for col in TEAM_COLS:
        rows[col] = dest[col].iloc[0]
    if rows.empty:
        return pd.Series(dtype=float)
    return pd.Series(model.fit_score(rows), index=rows.player_id.to_numpy())


def validate(pairs: pd.DataFrame) -> dict:
    """Train on earlier seasons, test on the last complete season's pairs."""
    test_season = pairs.season.max()
    train_p, test_p = pairs[pairs.season < test_season], pairs[pairs.season == test_season]
    y = _targets(test_p)
    out = {"train_pairs": len(train_p), "test_pairs": len(test_p),
           "test_moves": int(test_p.moved.sum())}  # fmt: skip
    models = {"with style": train(train_p, True), "strength only": train(train_p, False)}
    for label, mask in (("all", np.ones(len(test_p), bool)), ("movers", test_p.moved.to_numpy())):
        res = {"no change": float((y[mask] ** 2).mean())}
        for name, m in models.items():
            res[name] = float(((m.predict_delta(test_p[mask]) - y[mask]) ** 2).mean())
        out[label] = res
    return out
