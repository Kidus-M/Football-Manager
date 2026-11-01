"""League translation: projecting a player's next-season output in a new league.

Trained on consecutive player-seasons (season t -> t+1), both in and across
leagues: same-league pairs teach ageing and regression to the mean, moves
teach what changing league does. One gradient-boosted model per output
metric, with the from- and to-league as categorical features.

Validation holds out the most recent completed season as targets and
compares against the naive "same output next year" baseline, so the
reported skill is skill on unseen seasons, not fit to the training data.

The learned league factors (projected output in the reference league over
projected output staying put) replace the hand-set config/leagues.yaml.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

from recruit_intel import DATA_DIR
from recruit_intel.features.positions import GROUPS
from recruit_intel.models.dataset import FEATURES

MODEL_DIR = DATA_DIR / "models"
TARGETS = ["npxg", "xa", "shots", "key_passes", "xg_chain", "xg_buildup"]
REFERENCE = "ENG-Premier League"
# Output summary used for league factors: how much attacking value a player
# is involved in, per 90.
COMPOSITE = ["xg_chain", "npxg", "xa"]


def make_pairs(seasons: pd.DataFrame, last_complete: int) -> pd.DataFrame:
    """Join each player-season to the same player's next season."""
    nxt = seasons.assign(season=seasons.season - 1)
    pairs = seasons.merge(nxt, on=["player_id", "season"], suffixes=("", "_next"))
    pairs = pairs[pairs.season + 1 <= last_complete]
    pairs = pairs.rename(columns={"competition_id_next": "to_comp"})
    return pairs.rename(columns={"competition_id": "from_comp"})


class Projector:
    def __init__(self, competitions: list[str]) -> None:
        self.competitions = sorted(competitions)
        self.groups = list(GROUPS)
        self.models: dict[str, HistGradientBoostingRegressor] = {}

    def _x(self, df: pd.DataFrame, from_col: str, to_col: str) -> pd.DataFrame:
        comp = {c: i for i, c in enumerate(self.competitions)}
        grp = {g: i for i, g in enumerate(self.groups)}
        x = df[FEATURES].astype(float).copy()
        x["age"] = df["age"]
        x["minutes"] = df["minutes"]
        x["from_comp"] = df[from_col].map(comp).astype(float)
        x["to_comp"] = df[to_col].map(comp).astype(float)
        x["group"] = df["group"].map(grp).astype(float)
        return x

    def fit(self, pairs: pd.DataFrame) -> "Projector":
        x = self._x(pairs, "from_comp", "to_comp")
        categorical = [c in ("from_comp", "to_comp", "group") for c in x.columns]
        for target in TARGETS:
            model = HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                l2_regularization=1.0, categorical_features=categorical, random_state=7,
            )  # fmt: skip
            model.fit(x, pairs[f"{target}_next"])
            self.models[target] = model
        return self

    def predict(self, df: pd.DataFrame, to_comp: str | None = None) -> pd.DataFrame:
        """Next-season per-90 projection. `to_comp` None = staying in his league."""
        df = df.assign(_to=to_comp or df["competition_id"])
        x = self._x(df, "competition_id", "_to")
        return pd.DataFrame(
            {t: np.clip(m.predict(x), 0, None) for t, m in self.models.items()}, index=df.index
        )

    def league_factors(self, recent: pd.DataFrame) -> dict[str, float]:
        """Per league: projected composite output in REFERENCE over staying put,
        averaged over the league's recent qualifying players."""
        factors = {}
        for comp, players in recent.groupby("competition_id"):
            stay = self.predict(players)[COMPOSITE].sum(axis=1).mean()
            move = self.predict(players, REFERENCE)[COMPOSITE].sum(axis=1).mean()
            factors[comp] = round(float(move / stay), 3)
        return factors

    def save(self, directory: Path = MODEL_DIR) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, directory / "projector.joblib")

    @staticmethod
    def load(directory: Path = MODEL_DIR) -> "Projector":
        path = directory / "projector.joblib"
        if not path.exists():
            raise FileNotFoundError("no trained projection model; run `recruit train`")
        return joblib.load(path)


def train_and_validate(seasons: pd.DataFrame) -> tuple[Projector, dict]:
    """Hold out the last complete season's targets; report R2 vs persistence."""
    current = seasons.season.max()
    last_complete = current - 1  # the current season is still being played
    pairs = make_pairs(seasons, last_complete)
    test_mask = pairs.season + 1 == last_complete
    train_pairs, test_pairs = pairs[~test_mask], pairs[test_mask]

    comps = sorted(seasons.competition_id.unique())
    held_out = Projector(comps).fit(train_pairs)
    pred = held_out.predict(test_pairs.rename(columns={"from_comp": "competition_id"}),
                            to_comp=None) if False else None  # fmt: skip
    x_test = test_pairs.rename(columns={"from_comp": "competition_id"})
    report = {"pairs": len(pairs), "train": len(train_pairs), "test": len(test_pairs),
              "moves_in_test": int((test_pairs.from_comp != test_pairs.to_comp).sum()),
              "metrics": {}}  # fmt: skip
    for target in TARGETS:
        x = held_out._x(x_test, "competition_id", "to_comp")
        pred = np.clip(held_out.models[target].predict(x), 0, None)
        truth = test_pairs[f"{target}_next"]
        report["metrics"][target] = {
            "r2_model": round(r2_score(truth, pred), 3),
            "r2_same_as_last_year": round(r2_score(truth, test_pairs[target]), 3),
        }

    final = Projector(comps).fit(pairs)  # all completed seasons for the saved model
    recent = seasons[seasons.season == last_complete]
    report["league_factors"] = final.league_factors(recent)
    return final, report


def save_league_factors(factors: dict[str, float], directory: Path = MODEL_DIR) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "league_factors.json").write_text(json.dumps(factors, indent=2))
