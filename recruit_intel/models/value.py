"""Market value model: what a player 'should' cost, and who is undervalued.

Gradient boosting on log market value from age, contract length, league,
position group, minutes and per-90 output. Predictions are out-of-fold
(each player is valued by a model that never saw him), so a large negative
residual means "cheaper than comparable players", not memorised noise.

Market value reflects much the model can't see (injuries, wages, agent,
hype), so a bargain flag is a lead to scout, not a verdict.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict

from recruit_intel.features.positions import GROUPS
from recruit_intel.models.dataset import FEATURES

MIN_VALUE_EUR = 500_000  # below this TM values are placeholders, not prices


def _x(df: pd.DataFrame, window_year: int, competitions: list[str]) -> pd.DataFrame:
    x = df[FEATURES].astype(float).copy()
    x["age"] = df["age"].astype(float)
    x["contract_years"] = (df["contract_year"].astype(float) - window_year).clip(lower=0)
    x["minutes"] = df["minutes"].astype(float)
    x["league"] = df["competition_id"].map({c: i for i, c in enumerate(competitions)})
    x["group"] = df["group"].map({g: i for i, g in enumerate(GROUPS)}).astype(float)
    return x


def value_model(pool: pd.DataFrame, window_year: int) -> tuple[pd.DataFrame, dict]:
    """pool: per90_pool rows. Returns valued rows with expected_value_eur and
    value_gap (actual / expected - 1; negative = cheaper than expected)."""
    df = pool.dropna(subset=[*FEATURES, "value_eur", "age", "group"])
    df = df[(df.value_eur >= MIN_VALUE_EUR) & (df.minutes >= 900)].reset_index(drop=True)
    competitions = sorted(df.competition_id.unique())
    x = _x(df, window_year, competitions)
    y = np.log(df.value_eur)
    categorical = [c in ("league", "group") for c in x.columns]
    model = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
        l2_regularization=1.0, categorical_features=categorical, random_state=7,
    )  # fmt: skip
    pred = cross_val_predict(model, x, y, cv=KFold(5, shuffle=True, random_state=7))

    out = df.assign(expected_value_eur=np.exp(pred))
    out["value_gap"] = out.value_eur / out.expected_value_eur - 1
    ratio = np.exp(np.abs(y - pred))
    metrics = {
        "players": len(df),
        "r2_log_value": round(r2_score(y, pred), 3),
        "median_error": f"x{np.median(ratio):.2f}",  # typical over/under-estimate factor
    }
    return out, metrics


def bargains(valued: pd.DataFrame, limit: int = 15, **filters) -> pd.DataFrame:
    """Most undervalued players after optional filters: group, max_age,
    max_value_eur, competitions, exclude."""
    df = valued
    if g := filters.get("group"):
        df = df[df.group == g]
    if (a := filters.get("max_age")) is not None:
        df = df[df.age <= a]
    if (v := filters.get("max_value_eur")) is not None:
        df = df[df.value_eur <= v]
    if comps := filters.get("competitions"):
        df = df[df.competition_id.isin(comps)]
    if ex := filters.get("exclude"):
        df = df[~df.player_id.isin(ex)]
    return df.sort_values("value_gap").head(limit)
