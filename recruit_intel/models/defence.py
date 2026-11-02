"""Defensive estimator: API-Football's defensive metrics for players the
analysis window has no measured numbers for.

API-Football's free plan stops at 2024-25 and Understat has no defensive
stats, so outside the Premier League (FPL) the window has no tackles,
interceptions, blocks, duels or dribbled-past at all. Without them a defender
can't be scored, and simply carrying a two-season-old number forward ignores
a change of club, role or age.

For each metric, gradient-boosted quantile models (10th, 50th, 90th
percentile) predict the possession-adjusted per-90 from:

- the player's Understat profile in the same season (how he plays on the ball)
- his team's context: possession, pressing (PPDA), chances and deep
  completions conceded
- position group, league and age
- his own latest measured API-Football season, and how long ago it was, when
  he has one ("carried" estimates); without one, the model works from the
  rest alone ("estimated")

Until older API-Football seasons are loaded the model has no anchors to learn
from; players with an older season then keep it as-is ("older") and only
players with none are estimated.

Training rows are every measured API-Football player-season, each seen with
every earlier season as its anchor and once with no anchor, so the model
learns all three situations. Validation holds out whole players.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from recruit_intel import DATA_DIR
from recruit_intel.features import history
from recruit_intel.features.context import CONTEXT
from recruit_intel.features.positions import GROUPS

TARGETS = history.METRICS_MEASURED
PROFILE = ["npxg", "xa", "shots", "key_passes", "xg_chain", "xg_buildup"]
LAGS = [f"lag_{m}" for m in TARGETS]
NUMERIC = PROFILE + CONTEXT + ["age"] + LAGS + ["lag_gap"]
CATEGORICAL = ["group", "competition_id"]
QUANTILES = (0.1, 0.5, 0.9)
BAND_COVERAGE = 0.8  # what the 10-90% range should cover
MIN_TRAIN_MINUTES = 600
MAX_LAG_GAP = 3
MIN_GROUP_ROWS = 20  # for within-group rank correlations

PATH = DATA_DIR / "models" / "defence.joblib"
REPORT = DATA_DIR / "models" / "defence.json"
# Until a model is validated: how far a filled value can be trusted to rank
# players the way measured data would (1 = as well as measured data).
# "older" (a season carried as-is) is a guess until a model trained with older
# seasons measures it; the rest are replaced by validation.
DEFAULT_RELIABILITY = {"measured": 1.0, "older": 0.5, "carried": 0.6, "estimated": 0.3}


def _age(birth_date: pd.Series, season: float) -> pd.Series:
    start = pd.Timestamp(f"{int(season)}-08-01") + pd.Timedelta(days=365.25 * (season % 1))
    return (start - pd.to_datetime(birth_date)).dt.days / 365.25


def _with_lags(rows: pd.DataFrame, anchors: pd.DataFrame, gap: pd.Series | float) -> pd.DataFrame:
    """`rows` plus LAGS from `anchors` (latest_before output) and lag_gap."""
    lag = anchors.set_index("player_id")[TARGETS].add_prefix("lag_")
    out = rows.drop(columns=[c for c in LAGS + ["lag_gap"] if c in rows]).join(lag, on="player_id")
    out["lag_gap"] = gap
    out.loc[out[list(lag.columns)].isna().all(axis=1), "lag_gap"] = np.nan
    return out


def training_rows(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """One row per measured player-season and anchor (each earlier measured
    season within MAX_LAG_GAP, plus none): NUMERIC, CATEGORICAL, player_id,
    season, minutes, and the TARGETS as measured."""
    from recruit_intel.features.per90 import per90_table

    measured = history.measured_seasons(con, min_minutes=MIN_TRAIN_MINUTES)
    every = history.measured_seasons(con)
    parts = []
    for season in sorted(measured.season.unique()):
        profile = per90_table(con, int(season), fill=False)
        profile = profile[["player_id", "competition_id", "group", "birth_date", *PROFILE,
                           *CONTEXT]].dropna(subset=PROFILE)  # fmt: skip
        target = measured[measured.season == season].drop(columns=["group", "possession"])
        rows = target.merge(profile, on=["player_id", "competition_id"])
        if rows.empty:
            continue
        rows["age"] = _age(rows.pop("birth_date"), season)
        parts.append(_with_lags(rows, every.iloc[:0], np.nan))
        for earlier in range(int(season) - 1, int(season) - MAX_LAG_GAP - 1, -1):
            anchors = history.latest_before(every[every.season == earlier])
            if not anchors.empty:
                lagged = _with_lags(rows, anchors, season - earlier)
                parts.append(lagged[lagged.lag_gap.notna()])
    if not parts:
        return pd.DataFrame(columns=["player_id", "season", "minutes", *NUMERIC, *CATEGORICAL,
                                     *TARGETS])  # fmt: skip
    return pd.concat(parts, ignore_index=True)


def _features(df: pd.DataFrame, competitions: list[str], numeric: list[str]) -> pd.DataFrame:
    x = df.reindex(columns=numeric).astype(float)
    x["group"] = pd.Categorical(df["group"], categories=list(GROUPS))
    x["competition_id"] = pd.Categorical(df["competition_id"], categories=competitions)
    return x


@dataclass
class DefenceModel:
    competitions: list[str]
    # NUMERIC features with any data in training (e.g. no anchors without
    # older seasons loaded): an all-missing column can't be binned.
    numeric: list[str] = field(default_factory=lambda: list(NUMERIC))
    models: dict = field(default_factory=dict)  # (target, quantile) -> regressor
    # target -> {"carried"|"estimated": k}: held-out ranges were too narrow,
    # so each is widened around the median by k until it covers BAND_COVERAGE.
    band_scale: dict = field(default_factory=dict)
    report: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, rows: pd.DataFrame) -> "DefenceModel":
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = cls(
            competitions=sorted(rows.competition_id.dropna().unique()),
            numeric=[c for c in NUMERIC if c in rows and rows[c].notna().any()],
        )
        x_all = _features(rows, model.competitions, model.numeric)
        weight = (rows.minutes.astype(float) / 2700).clip(0.2, 1.0)
        for target in TARGETS:
            known = rows[target].notna().to_numpy()
            if known.sum() < 50:
                continue
            for q in QUANTILES:
                reg = HistGradientBoostingRegressor(
                    loss="quantile", quantile=q, max_iter=150, learning_rate=0.08,
                    max_leaf_nodes=15, min_samples_leaf=30, l2_regularization=1.0,
                    categorical_features="from_dtype", random_state=0,
                )  # fmt: skip
                reg.fit(x_all[known], rows.loc[known, target], sample_weight=weight[known])
                model.models[(target, q)] = reg
        return model

    def predict(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """target -> DataFrame(lo, mid, hi), aligned with df, quantiles sorted."""
        x = _features(df, self.competitions, self.numeric)
        out = {}
        for target in TARGETS:
            if (target, 0.5) not in self.models:
                continue
            q = np.column_stack([self.models[(target, q)].predict(x) for q in QUANTILES])
            q = np.sort(q.clip(min=0), axis=1)
            scale = self.band_scale.get(target, {})
            anchored = df[f"lag_{target}"].notna() if f"lag_{target}" in df else False
            k = np.where(anchored, scale.get("carried", 1.0), scale.get("estimated", 1.0))
            q[:, 0] = (q[:, 1] - k * (q[:, 1] - q[:, 0])).clip(min=0)
            q[:, 2] = q[:, 1] + k * (q[:, 2] - q[:, 1])
            out[target] = pd.DataFrame(q, columns=["lo", "mid", "hi"], index=df.index)
        return out

    def fill(
        self, con: duckdb.DuckDBPyConnection, out: pd.DataFrame, seasons: list[int]
    ) -> pd.DataFrame:
        """Estimate every TARGET the window lacks, for players with an Understat
        profile. Sets <metric>_kind/_lo/_hi and, for carried estimates,
        defence_season (the season they're anchored on)."""
        out = out.copy()
        mid = float(np.mean(seasons))
        x = out.assign(age=_age(out.birth_date, mid))
        anchors = history.latest_before(
            history.measured_seasons(con, before=min(seasons),
                                     player_ids=out.player_id.unique().tolist())
        )  # fmt: skip
        anchors = anchors[mid - anchors.season <= MAX_LAG_GAP + 0.5]
        x = _with_lags(x, anchors, out.player_id.map(anchors.set_index("player_id").season))
        x["lag_gap"] = mid - x.lag_gap
        predictions = self.predict(x)
        can = x.reindex(columns=PROFILE).notna().all(axis=1) & out.group.notna()
        anchor_season = out.player_id.map(anchors.set_index("player_id").season)
        for target, pred in predictions.items():
            if target not in out:
                out[target] = np.nan
            gap = can & out[target].isna()
            anchor = x[f"lag_{target}"]
            if f"lag_{target}" in self.numeric:
                kind = np.where(anchor.notna(), "carried", "estimated")
            else:
                # The model hasn't learned from anchors (no older seasons were
                # loaded to train on), so a player's own older season beats an
                # estimate that ignores it: use it as-is.
                kind = np.where(anchor.notna(), "older", "estimated")
            kind = pd.Series(kind, index=out.index)
            estimate = gap & (kind != "older")
            out.loc[estimate, target] = pred.loc[estimate, "mid"]
            out.loc[estimate, f"{target}_lo"] = pred.loc[estimate, "lo"]
            out.loc[estimate, f"{target}_hi"] = pred.loc[estimate, "hi"]
            older = gap & (kind == "older")
            out.loc[older, target] = anchor[older]
            out.loc[gap, f"{target}_kind"] = kind[gap]
            own = gap & (kind != "estimated")
            out.loc[own, "defence_season"] = anchor_season[own]
        return out

    def save(self, path: Path = PATH) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        REPORT.write_text(json.dumps(self.report, indent=2))


_loaded: dict[Path, tuple[float, DefenceModel]] = {}


def load(path: Path = PATH) -> DefenceModel | None:
    """The trained model, or None (reloaded when the file changes)."""
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    if path not in _loaded or _loaded[path][0] != mtime:
        import joblib

        # Our own file, written by `recruit train` (like the other models in
        # data/models/); never load one from elsewhere.

        _loaded[path] = (mtime, joblib.load(path))
    return _loaded[path][1]


def reliability() -> dict[str, float]:
    """How well each kind of value ranks players, from the last validation."""
    model = load()
    learned = model.report.get("reliability", {}) if model else {}
    return {**DEFAULT_RELIABILITY, **learned}


# ----------------------------------------------------------------- validation


def _within_group_corr(df: pd.DataFrame, pred: str, truth: str) -> float | None:
    """Spearman correlation inside each position group (percentiles are
    within-group), averaged weighted by group size."""
    total, n = 0.0, 0
    for _, g in df.dropna(subset=[pred, truth]).groupby("group"):
        if len(g) < MIN_GROUP_ROWS or g[pred].nunique() < 2 or g[truth].nunique() < 2:
            continue
        total += g[pred].rank().corr(g[truth].rank()) * len(g)
        n += len(g)
    return round(total / n, 3) if n else None


SCENARIOS = {
    "carried, anchor 2 seasons back": lambda r: r.lag_gap == 2,
    "carried, anchor 1 season back": lambda r: r.lag_gap == 1,
    "estimated, no anchor": lambda r: r.lag_gap.isna(),
}


def validate(rows: pd.DataFrame, folds: int = 5) -> dict:
    """Out-of-fold predictions with whole players held out, scored per
    scenario: within-group rank correlation of the estimate with the measured
    value, the same for the naive "anchor carried as-is", and how often the
    measured value falls inside the 10-90% band (should be ~80%)."""
    from sklearn.model_selection import GroupKFold

    preds = {t: pd.DataFrame(index=rows.index, columns=["lo", "mid", "hi"], dtype=float)
             for t in TARGETS}  # fmt: skip
    for train_idx, test_idx in GroupKFold(n_splits=folds).split(rows, groups=rows.player_id):
        model = DefenceModel.fit(rows.iloc[train_idx])
        for t, p in model.predict(rows.iloc[test_idx]).items():
            preds[t].loc[p.index] = p
    table = []
    for name, select in SCENARIOS.items():
        part = rows[select(rows)]
        if part.empty:
            continue
        for t in TARGETS:
            p = preds[t].loc[part.index]
            scored = part.assign(_pred=p.mid, _lag=part.get(f"lag_{t}", np.nan))
            known = scored[t].notna() & p.mid.notna()
            inside = ((part[t] >= p.lo) & (part[t] <= p.hi))[known]
            table.append({
                "scenario": name, "metric": t, "rows": int(known.sum()),
                "model": _within_group_corr(scored, "_pred", t),
                "anchor as-is": _within_group_corr(scored, "_lag", t),
                "band hit": round(float(inside.mean()), 2) if known.any() else None,
            })  # fmt: skip
    return {"validation": table, "reliability": _reliability(table),
            "band_scale": _band_scale(rows, preds)}  # fmt: skip


def _band_scale(rows: pd.DataFrame, preds: dict[str, pd.DataFrame]) -> dict:
    """Per target and kind, the factor that makes held-out ranges cover
    BAND_COVERAGE of measured values (split-conformal calibration)."""
    out = {}
    for t, p in preds.items():
        known = rows[t].notna() & p.mid.notna()
        if not known.any():
            continue
        y, p = rows.loc[known, t], p[known]
        eps = 1e-6
        need = np.maximum((p.mid - y) / (p.mid - p.lo + eps), (y - p.mid) / (p.hi - p.mid + eps))
        anchored = (rows.loc[known, f"lag_{t}"].notna() if f"lag_{t}" in rows
                    else pd.Series(False, index=y.index))  # fmt: skip
        scales = {}
        for kind, mask in (("carried", anchored), ("estimated", ~anchored)):
            if np.any(mask) and int(np.sum(mask)) >= MIN_GROUP_ROWS:
                k = float(np.quantile(need[mask].clip(lower=0), BAND_COVERAGE))
                scales[kind] = round(max(1.0, k), 2)
        out[t] = scales
    return out


def _reliability(table: list[dict]) -> dict[str, float]:
    def mean(rows, col):
        vals = [r[col] for r in rows if r[col] is not None]
        return float(np.clip(np.mean(vals), 0.05, 1.0)) if vals else None

    two_back = [r for r in table if r["scenario"].startswith("carried, anchor 2")]
    one_back = [r for r in table if r["scenario"].startswith("carried, anchor 1")]
    carried = two_back if any(r["model"] is not None for r in two_back) else one_back
    out = {
        "carried": mean(carried, "model"),
        "older": mean(carried, "anchor as-is"),
        "estimated": mean([r for r in table if r["scenario"].startswith("estimated")], "model"),
    }
    return {k: round(v, 2) for k, v in out.items() if v is not None}


def external_check(
    con: duckdb.DuckDBPyConnection, model: DefenceModel, season: int, competition: str
) -> dict | None:
    """Score the model on a season it never saw: FPL's measured tackles in the
    Premier League against the estimated ones, and against just reusing each
    player's latest API-Football season. Rank correlations within group; FPL
    counts tackles its own way, so only the ordering is comparable. (FPL's
    clearances+blocks+interceptions is mostly clearances, which no estimated
    metric measures, so it isn't compared.)"""
    from recruit_intel.features.per90 import per90_table

    table = per90_table(con, season, fill=False)
    table = table[(table.competition_id == competition) & (table.minutes >= history.MIN_MINUTES)]
    if table.empty or "tackles" not in table:
        return None
    blank = table.drop(columns=[c for c in TARGETS if c in table])
    filled = model.fill(con, blank, [season]).set_index("player_id")
    anchors = history.latest_before(
        history.measured_seasons(con, before=season, player_ids=table.player_id.tolist())
    ).set_index("player_id")
    both = table[["player_id", "group", "tackles"]].assign(
        estimate=table.player_id.map(filled.tackles),
        anchor=table.player_id.map(anchors.tackles) if not anchors.empty else np.nan,
    )
    with_anchor = both[both.anchor.notna()]
    return {
        "season": season, "competition": competition, "players": len(both),
        "model": _within_group_corr(both, "estimate", "tackles"),
        "players with an older season": len(with_anchor),
        "model, those players": _within_group_corr(with_anchor, "estimate", "tackles"),
        "older season as-is, those players": _within_group_corr(with_anchor, "anchor", "tackles"),
    }  # fmt: skip
