"""Filter the pool for a slot and rank candidates with explainable scores.

The score is a weighted sum of four 0-100 components, and each candidate's
row says what drove it, so a rank is never a black-box verdict:

- fit: role fit vs league peers, discounted by league strength and shrunk
  toward 50 for small samples (the dominant term)
- age: resale and peak-years profile, best at 23-26
- value: cheaper is better (relative to the budget when there is one)
- contract: fewer years left means a lower fee
"""

from dataclasses import dataclass, field

import pandas as pd

from recruit_intel.analysis.lineup import shrink
from recruit_intel.analysis.profile import role_fit
from recruit_intel.config import Role

WEIGHTS = {"fit": 0.65, "age": 0.15, "value": 0.10, "contract": 0.10}
RELIABLE_MINUTES = 1800  # two-season window: about 20 full matches


@dataclass
class Filters:
    groups: list[str]
    competitions: list[str] | None = None
    max_age: int | None = None
    max_value_eur: int | None = None
    contract_until: int | None = None  # contract ends in this year or earlier
    min_minutes: int = 900
    exclude: set[str] = field(default_factory=set)  # player_ids (own squad)


def age_score(age) -> float:
    if pd.isna(age):
        return 50.0
    curve = {20: 75, 21: 90, 22: 90, 23: 100, 24: 100, 25: 100, 26: 100, 27: 85, 28: 70,
             29: 50, 30: 30}  # fmt: skip
    return float(curve.get(int(age), 75 if age < 20 else 10))


def contract_score(contract_year, window_year: int) -> float:
    """window_year: the summer a deal would happen. Expiring then = free."""
    if pd.isna(contract_year):
        return 50.0
    years_left = int(contract_year) - window_year
    return {0: 100.0, 1: 85.0, 2: 60.0, 3: 40.0}.get(years_left, 100.0 if years_left < 0 else 25.0)


def value_scores(values: pd.Series, budget: int | None) -> pd.Series:
    if budget:
        return (100 * (1 - values / budget)).clip(0, 100).fillna(50)
    # No budget: cheaper than the other candidates is better.
    return (100 - values.rank(pct=True) * 100).fillna(50)


def apply_filters(pool: pd.DataFrame, f: Filters) -> pd.DataFrame:
    keep = pool.group.isin(f.groups) & (pool.minutes >= f.min_minutes)
    keep &= ~pool.player_id.isin(f.exclude)
    if f.competitions:
        keep &= pool.competition_id.isin(f.competitions)
    if f.max_age is not None:
        keep &= pool.age.fillna(99) <= f.max_age
    if f.max_value_eur is not None:
        keep &= pool.value_eur.fillna(0) <= f.max_value_eur
    if f.contract_until is not None:
        keep &= pool.contract_year.fillna(9999) <= f.contract_until
    return pool[keep]


def fit_scores(pool: pd.DataFrame, role: Role, groups: list[str],
               strength: dict[str, float]) -> pd.DataFrame:  # fmt: skip
    """fit_raw, coverage and the league- and sample-adjusted fit per player."""
    rows = []
    for _, row in pool.iterrows():
        fit = role_fit(row, role, groups=groups)
        if fit is None:
            continue
        s = strength.get(row.competition_id, 0.8)
        adjusted = shrink(50 + (fit.score - 50) * s, row.minutes, RELIABLE_MINUTES)
        rows.append({"player_id": row.player_id, "fit_raw": fit.score,
                     "coverage": fit.coverage, "fit": adjusted})  # fmt: skip
    return pd.DataFrame(rows, columns=["player_id", "fit_raw", "coverage", "fit"])


def rank(
    pool: pd.DataFrame,
    role: Role,
    filters: Filters,
    strength: dict[str, float],
    window_year: int,
    budget: int | None = None,
) -> pd.DataFrame:
    cands = apply_filters(pool, filters)
    cands = cands.merge(fit_scores(cands, role, filters.groups, strength), on="player_id")
    if cands.empty:
        return cands
    cands["age_score"] = cands.age.map(age_score)
    cands["value_score"] = value_scores(cands.value_eur, budget).to_numpy()
    cands["contract_score"] = cands.contract_year.map(lambda y: contract_score(y, window_year))
    cands["score"] = (
        WEIGHTS["fit"] * cands.fit
        + WEIGHTS["age"] * cands.age_score
        + WEIGHTS["value"] * cands.value_score
        + WEIGHTS["contract"] * cands.contract_score
    )
    cands["why"] = [_why(r) for r in cands.itertuples()]
    return cands.sort_values("score", ascending=False).reset_index(drop=True)


def _why(r) -> str:
    league = r.competition_id.split("-", 1)[-1]
    parts = [f"fit {r.fit_raw:.0f} in {league} -> {r.fit:.0f}"]
    if r.coverage < 1:
        parts.append(f"{r.coverage:.0%} of role data")
    if pd.notna(r.age):
        parts.append(f"age {r.age}")
    if pd.notna(r.value_eur):
        parts.append(f"EUR {r.value_eur / 1e6:.1f}M")
    if pd.notna(r.contract_year):
        parts.append(f"contract {r.contract_year}")
    return ", ".join(parts)
