"""Squad profile: strengths, weaknesses and role fit from percentiles.

Everything here reads percentile rows (see features/percentiles.py), so the
judgements are always relative to positional peers in the club's peer group.
"""

from dataclasses import dataclass

import duckdb
import pandas as pd

from recruit_intel.config import ClubConfig, Role, load_roles
from recruit_intel.features import history
from recruit_intel.features.metrics import METRICS
from recruit_intel.features.per90 import per90_table
from recruit_intel.features.percentiles import percentiles
from recruit_intel.features.positions import GROUPS, group_of
from recruit_intel.identity.names import normalize
from recruit_intel.models import defence

STRENGTH = 80  # percentile at or above which a metric is a strength
WEAKNESS = 20  # at or below which it is a weakness
MIN_ROLE_COVERAGE = 0.6  # share of a role's weight that must have data


@dataclass(frozen=True)
class RoleFit:
    role: str
    score: float  # 0-100, weighted mean percentile
    coverage: float  # share of the role's weight with data (estimates included)
    lo: float | None = None  # score with every estimate at its 10th percentile...
    hi: float | None = None  # ...and at its 90th (equal to score when all measured)
    # Share of the role's weight backed by data, each metric counted by how
    # reliably its kind of value ranks players (measured = 1; see
    # models/defence.py). Search discounts fits by it.
    confidence: float = 1.0
    own_data: bool = True  # False if any metric is estimated with no data of his own


def strengths_weaknesses(row: pd.Series) -> tuple[list[str], list[str]]:
    """Metrics of the player's position group at the extremes, best/worst first.
    Estimates don't count: a strength has to be in the player's own data."""
    metrics = [m for m in GROUPS[row["group"]]["metrics"]
               if pd.notna(row.get(m)) and _kind(row, m) in ("measured", "older")]  # fmt: skip
    ranked = sorted(metrics, key=lambda m: row[m], reverse=True)
    strengths = [m for m in ranked if row[m] >= STRENGTH]
    weaknesses = [m for m in reversed(ranked) if row[m] <= WEAKNESS]
    return strengths, weaknesses


def role_fit(row: pd.Series, role: Role, groups: list[str] | None = None) -> RoleFit | None:
    """Weighted mean percentile over the role's metrics, renormalised over the
    metrics that have data. None if the player's position group can't play the
    role or too little of the role is covered (e.g. no FPL defensive data).

    `groups` overrides which position groups may play it: a formation slot can
    let a winger play wing-back even though the role is defined for full-backs.
    """
    if row.get("group") not in (groups or role.groups):
        return None
    available = {m: w for m, w in role.weights.items() if pd.notna(row.get(m))}
    coverage = sum(available.values()) / sum(role.weights.values())
    if coverage < MIN_ROLE_COVERAGE:
        return None
    total = sum(available.values())
    score = sum(row[m] * w for m, w in available.items()) / total
    reliable = defence.reliability()
    kinds = {m: _kind(row, m) for m in available}

    def bound(side: str) -> float:
        return sum(_value(row, m, side) * w for m, w in available.items()) / total

    confidence = sum(reliable.get(kinds[m], 0.0) * w for m, w in available.items()) / sum(
        role.weights.values()
    )
    return RoleFit(role.key, score, coverage, lo=bound("lo"), hi=bound("hi"),
                   confidence=confidence,
                   own_data="estimated" not in kinds.values())  # fmt: skip


def _kind(row: pd.Series, metric: str) -> str:
    kind = row.get(f"{metric}_kind")
    return kind if isinstance(kind, str) else "measured"


def _value(row: pd.Series, metric: str, side: str) -> float:
    bound = row.get(f"{metric}_{side}")
    return float(row[metric] if bound is None or pd.isna(bound) else bound)


def role_fits(row: pd.Series, roles: dict[str, Role]) -> list[RoleFit]:
    """Every role the player can be scored for, best first."""
    fits = [f for r in roles.values() if (f := role_fit(row, r))]
    return sorted(fits, key=lambda f: f.score, reverse=True)


def best_role(row: pd.Series, roles: dict[str, Role]) -> RoleFit | None:
    fits = role_fits(row, roles)
    return fits[0] if fits else None


def label(metric: str) -> str:
    return METRICS[metric].label


def peer_percentiles(con: duckdb.DuckDBPyConnection, club: ClubConfig, season: int):
    """Percentiles for every player in the club's peer-group competition."""
    return percentiles(per90_table(con, season), club.peer_group,
                       reference=history.reference(con))  # fmt: skip


def find_players(pct: pd.DataFrame, query: str) -> pd.DataFrame:
    """Players whose name contains every word of `query`, accents ignored.
    An exact full-name match wins outright."""
    words = normalize(query).split()
    names = pct["name"].map(normalize)
    exact = pct[names == " ".join(words)]
    if not exact.empty:
        return exact
    return pct[names.map(lambda n: all(w in n.split() or w in n for w in words))]


def squad_roster(con: duckdb.DuckDBPyConnection, club: ClubConfig) -> pd.DataFrame:
    """Every registered squad player (the club's FPL roster), rated or not:
    player_id, name, group."""
    roster = con.execute(
        """
        SELECT DISTINCT ids.player_id, pl.name, pl.position, pl.detailed_position
        FROM source_players p
        JOIN player_source_ids ids USING (source, source_player_id)
        JOIN players pl USING (player_id)
        WHERE p.source = 'fpl' AND p.team_name = ?
        """,
        [str(club.source_name("fpl"))],
    ).df()
    roster["group"] = [
        group_of(d, b) for d, b in zip(roster.detailed_position, roster.position, strict=True)
    ]
    return roster[["player_id", "name", "group"]]


def squad_percentiles(
    con: duckdb.DuckDBPyConnection, club: ClubConfig, season: int
) -> pd.DataFrame:
    """Percentile rows for the club's squad (its FPL roster); attrs carry min_minutes."""
    squad_ids = squad_roster(con, club).player_id
    pct = peer_percentiles(con, club, season)
    squad = pct[pct.player_id.isin(squad_ids)].reset_index(drop=True)
    squad.attrs["min_minutes"] = pct.attrs["min_minutes"]
    return squad


def squad_profile(con: duckdb.DuckDBPyConnection, club: ClubConfig, season: int) -> pd.DataFrame:
    """One row per squad player: group, qualified, strengths, weaknesses, best role."""
    squad = pct = squad_percentiles(con, club, season)

    roles = load_roles()
    rows = []
    for _, row in squad.iterrows():
        strengths, weaknesses = strengths_weaknesses(row)
        fit = best_role(row, roles)
        rows.append(
            {
                "player_id": row.player_id,
                "name": row["name"],
                "group": row["group"],
                "minutes": int(row.minutes),
                "qualified": bool(row.qualified),
                "best_role": roles[fit.role].label if fit else None,
                "role_fit": round(fit.score) if fit else None,
                "strengths": [label(m) for m in strengths],
                "weaknesses": [label(m) for m in weaknesses],
            }
        )
    out = pd.DataFrame(rows)
    order = {g: i for i, g in enumerate(GROUPS)}
    out = out.sort_values(["group", "minutes"], key=lambda s: s.map(order) if s.name == "group"
                          else -s)  # fmt: skip
    out.attrs["min_minutes"] = pct.attrs["min_minutes"]
    return out.reset_index(drop=True)
