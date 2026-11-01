"""How a club plays now, and how its manager tends to play: the style side
of the tactics report. Needs the trained style model (`recruit train`)."""

from dataclasses import dataclass

import duckdb
import pandas as pd

from recruit_intel.config import ClubConfig
from recruit_intel.models import managers, team_style


@dataclass
class ClubContext:
    team: str
    season: int
    traits: str
    style_z: dict[str, float]  # strength-adjusted, STYLE_FEATURES
    manager: str | None
    manager_matches: int | None
    manager_clubs: int | None
    manager_signature: str | None
    manager_z: dict[str, float] | None


def club_context(con: duckdb.DuckDBPyConnection, club: ClubConfig) -> ClubContext:
    model = team_style.load()
    seasons = team_style.team_seasons(con)
    name = str(club.source_name("understat"))
    mine = seasons[seasons.team_name == name].sort_values("season")
    if mine.empty:
        raise LookupError(f"no team-style data for {name}")
    latest = mine.tail(1)
    z = dict(zip(team_style.STYLE_FEATURES, model.adjusted(latest)[0].tolist(), strict=True))

    ctx = ClubContext(name, int(latest.season.iloc[0]), model.traits(latest)[0], z,
                      None, None, None, None, None)  # fmt: skip
    current = con.execute(
        """
        SELECT manager_id, name FROM manager_spells
        WHERE club_id = ? AND ended IS NULL ORDER BY appointed DESC LIMIT 1
        """,
        [str(club.source_name("transfermarkt_id"))],
    ).fetchone()
    if current is None:
        return ctx
    try:
        profs = managers.profiles(managers.stints(con, model))
    except LookupError:
        return ctx
    prof = profs[profs.manager_id == current[0]]
    ctx.manager = current[1]
    if not prof.empty:
        p = prof.iloc[0]
        ctx.manager_matches, ctx.manager_clubs = int(p.matches), int(p.clubs)
        ctx.manager_signature = managers.signature(p)
        ctx.manager_z = {f: float(p[f]) for f in team_style.STYLE_FEATURES}
    return ctx


def lever_z(z: dict[str, float] | None, feature: str, sign: int) -> float | None:
    if z is None or pd.isna(z.get(feature)):
        return None
    return sign * z[feature]
