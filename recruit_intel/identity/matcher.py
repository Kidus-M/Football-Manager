"""Cross-source player identity: one canonical player_id per person.

Transfermarkt is the anchor (birth dates, widest coverage), so a matched
player's id is 'tm-<transfermarkt id>'. Other sources link to it:

- FPL has birth dates: candidates must share one, then the best name wins.
- Understat has none: match within the mapped club first, then across the
  competition, then everywhere, each step demanding a higher name score and
  a clear margin over the runner-up so near-ties stay unmatched.
- FPL and Understat players that no Transfermarkt squad lists (loans,
  academy) are linked to each other within the same club.

Unmatched players keep a source-prefixed id ('fpl-12', 'us-345') so every
source row still has exactly one canonical player.
"""

from dataclasses import dataclass

import pandas as pd
from rapidfuzz import fuzz

from recruit_intel.identity.names import normalize
from recruit_intel.identity.teams import match_teams

PREFIX = {"transfermarkt": "tm", "fpl": "fpl", "understat": "us"}
EPL = "ENG-Premier League"


def name_score(a: str, b: str) -> float:
    """0-100. Subset names ('Bruno Fernandes' in 'Bruno Borges Fernandes') score
    high, but an exact full-name match still outranks a subset one."""
    a, b = normalize(a), normalize(b)
    return 0.7 * fuzz.token_set_ratio(a, b) + 0.3 * fuzz.token_sort_ratio(a, b)


@dataclass(frozen=True)
class Stage:
    method: str
    min_score: float
    margin: float


# Understat -> Transfermarkt, narrowest scope first.
UNDERSTAT_STAGES = (
    Stage("team", 75, 5),
    Stage("competition", 88, 8),
    Stage("global", 95, 10),
)
DOB_MIN_SCORE = 55
DIRECT_STAGE = Stage("direct", 80, 5)

Links = dict[str, tuple[str, float, str]]  # source id -> (target id, score, method)


def _assign(candidates: list[tuple[float, str, str]], links: Links, taken: set[str], method: str):
    """Greedy one-to-one: best-scoring pairs claim first."""
    for score, src_id, target_id in sorted(candidates, reverse=True):
        if src_id in links or target_id in taken:
            continue
        links[src_id] = (target_id, score, method)
        taken.add(target_id)


def _best(name: str, pool: pd.DataFrame, stage: Stage) -> tuple[float, str] | None:
    if pool.empty:
        return None
    scores = sorted(
        (
            (name_score(name, n), pid)
            for pid, n in zip(pool.source_player_id, pool.name, strict=True)
        ),
        reverse=True,
    )
    top = scores[0]
    runner_up = scores[1][0] if len(scores) > 1 else 0.0
    if top[0] >= stage.min_score and top[0] - runner_up >= stage.margin:
        return top
    return None


def match_by_birth_date(src: pd.DataFrame, tm: pd.DataFrame) -> Links:
    pairs = src.dropna(subset=["birth_date"]).merge(
        tm.dropna(subset=["birth_date"]), on="birth_date", suffixes=("", "_tm")
    )
    cands = []
    for row in pairs.itertuples():
        score = name_score(row.name, row.name_tm)
        if score >= DOB_MIN_SCORE:
            cands.append((score, row.source_player_id, row.source_player_id_tm))
    links: Links = {}
    _assign(cands, links, set(), "birth_date")
    return links


def understat_team_map(us: pd.DataFrame, tm: pd.DataFrame) -> dict[tuple[str, str], str]:
    """(competition, Understat team) -> Transfermarkt team, per competition."""
    team_map = {}
    for comp, group in us.groupby("competition_id"):
        tm_teams = tm.loc[tm.competition_id == comp, "team_name"].dropna().unique().tolist()
        for src_team, tm_team in match_teams(group.team_name.unique().tolist(), tm_teams).items():
            team_map[(comp, src_team)] = tm_team
    return team_map


def match_understat(us: pd.DataFrame, tm: pd.DataFrame, team_map: dict) -> Links:
    links: Links = {}
    taken: set[str] = set()
    for stage in UNDERSTAT_STAGES:
        free_tm = tm[~tm.source_player_id.isin(taken)]
        cands = []
        for row in us[~us.source_player_id.isin(links)].itertuples():
            if stage.method == "team":
                tm_team = team_map.get((row.competition_id, row.team_name))
                pool = free_tm[free_tm.team_name == tm_team] if tm_team else free_tm.iloc[:0]
            elif stage.method == "competition":
                pool = free_tm[free_tm.competition_id == row.competition_id]
            else:
                pool = free_tm
            best = _best(row.name, pool, stage)
            if best:
                cands.append((best[0], row.source_player_id, best[1]))
        _assign(cands, links, taken, stage.method)
    return links


def match_direct(fpl: pd.DataFrame, us: pd.DataFrame, fpl_clubs: dict, us_clubs: dict) -> Links:
    """Leftover Understat -> FPL players at the same club (compared through the
    Transfermarkt club each source's team name maps to)."""
    fpl = fpl.assign(club=fpl.team_name.map(fpl_clubs))
    cands = []
    for u in us.itertuples():
        club = us_clubs.get(u.team_name)
        if club:
            best = _best(u.name, fpl[fpl.club == club], DIRECT_STAGE)
            if best:
                cands.append((best[0], u.source_player_id, best[1]))
    links: Links = {}
    _assign(cands, links, set(), "direct")
    return links


def build_identity(
    tm: pd.DataFrame, fpl: pd.DataFrame, us: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Rows for `players` and `player_source_ids`.

    tm:  source_player_id, name, birth_date, nationality, position, team_name, competition_id
    fpl: source_player_id, name, birth_date, position, team_name
    us:  source_player_id, name, position, team_name, competition_id
    """
    fpl_links = match_by_birth_date(fpl, tm)
    team_map = understat_team_map(us, tm)
    us_links = match_understat(us, tm, team_map)

    tm_epl_teams = tm.loc[tm.competition_id == EPL, "team_name"].dropna().unique().tolist()
    direct = match_direct(
        fpl[~fpl.source_player_id.isin(fpl_links)],
        us[~us.source_player_id.isin(us_links) & (us.competition_id == EPL)],
        match_teams(fpl.team_name.unique().tolist(), tm_epl_teams),
        {team: club for (comp, team), club in team_map.items() if comp == EPL},
    )

    rows = [("tm-" + pid, "transfermarkt", pid, 100.0, "anchor") for pid in tm.source_player_id]
    for source, links in (("fpl", fpl_links), ("understat", us_links)):
        for src_id, (tm_id, score, method) in links.items():
            rows.append(("tm-" + tm_id, source, src_id, score, method))
    for us_id, (fpl_id, score, method) in direct.items():
        # The FPL player may itself be unmatched, so it carries an fpl- id.
        rows.append(("fpl-" + fpl_id, "understat", us_id, score, method))

    linked = {(source, pid) for _, source, pid, _, _ in rows}
    for source, df in (("fpl", fpl), ("understat", us)):
        for pid in df.source_player_id:
            if (source, pid) not in linked:
                rows.append((f"{PREFIX[source]}-{pid}", source, pid, None, "unmatched"))

    ids = pd.DataFrame(
        rows, columns=["player_id", "source", "source_player_id", "match_score", "method"]
    )

    # Canonical attributes: prefer Transfermarkt, then FPL, then Understat.
    attrs = (
        pd.concat(
            [
                tm.assign(source="transfermarkt", rank=0),
                fpl.assign(source="fpl", rank=1),
                us.assign(source="understat", rank=2),
            ],
            ignore_index=True,
        )
        .merge(ids[["player_id", "source", "source_player_id"]], on=["source", "source_player_id"])
        .sort_values("rank")
    )
    players = attrs.groupby("player_id", as_index=False).agg(
        name=("name", "first"),
        birth_date=("birth_date", "first"),
        nationality=("nationality", "first"),
        position=("position", "first"),
    )
    return {"players": players, "player_source_ids": ids}
