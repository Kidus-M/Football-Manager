"""Computations behind the API, cached per process.

Everything here reads the DuckDB file read-only and reuses the analysis
modules the CLI uses, so the API and CLI always agree. Results that are
expensive (per-90 tables, percentiles, the candidate pool) are cached for
the life of the process; restart the API after `recruit ingest`.
"""

import math
import os
import threading
from functools import cache

import duckdb
import numpy as np
import pandas as pd

from recruit_intel import db
from recruit_intel.analysis import gaps, profile, tactics
from recruit_intel.config import ClubConfig, lineup, load_club, load_formations, load_roles
from recruit_intel.features.metrics import METRICS
from recruit_intel.features.positions import GROUPS
from recruit_intel.search import rank as rank_mod
from recruit_intel.search.pool import candidate_pool, league_strength

_lock = threading.Lock()


@cache
def _con() -> duckdb.DuckDBPyConnection:
    path = os.environ.get("RECRUIT_DB", str(db.DEFAULT_DB))
    return duckdb.connect(path, read_only=True)


def con() -> duckdb.DuckDBPyConnection:
    """A cursor on the shared read-only connection (cursors are thread-safe
    to use one per request)."""
    with _lock:
        return _con().cursor()


@cache
def club() -> ClubConfig:
    return load_club(os.environ.get("RECRUIT_CLUB", "man_utd"))


@cache
def season() -> int:
    return (
        con()
        .execute("SELECT max(season) FROM player_season_stats WHERE source = 'fpl'")
        .fetchone()[0]
    )


def clean(value):
    """JSON-safe: NaN/NA -> None, numpy scalars -> Python."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float | np.floating):
        return None if math.isnan(value) else round(float(value), 3)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def records(df: pd.DataFrame) -> list[dict]:
    return [{k: clean(v) for k, v in row.items()} for row in df.to_dict("records")]


# ---------------------------------------------------------------- club


def club_info() -> dict:
    c = club()
    return {
        "club": c.club,
        "season": season(),
        "season_label": f"{season()}-{(season() + 1) % 100:02d}",
        "formation": c.formation,
        "peer_group": c.peer_group,
        "theme": c.theme.model_dump() if c.theme else None,
        "recruitment": c.recruitment.model_dump(),
        "formations": sorted(load_formations()),
    }


# ---------------------------------------------------------------- lineup


@cache
def _gap_report(formation: str | None, choices: tuple = ()) -> tuple[pd.DataFrame, list]:
    roles = tactics.reweight(load_roles(), dict(choices)) if choices else None
    return gaps.gap_report(con(), club(), season(), formation, roles=roles)


def lineup_view(formation: str | None = None, choices: tuple = ()) -> dict:
    """Best XI; `choices` (e.g. (("press", "high"),)) re-runs it under
    tactically reweighted roles."""
    report, chart = _gap_report(formation, choices)
    roles = tactics.reweight(load_roles(), dict(choices)) if choices else load_roles()
    slots = []
    for row, depth in zip(report.itertuples(), chart, strict=True):
        s = depth.slot
        starter = depth.starter
        slots.append(
            {
                "key": s.key,
                "x": s.x,
                "y": s.y,
                "groups": s.groups,
                "role": roles[s.role].label,
                "starter": None if starter is None else {
                    "player_id": starter.player_id, "name": starter.name,
                    "fit": round(starter.adjusted), "minutes": starter.minutes,
                },
                "backups": [{"player_id": b.player_id, "name": b.name,
                             "fit": round(b.adjusted)} for b in depth.backups[:4]],
                "unrated": depth.unrated,
                "age": clean(row.age),
                "contract": clean(row.contract),
                "severity": clean(row.severity),
                "issues": list(row.issues),
            }
        )  # fmt: skip
    return {
        "formation": formation or club().formation,
        "slots": slots,
        "priorities": records(gaps.priorities(report)[["slot", "role", "severity", "issues"]]),
    }


# ---------------------------------------------------------------- players


@cache
def _peer_pct() -> pd.DataFrame:
    return profile.peer_percentiles(con(), club(), season())


@cache
def _pool(n_seasons: int = 2) -> pd.DataFrame:
    seasons = list(range(season() - n_seasons + 1, season() + 1))
    return candidate_pool(con(), seasons)


def squad() -> list[dict]:
    table = profile.squad_profile(con(), club(), season())
    return records(table)


def player(player_id: str) -> dict | None:
    pct = _peer_pct()
    pool = _pool()
    rows = pct[pct.player_id == player_id]
    basis = "peer_group"
    if rows.empty:  # not in the club's league: use his own league, two seasons
        rows, basis = pool[pool.player_id == player_id], "own_league"
    if rows.empty:
        return None
    row = rows.iloc[0]
    group = row["group"]
    metrics = GROUPS[group]["metrics"] if group in GROUPS else []
    strengths, weaknesses = profile.strengths_weaknesses(row) if group in GROUPS else ([], [])
    roles = load_roles()
    market = pool[pool.player_id == player_id]
    m = market.iloc[0] if not market.empty else None
    return {
        "player_id": player_id,
        "name": row["name"],
        "group": group,
        "group_label": GROUPS[group]["label"] if group in GROUPS else None,
        "detailed_position": clean(row.get("detailed_position")),
        "competition": clean(row.get("competition_id")),
        "basis": basis,
        "minutes": clean(row.minutes),
        "qualified": clean(row.get("qualified")),
        # Season the player's own older API-Football numbers come from, when
        # they are used as-is or anchor an estimate.
        "defence_season": clean(row.get("defence_season")),
        # Metrics not measured in the window: "older" (an older season as-is),
        # "carried" (estimated from his older season and current profile) or
        # "estimated" (no data of his own), with the estimate's likely range
        # in percentiles.
        "provenance": {
            k: {"kind": kind, "lo": clean(row.get(f"{k}_lo")), "hi": clean(row.get(f"{k}_hi"))}
            for k in METRICS
            if isinstance(kind := row.get(f"{k}_kind"), str)
        },
        "peers": clean(row.get("peers")),
        "club": clean(m.club) if m is not None else None,
        "age": clean(m.age) if m is not None else None,
        "value_eur": clean(m.value_eur) if m is not None else None,
        "contract_year": clean(m.contract_year) if m is not None else None,
        "percentiles": [
            {"metric": k, "label": profile.label(k), "value": clean(row.get(k))} for k in metrics
        ],  # fmt: skip
        # Every metric, so players from different position groups can be
        # compared on one radar.
        "all_percentiles": {k: clean(row.get(k)) for k in METRICS if k in row.index},
        "strengths": [profile.label(k) for k in strengths],
        "weaknesses": [profile.label(k) for k in weaknesses],
        "roles": [
            {
                "role": roles[f.role].label,
                "fit": round(f.score),
                "lo": round(f.lo),
                "hi": round(f.hi),
                "confidence": round(f.confidence, 2),
                "own_data": f.own_data,
            }
            for f in profile.role_fits(row, roles)[:3]
        ],  # fmt: skip
    }


def find(q: str, limit: int = 12) -> list[dict]:
    pool = _pool()
    hits = profile.find_players(pool, q) if q.strip() else pool.iloc[:0]
    hits = hits.sort_values("minutes", ascending=False).head(limit)
    return records(hits[["player_id", "name", "club", "group", "competition_id", "age"]])


# ---------------------------------------------------------------- search


def search(slot: str, formation: str | None = None, max_age: int | None = None,
           max_value_m: float | None = None, contract_until: int | None = None,
           leagues: list[str] | None = None, min_minutes: int | None = None,
           upgrades: bool = False, limit: int = 20,
           own_data_only: bool = False) -> dict:  # fmt: skip
    from recruit_intel.ingest import transfermarkt

    c = club()
    _, chart = _gap_report(formation)
    depth = next((d for d in chart if d.slot.key == slot), None)
    if depth is None:
        raise KeyError(slot)
    role = load_roles()[depth.slot.role]
    pool = _pool()
    countries = leagues or c.recruitment.pool
    budget = int(max_value_m * 1e6) if max_value_m else c.recruitment.max_value_eur
    filters = rank_mod.Filters(
        groups=depth.slot.groups,
        competitions=[transfermarkt.LEAGUES[x][1] for x in countries if x in transfermarkt.LEAGUES],
        max_age=max_age or c.recruitment.max_age,
        max_value_eur=budget,
        contract_until=contract_until,
        min_minutes=min_minutes or c.recruitment.min_minutes,
        exclude=set(profile.squad_roster(con(), c).player_id),
        own_data_only=own_data_only,
    )
    strength = league_strength()
    ranked = rank_mod.rank(pool, role, filters, strength, c.season_start_year + 1, budget)
    base = None
    if depth.starter is not None:
        fits = rank_mod.fit_scores(pool[pool.player_id == depth.starter.player_id], role,
                                   depth.slot.groups, strength)  # fmt: skip
        base = None if fits.empty else float(fits.fit.iloc[0])
    if upgrades and base is not None and not ranked.empty:
        ranked = ranked[ranked.fit > base]
    if not ranked.empty:
        ranked = ranked.assign(vs_starter=(ranked.fit - base) if base is not None else None)
    cols = ["player_id", "name", "club", "competition_id", "age", "value_eur",
            "contract_year", "fit", "fit_lo", "fit_hi", "confidence", "own_data", "score",
            "vs_starter", "why"]  # fmt: skip
    return {
        "slot": slot,
        "role": role.label,
        "starter": None if depth.starter is None else {"name": depth.starter.name, "fit": base},
        "candidates": records(ranked.head(limit).reindex(columns=cols)),
    }


# ---------------------------------------------------------------- tactics


def tactics_view(press: str | None = None, buildup: str | None = None,
                 width: str | None = None) -> dict:  # fmt: skip
    report, chart = _gap_report(None)
    sq = profile.squad_percentiles(con(), club(), season()).set_index("player_id")
    starters = sq.loc[[d.starter.player_id for d in chart if d.starter]].reset_index()
    levers = tactics.lever_readings(starters, min_minutes=sq.attrs["min_minutes"])
    context = None
    try:
        from recruit_intel.models.club_context import club_context, lever_z

        ctx = club_context(con(), club())
        context = {"team": ctx.team, "traits": ctx.traits, "manager": ctx.manager,
                   "manager_signature": ctx.manager_signature,
                   "manager_clubs": ctx.manager_clubs}  # fmt: skip
        for r in levers:
            if r.lever in tactics.STYLE_SIGNAL:
                feature, sign = tactics.STYLE_SIGNAL[r.lever]
                r.style_z = lever_z(ctx.style_z, feature, sign)
    except (ImportError, FileNotFoundError, LookupError):
        pass
    choices = tuple((k, v) for k, v in (("press", press), ("buildup", buildup),
                                         ("width", width)) if v)  # fmt: skip
    changes = []
    if choices:
        what_if, _ = _gap_report(None, choices)
        changes = [c.__dict__ for c in tactics.compare(report, what_if)]
    return {
        "context": context,
        "levers": [
            {
                "lever": r.lever,
                "squad": clean(r.squad),
                "verdict": r.verdict,
                "best": r.best,
                "worst": r.worst,
                "advice": tactics.advise(r),
            }
            for r in levers
        ],  # fmt: skip
        "instructions": [{"player": n, "text": t} for n, t in tactics.instructions(starters)],
        "settings": {k: list(v) for k, v in tactics.load_config()["settings"].items()},
        "what_if": [{k: clean(v) for k, v in c.items()} for c in changes],
    }


# ---------------------------------------------------------------- models (optional)


@cache
def _per90_pool() -> pd.DataFrame:
    from recruit_intel.search.pool import per90_pool

    return per90_pool(con(), list(range(season() - 1, season() + 1)))


@cache
def similarity_map() -> list[dict]:
    """2-D map of player embeddings (PCA of the autoencoder space) for
    qualifying players; needs `recruit train`."""
    from recruit_intel.models.dataset import EMBED_FEATURES
    from recruit_intel.models.embeddings import Embedder

    pool = _per90_pool()
    pool = pool[(pool.minutes >= 900) & pool.group.notna()].dropna(subset=EMBED_FEATURES)
    pool = pool.reset_index(drop=True)
    v = Embedder.load().embed(pool)
    v = v - v.mean(axis=0)
    _, _, vt = np.linalg.svd(v, full_matrices=False)
    xy = v @ vt[:2].T
    out = pool[["player_id", "name", "club", "group", "competition_id", "age", "value_eur"]]
    out = out.assign(x=xy[:, 0], y=xy[:, 1])
    return records(out)


def similar(player_id: str, younger: bool, cheaper: bool, same_position: bool,
            limit: int = 12) -> list[dict]:  # fmt: skip
    from recruit_intel.models.embeddings import Embedder
    from recruit_intel.models.similarity import similar_players

    out = similar_players(_per90_pool(), player_id, Embedder.load(), k=limit,
                          same_group=same_position, younger=younger, cheaper=cheaper,
                          exclude=set(profile.squad_roster(con(), club()).player_id))  # fmt: skip
    return records(out[["player_id", "name", "club", "group", "competition_id", "age",
                        "value_eur", "distance"]])  # fmt: skip


def formations() -> dict:
    return {
        name: [{"key": s.key, "x": s.x, "y": s.y, "groups": s.groups} for s in lineup(club(), name)]
        for name in load_formations()
    }
