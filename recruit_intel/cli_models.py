"""ML commands: `recruit train | similar | roles | bargains`.

Kept apart from cli.py so the core commands never import torch/sklearn.
"""

import argparse

import pandas as pd

from recruit_intel import db
from recruit_intel.config import load_club

WIDE = {"display.width": 250, "display.max_colwidth": 80}


def _seasons(con, n: int) -> list[int]:
    from recruit_intel.models.dataset import seasons_available

    return seasons_available(con)[-n:]


def cmd_train(args: argparse.Namespace) -> int:
    from recruit_intel.models import embeddings, projection
    from recruit_intel.models.dataset import player_seasons

    con = db.connect(args.db)
    seasons = player_seasons(con)
    print(f"{len(seasons)} player-seasons with 900+ minutes, "
          f"{seasons.season.min()}-{seasons.season.max()}\n")  # fmt: skip

    emb, m = embeddings.train(seasons)
    emb.save()
    print("Player embeddings (autoencoder, 6 xG-based per-90 features -> 3 dims)")
    print(f"  reconstruction MSE: train {m['train_mse']:.3f}, held-out {m['test_mse']:.3f} "
          f"(predicting the mean: {m['baseline_mse']:.1f})\n")  # fmt: skip

    proj, report = projection.train_and_validate(seasons)
    proj.save()
    projection.save_league_factors(report["league_factors"])
    print("League translation (gradient boosting, next-season per-90)")
    print(f"  {report['train']} training pairs; tested on {report['test']} unseen next "
          f"seasons ({report['moves_in_test']} league moves)")  # fmt: skip
    rows = [{"metric": t, "R2 model": v["r2_model"], "R2 'same as last year'":
             v["r2_same_as_last_year"]} for t, v in report["metrics"].items()]  # fmt: skip
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\n  Learned league factors (output kept moving to {projection.REFERENCE}):")
    for comp, f in sorted(report["league_factors"].items(), key=lambda kv: -kv[1]):
        print(f"    {comp:<22} {f:.2f}")

    from recruit_intel.models import team_style

    style, sm = team_style.fit(team_style.team_seasons(con))
    team_style.save(style)
    print(f"\nTeam style ({sm['team_seasons']} team-seasons)")
    print("  share of each feature explained by team strength (removed before styling):")
    print("   " + ", ".join(f"{k} {v:.0%}" for k, v in sm["strength_r2"].items()))
    print(f"  2-D style map reconstruction MSE {sm['map_test_mse']} (mean predictor: 1.0)")

    print("\nSaved to data/models/; `recruit search` now uses the learned league factors.")
    return 0


def _find_team(seasons: pd.DataFrame, query: str) -> pd.DataFrame:
    from recruit_intel.identity.names import normalize

    words = normalize(query).split()
    names = seasons.team_name.map(normalize)
    hits = seasons[names.map(lambda n: all(w in n for w in words))]
    exact = hits[names[hits.index] == " ".join(words)]
    return exact if not exact.empty else hits


def cmd_manager(args: argparse.Namespace) -> int:
    from recruit_intel.analysis.profile import find_players
    from recruit_intel.models import managers, team_style

    con = db.connect(args.db)
    st = managers.stints(con, team_style.load())
    profs = managers.profiles(st)
    hits = find_players(profs.rename(columns={"manager": "name"}), args.manager)
    if len(hits) != 1:
        print(f"'{args.manager}' matched {len(hits)} managers: "
              f"{', '.join(hits.name.head(8)) or 'none'}")  # fmt: skip
        return 1
    prof = profs[profs.manager_id == hits.manager_id.iloc[0]].iloc[0]
    mine = st[st.manager_id == prof.manager_id].copy()
    mine["traits"] = [team_style.describe(r) for r in mine[team_style.STYLE_FEATURES].to_numpy()]
    mine["season_"] = mine.season.map(lambda s: f"{s}-{(s + 1) % 100:02d}")
    mine["strength"] = mine.xpts_pm.round(2)
    print(f"{prof.manager}: {prof.matches} top-five-league matches at {prof.clubs} club(s) "
          f"since {mine.season_.iloc[0]}\n")  # fmt: skip
    with pd.option_context(*[kv for pair in WIDE.items() for kv in pair]):
        print(mine[["team_name", "season_", "matches", "strength", "traits"]]
              .to_string(index=False))  # fmt: skip
    print(f"\nSignature (match-weighted, strength-adjusted): {managers.signature(prof)}")
    if prof.clubs == 1:
        print("  (one club only: can't yet tell his style from that squad's)")

    near = managers.similar_managers(profs, prof.manager_id, k=args.limit)
    near = near.assign(signature=[managers.signature(r) for _, r in near.iterrows()])
    print("\nManagers with the most similar profile (38+ matches):\n")
    with pd.option_context(*[kv for pair in WIDE.items() for kv in pair]):
        print(near[["manager", "matches", "clubs", "signature", "distance"]]
              .round(2).to_string(index=False))  # fmt: skip
    return 0


def cmd_style(args: argparse.Namespace) -> int:
    from recruit_intel.models import team_style

    con = db.connect(args.db)
    seasons = team_style.team_seasons(con)
    model = team_style.load()
    hits = _find_team(seasons, args.team)
    teams = hits.team_name.unique()
    if len(teams) != 1:
        print(f"'{args.team}' matched {len(teams)} teams: {', '.join(teams[:8]) or 'none'}")
        return 1
    team = hits.sort_values("season")
    team = team.assign(traits=model.traits(team))
    team["strength"] = team.xpts_pm.round(2)
    team["season_"] = team.season.map(lambda s: f"{s}-{(s + 1) % 100:02d}")
    print(f"{teams[0]}: style by season (strength = expected points per match)\n")
    with pd.option_context(*[kv for pair in WIDE.items() for kv in pair]):
        print(team[["season_", "competition_id", "matches", "strength", "traits"]]
              .tail(args.last).to_string(index=False))  # fmt: skip

    row = seasons.index.get_loc(team.index[-1])
    near = team_style.nearest_teams(seasons, model, row, k=args.limit)
    near = near.assign(traits=model.traits(near), strength=near.xpts_pm.round(2),
                       season_=near.season.map(lambda s: f"{s}-{(s + 1) % 100:02d}"))  # fmt: skip
    print(f"\nTeams that played most like {teams[0]} {team.season_.iloc[-1]} "
          f"(style map, strength removed):\n")  # fmt: skip
    with pd.option_context(*[kv for pair in WIDE.items() for kv in pair]):
        print(near[["team_name", "season_", "strength", "traits", "distance"]]
              .round(2).to_string(index=False))  # fmt: skip
    return 0


def _pool(con, n_seasons: int):
    from recruit_intel.search.pool import per90_pool

    return per90_pool(con, _seasons(con, n_seasons))


def cmd_similar(args: argparse.Namespace) -> int:
    from recruit_intel.analysis.profile import find_players, squad_roster
    from recruit_intel.models.embeddings import Embedder
    from recruit_intel.models.similarity import similar_players

    club = load_club(args.club)
    con = db.connect(args.db)
    pool = _pool(con, args.seasons)
    hits = find_players(pool, args.player)
    if len(hits) != 1:
        print(f"'{args.player}' matched {len(hits)} players: {', '.join(hits.name.head(8))}")
        return 1
    target = hits.iloc[0]
    exclude = set(squad_roster(con, club).player_id) if args.not_ours else None
    out = similar_players(pool, target.player_id, Embedder.load(), k=args.limit,
                          same_group=args.same_position, younger=args.younger,
                          cheaper=args.cheaper, exclude=exclude)  # fmt: skip
    value = "-" if pd.isna(target.value_eur) else f"EUR {target.value_eur / 1e6:.0f}M"
    print(f"Players like {target['name']} ({target.club}, {target.group}, age {target.age}, "
          f"{value})\n")  # fmt: skip
    out["value"] = (out.value_eur / 1e6).round(1)
    out["league"] = out.competition_id.str.split("-", n=1).str[-1]
    cols = ["name", "club", "league", "group", "age", "value", "distance"]
    with pd.option_context(*[kv for pair in WIDE.items() for kv in pair]):
        print(out[cols].round(2).to_string(index=False, na_rep="-"))
    return 0


def cmd_roles(args: argparse.Namespace) -> int:
    from recruit_intel.models.embeddings import Embedder
    from recruit_intel.models.role_discovery import describe, fit_mixture

    con = db.connect(args.db)
    pool = _pool(con, args.seasons)
    pool = pool[(pool.minutes >= 900) & pool.group.notna() & (pool.group != "GK")]
    pool = pool.dropna(subset=["npxg"]).reset_index(drop=True)
    vectors = Embedder.load().embed(pool)
    gm = fit_mixture(vectors)
    print(f"{gm.n_components} playing roles found among {len(pool)} outfield players "
          f"(number chosen by BIC)\n")  # fmt: skip
    with pd.option_context(*[kv for pair in WIDE.items() for kv in pair]):
        print(describe(pool, gm.predict(vectors)).to_string(index=False))
    return 0


def cmd_bargains(args: argparse.Namespace) -> int:
    from recruit_intel.analysis.profile import squad_roster
    from recruit_intel.models.value import bargains, value_model

    club = load_club(args.club)
    con = db.connect(args.db)
    valued, m = value_model(_pool(con, args.seasons), club.season_start_year + 1)
    print(f"Value model on {m['players']} players: R2 {m['r2_log_value']} on log value, "
          f"typical error {m['median_error']} (out-of-fold)\n")  # fmt: skip
    max_value = int(args.max_value * 1e6) if args.max_value else club.recruitment.max_value_eur
    top = bargains(valued, args.limit, group=args.group,
                   max_age=args.max_age or club.recruitment.max_age, max_value_eur=max_value,
                   exclude=set(squad_roster(con, club).player_id))  # fmt: skip
    top = top.assign(
        value=(top.value_eur / 1e6).round(1),
        expected=(top.expected_value_eur / 1e6).round(1),
        gap=(top.value_gap * 100).round(0).astype(int).astype(str) + "%",
        league=top.competition_id.str.split("-", n=1).str[-1],
    )
    cols = ["name", "club", "league", "group", "age", "contract_year", "value", "expected", "gap"]
    with pd.option_context(*[kv for pair in WIDE.items() for kv in pair]):
        print(top[cols].to_string(index=False, na_rep="-"))
    print("\nA lead to scout, not a verdict: market value also prices injuries, wages and form.")
    return 0


def register(sub) -> None:
    p = sub.add_parser("train", help="train embeddings and the league-translation model")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("similar", help="players who contribute like a given player")
    p.add_argument("player")
    p.add_argument("--younger", action="store_true")
    p.add_argument("--cheaper", action="store_true")
    p.add_argument("--same-position", action="store_true")
    p.add_argument("--not-ours", action="store_true", help="exclude the club's own squad")
    p.add_argument("--seasons", type=int, default=2)
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(func=cmd_similar)

    p = sub.add_parser("roles", help="discover playing roles by clustering embeddings")
    p.add_argument("--seasons", type=int, default=2)
    p.set_defaults(func=cmd_roles)

    p = sub.add_parser("style", help="a team's playing style by season, and teams like it")
    p.add_argument("team")
    p.add_argument("--last", type=int, default=6, help="how many seasons to show")
    p.add_argument("--limit", type=int, default=8)
    p.set_defaults(func=cmd_style)

    p = sub.add_parser("manager", help="a manager's style across his clubs, and similar ones")
    p.add_argument("manager")
    p.add_argument("--limit", type=int, default=6)
    p.set_defaults(func=cmd_manager)

    p = sub.add_parser("bargains", help="players valued below what comparable players cost")
    p.add_argument("--group", help="position group, e.g. CB, W, ST")
    p.add_argument("--max-age", type=int)
    p.add_argument("--max-value", type=float, help="EUR millions")
    p.add_argument("--seasons", type=int, default=2)
    p.add_argument("--limit", type=int, default=15)
    p.set_defaults(func=cmd_bargains)
