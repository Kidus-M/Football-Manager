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
    print("\nSaved to data/models/; `recruit search` now uses the learned league factors.")
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

    p = sub.add_parser("bargains", help="players valued below what comparable players cost")
    p.add_argument("--group", help="position group, e.g. CB, W, ST")
    p.add_argument("--max-age", type=int)
    p.add_argument("--max-value", type=float, help="EUR millions")
    p.add_argument("--seasons", type=int, default=2)
    p.add_argument("--limit", type=int, default=15)
    p.set_defaults(func=cmd_bargains)
