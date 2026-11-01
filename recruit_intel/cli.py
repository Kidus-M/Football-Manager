"""Command-line entry point: `recruit ingest`, `recruit match` and `recruit squad`."""

import argparse
import sys
from collections.abc import Callable

import pandas as pd

from recruit_intel import db
from recruit_intel.analysis.squad import current_squad
from recruit_intel.config import load_club
from recruit_intel.identity import run as identity_run
from recruit_intel.ingest import clubelo, fpl, transfermarkt, understat
from recruit_intel.ingest.http import DAY, CachedClient


class _Runner:
    """Runs adapters one by one; a failing source is reported and skipped, never fatal."""

    def __init__(self, con) -> None:
        self.con = con
        self.ok = 0
        self.total = 0

    def __call__(self, name: str, fn: Callable[[], dict[str, pd.DataFrame]]):
        self.total += 1
        try:
            tables = fn()
        except Exception as exc:  # noqa: BLE001 - any scraper failure is isolated
            print(f"  ✗ {name}: {type(exc).__name__}: {exc}")
            return None
        counts = {t: db.upsert(self.con, t, df) for t, df in tables.items()}
        print(f"  ✓ {name}: " + ", ".join(f"{n} {t}" for t, n in counts.items()))
        self.ok += 1
        return tables


def cmd_ingest(args: argparse.Namespace) -> int:
    club = load_club(args.club)
    con = db.connect(args.db)
    run = _Runner(con)

    with CachedClient(fpl.SOURCE) as client:
        bootstrap = fpl.fetch_bootstrap(client)
    current = fpl.season_of(bootstrap)
    seasons = range(current - args.seasons + 1, current + 1)
    print(f"{club.club}: ingesting seasons {seasons.start}-{current}")

    run("fpl", lambda: fpl.parse(bootstrap))

    with understat.make_client() as us_client:
        # Oldest season first: source_players keeps whichever row is written
        # last, so a player's current club must come from the newest season.
        for season in seasons:
            for league in understat.leagues_for_pool(club.recruitment.pool):
                run(
                    f"understat {league} {season}",
                    lambda lg=league, s=season: understat.ingest(lg, s, current, us_client),
                )

    run("clubelo", lambda: {"team_elo": clubelo.ingest()})

    with transfermarkt.make_client(ttl=7 * DAY) as tm_client:
        own_id = str(club.source_name("transfermarkt_id"))
        run(
            f"transfermarkt {club.club}",
            lambda: transfermarkt.fetch_squad(tm_client, own_id, current),
        )
        if args.market_pool is not None:
            countries = args.market_pool or club.recruitment.pool
            for country in countries:
                if country not in transfermarkt.LEAGUES:
                    print(f"  ✗ transfermarkt {country}: no league mapping")
                    continue
                listing = run(
                    f"transfermarkt {country} clubs",
                    lambda c=country: {
                        "teams": transfermarkt.fetch_league_clubs(tm_client, c, current)
                    },
                )
                if listing is None:
                    continue
                for team in listing["teams"].itertuples():
                    if team.source_team_id == own_id:
                        continue
                    run(
                        f"transfermarkt {team.name}",
                        lambda t=team.source_team_id: transfermarkt.fetch_squad(
                            tm_client, t, current
                        ),
                    )

    print(f"{run.ok}/{run.total} sources loaded into {args.db}")
    _print_match_report(con)
    return 0 if run.ok else 1


def _print_match_report(con) -> None:
    print("\nIdentity matching")
    print(identity_run.rebuild(con).to_string(index=False))


def cmd_match(args: argparse.Namespace) -> int:
    _print_match_report(db.connect(args.db))
    return 0


def cmd_squad(args: argparse.Namespace) -> int:
    club = load_club(args.club)
    con = db.connect(args.db)
    season = (
        args.season
        or con.execute(
            "SELECT max(season) FROM player_season_stats WHERE source = 'fpl'"
        ).fetchone()[0]
    )
    squad = current_squad(con, club, season)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(f"{club.club} squad, {season}-{(season + 1) % 100:02d} (per 90)\n")
        print(squad.round(2).to_string(index=False, na_rep="-"))
    return 0


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="recruit")
    parser.add_argument("--club", default="man_utd", help="club config file stem")
    parser.add_argument("--db", default=str(db.DEFAULT_DB))
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="fetch all sources into DuckDB")
    p_ingest.add_argument("--seasons", type=int, default=2, help="how many seasons back")
    p_ingest.add_argument(
        "--market-pool",
        nargs="*",
        metavar="COUNTRY",
        help="also scrape Transfermarkt squads for these countries (no value: the club's "
        "whole recruitment pool). Slow on first run: ~20 pages per league at 4s each",
    )
    p_ingest.set_defaults(func=cmd_ingest)

    p_match = sub.add_parser("match", help="rebuild canonical player ids across sources")
    p_match.set_defaults(func=cmd_match)

    p_squad = sub.add_parser("squad", help="print the current squad with per-90 stats")
    p_squad.add_argument("--season", type=int)
    p_squad.set_defaults(func=cmd_squad)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
