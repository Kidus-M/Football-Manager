"""Command-line entry point: `recruit ingest` and `recruit squad`."""

import argparse
import sys
from collections.abc import Callable

import pandas as pd

from recruit_intel import db
from recruit_intel.analysis.squad import current_squad
from recruit_intel.config import load_club
from recruit_intel.ingest import clubelo, fpl, understat
from recruit_intel.ingest.http import CachedClient


def _run_source(name: str, fn: Callable[[], dict[str, pd.DataFrame]], con) -> bool:
    """Run one adapter; a failing source is reported and skipped, never fatal."""
    try:
        tables = fn()
    except Exception as exc:  # noqa: BLE001 - any scraper failure is isolated
        print(f"  ✗ {name}: {type(exc).__name__}: {exc}")
        return False
    counts = {t: db.upsert(con, t, df) for t, df in tables.items()}
    print(f"  ✓ {name}: " + ", ".join(f"{n} {t}" for t, n in counts.items()))
    return True


def cmd_ingest(args: argparse.Namespace) -> int:
    club = load_club(args.club)
    con = db.connect(args.db)

    with CachedClient(fpl.SOURCE) as client:
        bootstrap = fpl.fetch_bootstrap(client)
    current = fpl.season_of(bootstrap)
    seasons = range(current - args.seasons + 1, current + 1)
    print(f"{club.club}: ingesting seasons {seasons.start}-{current}")

    jobs: list[tuple[str, Callable]] = [("fpl", lambda: fpl.parse(bootstrap))]
    with understat.make_client() as us_client:
        # Oldest season first: source_players keeps whichever row is written
        # last, so a player's current club must come from the newest season.
        for season in seasons:
            for league in understat.leagues_for_pool(club.recruitment.pool):
                jobs.append(
                    (
                        f"understat {league} {season}",
                        lambda lg=league, s=season: understat.ingest(lg, s, current, us_client),
                    )
                )
        jobs.append(("clubelo", lambda: {"team_elo": clubelo.ingest()}))
        ok = sum(_run_source(name, fn, con) for name, fn in jobs)

    print(f"{ok}/{len(jobs)} sources loaded into {args.db}")
    return 0 if ok else 1


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
    p_ingest.set_defaults(func=cmd_ingest)

    p_squad = sub.add_parser("squad", help="print the current squad with per-90 stats")
    p_squad.add_argument("--season", type=int)
    p_squad.set_defaults(func=cmd_squad)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
