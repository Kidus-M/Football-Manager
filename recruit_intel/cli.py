"""Command-line entry point: `recruit ingest | match | squad | profile | gaps | search`."""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from recruit_intel import DATA_DIR, db
from recruit_intel.analysis import gaps, profile, radar
from recruit_intel.analysis.squad import current_squad
from recruit_intel.config import load_club, load_roles
from recruit_intel.features.positions import GROUPS
from recruit_intel.identity import run as identity_run
from recruit_intel.ingest import clubelo, fpl, transfermarkt, understat
from recruit_intel.ingest.http import DAY, CachedClient
from recruit_intel.search import rank as rank_mod
from recruit_intel.search.pool import candidate_pool, league_strength


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


def _season_label(season: int) -> str:
    return f"{season}-{(season + 1) % 100:02d}"


def _latest_season(con) -> int:
    return con.execute(
        "SELECT max(season) FROM player_season_stats WHERE source = 'fpl'"
    ).fetchone()[0]


def _print_squad_profile(con, club, season) -> None:
    table = profile.squad_profile(con, club, season)
    table["minutes"] = [
        f"{m}" if q else f"{m}*" for m, q in zip(table.minutes, table.qualified, strict=True)
    ]
    for col in ("strengths", "weaknesses"):
        table[col] = table[col].map(lambda xs: ", ".join(xs) or "-")
    print(
        f"{club.club} squad profile, {_season_label(season)}: percentiles vs "
        f"{club.peer_group} players in the same position group\n"
        f"* under {table.attrs['min_minutes']} minutes: ranked, but a small sample\n"
    )
    cols = ["name", "group", "minutes", "best_role", "role_fit", "strengths", "weaknesses"]
    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        print(table[cols].to_string(index=False, na_rep="-"))


def _print_player(row: pd.Series, min_minutes: int) -> None:
    group = GROUPS[row["group"]]
    sample = "" if row.qualified else f"  (under {min_minutes} min: small sample)"
    print(f"\n{row['name']}  ·  {row.detailed_position or row['group']}  ·  "
          f"{int(row.minutes)} min{sample}")  # fmt: skip
    print(f"vs {int(row.peers)} {group['label'].lower()}")
    for m in group["metrics"]:
        pct = row.get(m)
        bar = "-" if pd.isna(pct) else "█" * round(pct / 5)
        value = "n/a" if pd.isna(pct) else f"{pct:3.0f}"
        print(f"  {profile.label(m):<36} {value}  {bar}")
    strengths, weaknesses = profile.strengths_weaknesses(row)
    print("  strengths: " + (", ".join(profile.label(m) for m in strengths) or "-"))
    print("  weaknesses: " + (", ".join(profile.label(m) for m in weaknesses) or "-"))
    roles = load_roles()
    fits = profile.role_fits(row, roles)[:3]
    print("  roles: " + (", ".join(f"{roles[f.role].label} {f.score:.0f}" for f in fits) or "-"))


def cmd_gaps(args: argparse.Namespace) -> int:
    club = load_club(args.club)
    con = db.connect(args.db)
    season = args.season or _latest_season(con)
    report, chart = gaps.gap_report(con, club, season, args.formation)
    formation = args.formation or club.formation

    print(f"{club.club} · {formation} · {_season_label(season)}")
    print("fit = role fit vs peers, shrunk toward 50 for small samples\n")
    table = report.assign(issues=report.issues.map(lambda xs: "; ".join(xs) or "-"))
    cols = ["slot", "role", "starter", "fit", "age", "contract", "cover", "issues"]
    with pd.option_context("display.width", 250, "display.max_colwidth", 70):
        print(table[cols].to_string(index=False, na_rep="-"))

    top = gaps.priorities(report)
    print("\nRecruitment priorities")
    if top.empty:
        print("  none: every slot has a solid starter and cover")
    for i, row in enumerate(top.itertuples(), 1):
        print(f"  {i}. {row.slot} ({row.role}) [{row.severity}]: {'; '.join(row.issues)}")

    if args.depth:
        print("\nDepth chart")
        for d in chart:
            names = [f"{c.name} {c.adjusted:.0f}" for c in d.backups[:4]]
            starter = f"{d.starter.name} {d.starter.adjusted:.0f}" if d.starter else "-"
            print(f"  {d.slot.key:<4} {starter:<28} | " + (", ".join(names) or "-"))
    return 0


def _search_slot(pool, slot, starter, club, args, roster_ids, strength) -> None:
    role = load_roles()[slot.role]
    pool_countries = args.leagues or club.recruitment.pool
    competitions = [transfermarkt.LEAGUES[c][1] for c in pool_countries
                    if c in transfermarkt.LEAGUES]  # fmt: skip
    max_value = int(args.max_value * 1e6) if args.max_value else club.recruitment.max_value_eur
    filters = rank_mod.Filters(
        groups=slot.groups,
        competitions=competitions,
        max_age=args.max_age or club.recruitment.max_age,
        max_value_eur=max_value,
        contract_until=args.contract_until,
        min_minutes=args.min_minutes or club.recruitment.min_minutes,
        exclude=roster_ids,
    )
    window = club.season_start_year + 1
    ranked = rank_mod.rank(pool, role, filters, strength, window, budget=max_value)

    # The current starter, scored on the same basis (same window, same discounts).
    base = None
    if starter is not None:
        row = pool[pool.player_id == starter.player_id]
        fits = rank_mod.fit_scores(row, role, slot.groups, strength)
        base = None if fits.empty else fits.fit.iloc[0]
    base_txt = f"{starter.name} {base:.0f}" if starter and base is not None else "-"
    print(f"\n{slot.key} · {role.label} · current starter: {base_txt}")
    if ranked.empty:
        print("  no candidates match the filters")
        return
    top = ranked.head(args.limit).copy()
    top["vs_starter"] = (top.fit - base).round(0) if base is not None else None
    top["value"] = (top.value_eur / 1e6).round(1)
    cols = ["name", "club", "age", "value", "score", "vs_starter", "why"]
    with pd.option_context("display.width", 250, "display.max_colwidth", 90):
        print(top[cols].round(1).to_string(index=False, na_rep="-"))


def cmd_search(args: argparse.Namespace) -> int:
    club = load_club(args.club)
    con = db.connect(args.db)
    season = _latest_season(con)
    report, chart = gaps.gap_report(con, club, season, args.formation)
    by_slot = {d.slot.key: d for d in chart}

    if args.slot:
        if args.slot not in by_slot:
            print(f"no slot {args.slot} in {args.formation or club.formation}: {list(by_slot)}")
            return 1
        slots = [args.slot]
    else:
        slots = list(gaps.priorities(report).slot.head(args.gaps))

    seasons = list(range(season - args.seasons + 1, season + 1))
    pool = candidate_pool(con, seasons)
    roster_ids = set(profile.squad_roster(con, club).player_id)
    print(f"{club.club} · market search · seasons {seasons[0]}-{_season_label(seasons[-1])}")
    print("score = 65% fit (league-discounted) + 15% age + 10% value + 10% contract")
    for key in slots:
        depth = by_slot[key]
        _search_slot(pool, depth.slot, depth.starter, club, args, roster_ids, league_strength())
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    club = load_club(args.club)
    con = db.connect(args.db)
    season = args.season or _latest_season(con)
    if not args.players:
        _print_squad_profile(con, club, season)
        return 0

    pct = profile.peer_percentiles(con, club, season)
    rows = []
    for query in args.players:
        hits = profile.find_players(pct, query)
        if len(hits) != 1:
            names = ", ".join(hits["name"].head(8)) if len(hits) else "none"
            print(f"'{query}' matched {len(hits)} {club.peer_group} players: {names}")
            return 1
        rows.append(hits.iloc[0])
    for row in rows:
        _print_player(row, pct.attrs["min_minutes"])

    if args.radar:
        group = rows[0]["group"]
        metrics = GROUPS[group]["metrics"]
        players = [
            (r["name"], {m: (None if pd.isna(r.get(m)) else r[m]) for m in metrics}) for r in rows
        ]
        title = " vs ".join(r["name"] for r in rows)
        peers = GROUPS[group]["label"].lower()
        subtitle = (
            f"{club.peer_group} {_season_label(season)} · percentile vs {peers} "
            f"(min {pct.attrs['min_minutes']} min)"
        )
        accent = club.theme.accent if club.theme else "#DA291C"
        path = Path(args.radar)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(radar.render(players, metrics, title, subtitle, accent), encoding="utf-8")
        print(f"\nradar written to {path}")
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

    p_profile = sub.add_parser(
        "profile", help="squad strengths/weaknesses, or percentiles for named players"
    )
    p_profile.add_argument("players", nargs="*", help="player name(s); omit for the squad")
    p_profile.add_argument("--season", type=int)
    p_profile.add_argument(
        "--radar",
        nargs="?",
        const=str(DATA_DIR / "radars" / "radar.svg"),
        help="write an SVG radar (overlaying up to 3 players)",
    )
    p_profile.set_defaults(func=cmd_profile)

    p_gaps = sub.add_parser("gaps", help="best XI, depth and recruitment gaps for a formation")
    p_gaps.add_argument("--formation", help="override the club's formation, e.g. 3-4-2-1")
    p_gaps.add_argument("--season", type=int)
    p_gaps.add_argument("--depth", action="store_true", help="also print the depth chart")
    p_gaps.set_defaults(func=cmd_gaps)

    p_search = sub.add_parser("search", help="rank market candidates for a slot or the top gaps")
    target = p_search.add_mutually_exclusive_group()
    target.add_argument("--slot", help="formation slot, e.g. ST")
    target.add_argument("--gaps", type=int, default=3, help="search the N top-priority slots")
    p_search.add_argument("--formation")
    p_search.add_argument("--leagues", nargs="+", metavar="COUNTRY", help="e.g. ENG ESP")
    p_search.add_argument("--max-age", type=int, help="default: club recruitment.max_age")
    p_search.add_argument("--max-value", type=float, help="EUR millions")
    p_search.add_argument("--contract-until", type=int, help="contract ends by this year")
    p_search.add_argument("--min-minutes", type=int, help="default: club recruitment.min_minutes")
    p_search.add_argument("--seasons", type=int, default=2, help="season window for stats")
    p_search.add_argument("--limit", type=int, default=8)
    p_search.set_defaults(func=cmd_search)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
