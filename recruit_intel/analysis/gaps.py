"""Squad-vs-formation gap report: which slots need recruitment, and why.

For each slot of the club's formation the report looks at the starter the
best XI assigns, the cover behind him, and the starter's age and contract,
and turns them into flagged issues with a severity. Fits are the sample-size
shrunk ones from analysis/lineup.py, so an early-season cameo neither hides
nor creates a gap on its own.
"""

from dataclasses import dataclass
from datetime import date

import duckdb
import pandas as pd

from recruit_intel.analysis.lineup import SlotDepth, depth_chart
from recruit_intel.analysis.profile import squad_percentiles, squad_roster
from recruit_intel.analysis.squad import market_snapshot
from recruit_intel.config import ClubConfig, lineup, load_roles

SEVERITY = {"critical": 3, "high": 2, "medium": 1, "low": 0}

CRITICAL_FIT = 45  # starter this far below an average peer in the role
HIGH_FIT = 55
UPGRADE_FIT = 65
COVER_FIT = 50  # a backup below this isn't real cover
AGEING = 31


@dataclass(frozen=True)
class Issue:
    severity: str
    text: str


def slot_issues(depth: SlotDepth, age: int | None, contract_year: int | None,
                season_start: int) -> list[Issue]:  # fmt: skip
    starter = depth.starter
    unrated = ", ".join(depth.unrated)
    if starter is None:
        if depth.unrated:
            return [Issue("high", f"no rated player; unproven: {unrated}")]
        return [Issue("critical", "no squad player can play this slot")]
    issues = []
    if starter.adjusted < CRITICAL_FIT:
        issues.append(Issue("critical", f"starter fit {starter.adjusted:.0f}: well below peers"))
    elif starter.adjusted < HIGH_FIT:
        issues.append(Issue("high", f"starter fit {starter.adjusted:.0f}: below average"))
    elif starter.adjusted < UPGRADE_FIT:
        issues.append(Issue("medium", f"starter fit {starter.adjusted:.0f}: upgrade possible"))
    if not any(b.adjusted >= COVER_FIT for b in depth.backups):
        if depth.unrated:
            issues.append(Issue("low", f"cover unproven: {unrated} unrated"))
        else:
            issues.append(Issue("medium", "no adequate cover"))
    if contract_year is not None and contract_year <= season_start + 1:
        issues.append(Issue("medium", f"starter's contract ends {contract_year}"))
    if age is not None and age >= AGEING:
        issues.append(Issue("low", f"starter is {age}"))
    return issues


def gap_report(
    con: duckdb.DuckDBPyConnection, club: ClubConfig, season: int, formation: str | None = None
) -> tuple[pd.DataFrame, list[SlotDepth]]:
    """One row per slot (formation order) plus the underlying depth chart."""
    squad = squad_percentiles(con, club, season)
    slots = lineup(club, formation)
    roles = load_roles()
    roster = squad_roster(con, club)
    chart = depth_chart(squad, slots, roles, squad.attrs["min_minutes"], roster)

    starter_ids = [d.starter.player_id for d in chart if d.starter]
    info = (
        market_snapshot(con, starter_ids)
        .merge(
            con.execute(
                "SELECT player_id, birth_date FROM players WHERE player_id IN (SELECT unnest(?))",
                [starter_ids],
            ).df(),
            on="player_id",
            how="outer",
        )
        .set_index("player_id")
    )

    today = pd.Timestamp(date.today())
    rows = []
    for depth in chart:
        s = depth.starter
        age = contract = None
        if s is not None and s.player_id in info.index:
            born, end = info.at[s.player_id, "birth_date"], info.at[s.player_id, "contract_end"]
            age = None if pd.isna(born) else int((today - pd.Timestamp(born)).days // 365.25)
            contract = None if pd.isna(end) else pd.Timestamp(end).year
        issues = slot_issues(depth, age, contract, season)
        cover = next((b for b in depth.backups if b.adjusted >= COVER_FIT), None)
        rows.append(
            {
                "slot": depth.slot.key,
                "role": roles[depth.slot.role].label,
                "starter": s.name if s else None,
                "fit": round(s.adjusted) if s else None,
                "age": age,
                "contract": contract,
                "cover": f"{cover.name} ({cover.adjusted:.0f})" if cover else None,
                "severity": max((i.severity for i in issues), key=SEVERITY.get, default=None),
                "issues": [i.text for i in issues],
            }
        )
    return pd.DataFrame(rows), chart


def priorities(report: pd.DataFrame) -> pd.DataFrame:
    """Slots with issues, worst first: the answer to 'what should we buy?'."""
    flagged = report[report.severity.notna()].copy()
    flagged["_rank"] = flagged.severity.map(SEVERITY)
    return flagged.sort_values(["_rank", "fit"], ascending=[False, True]).drop(columns="_rank")
