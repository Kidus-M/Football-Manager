"""Best XI and depth chart for a formation.

Every squad player is scored for every slot he is eligible for (his position
group is one the slot accepts) on that slot's role. Fits from small samples
are shrunk toward 50 in proportion to minutes, so a 76-minute cameo can't
outrank a regular on noise alone. The XI is the one-player-per-slot
assignment that maximises the total shrunk fit.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from recruit_intel.analysis.profile import role_fit
from recruit_intel.config import Role, Slot


@dataclass
class Candidate:
    player_id: str
    name: str
    fit: float  # raw role fit, 0-100
    adjusted: float  # shrunk toward 50 by sample size
    minutes: int


@dataclass
class SlotDepth:
    slot: Slot
    starter: Candidate | None
    backups: list[Candidate] = field(default_factory=list)  # best first, non-starters
    # Eligible squad players with no rating (no minutes yet): unknown, not absent.
    unrated: list[str] = field(default_factory=list)


def shrink(fit: float, minutes: float, min_minutes: float) -> float:
    weight = min(1.0, minutes / min_minutes) if min_minutes else 1.0
    return 50 + (fit - 50) * weight


def slot_candidates(
    squad: pd.DataFrame, slot: Slot, role: Role, min_minutes: int
) -> list[Candidate]:
    """Eligible squad players with enough data to score, best first."""
    out = []
    for _, row in squad.iterrows():
        fit = role_fit(row, role, groups=slot.groups)
        if fit is None:
            continue
        adjusted = shrink(fit.score, row.minutes, min_minutes)
        out.append(Candidate(row.player_id, row["name"], fit.score, adjusted, int(row.minutes)))
    return sorted(out, key=lambda c: c.adjusted, reverse=True)


def best_xi(candidates: dict[str, list[Candidate]]) -> dict[str, Candidate]:
    """Slot key -> starter, maximising total adjusted fit, one slot per player."""
    players = sorted({c.player_id for cands in candidates.values() for c in cands})
    if not players:
        return {}
    slots = list(candidates)
    col = {p: j for j, p in enumerate(players)}
    # Ineligible pairs get a large negative score so they're only used when a
    # slot has no eligible player left, and are then dropped below.
    score = np.full((len(slots), len(players)), -1e6)
    for i, slot in enumerate(slots):
        for c in candidates[slot]:
            score[i, col[c.player_id]] = c.adjusted
    rows, cols = linear_sum_assignment(score, maximize=True)
    xi = {}
    for i, j in zip(rows, cols, strict=True):
        if score[i, j] > -1e6:
            xi[slots[i]] = next(c for c in candidates[slots[i]] if c.player_id == players[j])
    return xi


def depth_chart(
    squad: pd.DataFrame,
    slots: list[Slot],
    roles: dict[str, Role],
    min_minutes: int,
    roster: pd.DataFrame | None = None,
) -> list[SlotDepth]:
    """`squad` holds rated players (percentile rows); `roster` (player_id, name,
    group) is the full squad, so eligible players with no rating are listed."""
    candidates = {s.key: slot_candidates(squad, s, roles[s.role], min_minutes) for s in slots}
    xi = best_xi(candidates)
    starters = {c.player_id for c in xi.values()}
    chart = []
    for s in slots:
        rated = {c.player_id for c in candidates[s.key]}
        unrated = (
            []
            if roster is None
            else [
                r.name
                for r in roster.itertuples()
                if r.group in s.groups and r.player_id not in rated
            ]  # fmt: skip
        )
        chart.append(
            SlotDepth(
                slot=s,
                starter=xi.get(s.key),
                backups=[c for c in candidates[s.key] if c.player_id not in starters],
                unrated=unrated,
            )
        )
    return chart
