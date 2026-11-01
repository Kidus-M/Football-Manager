"""Tactics layer: how the squad should play, and what-ifs for tactical settings.

Rule-based by design (config/tactics.yaml): it encodes football reasoning
over the squad's percentiles rather than learning from outcomes, so every
recommendation can be traced to the numbers behind it.

Three parts:
- settings: a tactical choice (press high, build short, play wide) reweights
  the role library; re-running the best XI and gap report under it shows who
  fits the new setting and which gaps open. A rule-based simulation, not a
  prediction of results.
- levers: what the current XI is suited to (press, build-up, finishing),
  compared with how the team actually plays this season.
- instructions: position-by-position advice from each starter's percentiles.
"""

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pandas as pd
import yaml

from recruit_intel import CONFIG_DIR
from recruit_intel.config import Role


@cache
def load_config(path: Path = CONFIG_DIR / "tactics.yaml") -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def reweight(roles: dict[str, Role], choices: dict[str, str], config: dict | None = None
             ) -> dict[str, Role]:  # fmt: skip
    """The role library under tactical choices, e.g. {"press": "high"}."""
    config = config or load_config()
    out = dict(roles)
    for lever, option in choices.items():
        try:
            setting = config["settings"][lever][option]
        except KeyError:
            options = {k: list(v) for k, v in config["settings"].items()}
            raise ValueError(f"unknown setting {lever}={option}; options: {options}") from None
        groups = set(setting["groups"])
        for key, role in out.items():
            if not groups & set(role.groups):
                continue
            weights = dict(role.weights)
            for metric, mult in setting["boost"].items():
                if metric in weights:
                    weights[metric] *= mult
                elif mult > 1:
                    weights[metric] = config["added_weight"] * (mult - 1)
            out[key] = role.model_copy(update={"weights": weights})
    return out


def instructions(starters: pd.DataFrame, config: dict | None = None) -> list[tuple[str, str]]:
    """(player name, instruction) for every rule a starter satisfies, in XI
    order. Missing metrics never satisfy a condition. A rule marked `once`
    (a job only one player can do, like dropping between the centre-backs)
    goes to the qualifying starter with the highest value on its first
    condition. Starters flagged unqualified get a small-sample note."""
    config = config or load_config()
    winners: dict[int, str] = {}
    for i, rule in enumerate(config["instructions"]):
        if not rule.get("once"):
            continue
        first = next(iter(rule["when"]))
        eligible = starters[starters.group.isin(rule["groups"])]
        eligible = eligible[[_rule_holds(r, rule) for _, r in eligible.iterrows()]]
        if not eligible.empty:
            winners[i] = eligible.sort_values(first, ascending=False).iloc[0]["name"]

    out = []
    for _, row in starters.iterrows():
        for i, rule in enumerate(config["instructions"]):
            if row["group"] not in rule["groups"] or not _rule_holds(row, rule):
                continue
            if rule.get("once") and winners.get(i) != row["name"]:
                continue
            text = rule["say"].format(name=row["name"])
            if row.get("qualified") is False:
                text += f" (small sample: {int(row['minutes'])} min)"
            out.append((row["name"], text))
    return out


def _rule_holds(row: pd.Series, rule: dict) -> bool:
    return all(_holds(row.get(metric), cond) for metric, cond in rule["when"].items())


def _holds(value, condition: str) -> bool:
    if value is None or pd.isna(value):
        return False
    op, number = condition[:2], float(condition[2:])
    if op == ">=":
        return value >= number
    if op == "<=":
        return value <= number
    raise ValueError(f"condition must start with >= or <=: {condition!r}")


@dataclass
class SlotChange:
    slot: str
    before: str | None
    before_fit: float | None
    after: str | None
    after_fit: float | None
    severity_before: str | None
    severity_after: str | None


@dataclass
class LeverReading:
    lever: str
    squad: float | None  # mean percentile of the relevant starters
    verdict: str  # "high", "mid", "low" or "unknown"
    best: list[str]  # top contributors, "Name 85"
    worst: list[str]
    style_z: float | None = None  # how the team plays now (+ = more of it), adjusted
    manager_z: float | None = None
    advice: str = ""


# How each lever reads from a strength-adjusted style vector: (feature, sign),
# sign -1 where a lower value means more of the lever (ppda: fewer passes
# allowed per action = more pressing).
STYLE_SIGNAL = {"press": ("ppda", -1), "buildup": ("buildup_share", 1), "finishing": ("npxg", 1)}
STYLE_HIGH = 0.5  # adjusted z beyond which a team clearly leans one way


def lever_readings(starters: pd.DataFrame, config: dict | None = None,
                   min_minutes: int | None = None) -> list[LeverReading]:  # fmt: skip
    """starters: percentile rows of the XI (one per player, with `group`).
    With `min_minutes`, each player's percentile is shrunk toward 50 by his
    sample size, so a cameo can't swing a lever."""
    config = config or load_config()
    out = []
    for lever, spec in config["levers"].items():
        rows = starters[starters.group.isin(spec["slots_groups"])]
        metrics = [m for m in spec["metrics"] if m in rows]
        if rows.empty or not metrics:
            out.append(LeverReading(lever, None, "unknown", [], []))
            continue
        per_player = rows[metrics].mean(axis=1, skipna=True)
        if min_minutes:
            weight = (rows["minutes"] / min_minutes).clip(upper=1.0)
            per_player = 50 + (per_player - 50) * weight
        per_player.index = rows["name"].to_numpy()
        per_player = per_player.dropna().sort_values(ascending=False)
        value = float(per_player.mean()) if len(per_player) else None
        verdict = ("unknown" if value is None else "high" if value >= spec["high_at"]
                   else "low" if value <= spec["low_at"] else "mid")  # fmt: skip
        fmt = [f"{n} {v:.0f}" for n, v in per_player.items()]
        # With fewer than four players, best and worst lists would overlap.
        worst = fmt[-2:][::-1] if len(fmt) >= 4 else []
        out.append(LeverReading(lever, value, verdict, fmt[:2], worst))
    return out


def ordinal(n: float) -> str:
    n = round(n)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def style_lean(z: float | None) -> str:
    if z is None:
        return "unknown"
    return "high" if z >= STYLE_HIGH else "low" if z <= -STYLE_HIGH else "mid"


LEVER_WORDS = {
    "press": ("presses", "a higher press", "a lower block"),
    "buildup": ("builds up patiently", "patient build-up", "a more direct game"),
    "finishing": ("creates chances", "more chances for the forwards", "fewer, better chances"),
}


def advise(r: LeverReading) -> str:
    """One line comparing what the squad supports with how the team plays."""
    verb, more, less = LEVER_WORDS.get(r.lever, (r.lever, "more", "less"))
    squad = f"squad {r.squad:.0f}th pct" if r.squad is not None else "squad unknown"
    team = style_lean(r.style_z)
    if r.verdict == "unknown" or team == "unknown":
        return f"{squad}; no style comparison available"
    if r.verdict == "high" and team != "high":
        return f"{squad}: the players support {more} than the team currently plays"
    if r.verdict == "low" and team == "high":
        return f"{squad}: the team {verb} beyond what these players' numbers support"
    if r.verdict == "low":
        return f"{squad}: consistent with {less}; recruit here before asking for more"
    return f"{squad}: style and squad are aligned"


MIN_FIT_CHANGE = 3  # smaller moves are rounding noise, not a tactical effect


def _missing(x) -> bool:
    return x is None or (isinstance(x, float) and pd.isna(x))


def _same(a, b) -> bool:
    return (_missing(a) and _missing(b)) or a == b


def compare(base: pd.DataFrame, what_if: pd.DataFrame) -> list[SlotChange]:
    """Slots whose starter, severity or fit (by MIN_FIT_CHANGE+) changes
    between two gap reports of the same formation."""
    changes = []
    for b, w in zip(base.itertuples(), what_if.itertuples(), strict=True):
        fit_moved = (not _missing(b.fit) and not _missing(w.fit)
                     and abs(w.fit - b.fit) >= MIN_FIT_CHANGE)  # fmt: skip
        if not _same(b.starter, w.starter) or not _same(b.severity, w.severity) or fit_moved:
            changes.append(SlotChange(
                b.slot, b.starter, b.fit, w.starter, w.fit,
                None if _missing(b.severity) else b.severity,
                None if _missing(w.severity) else w.severity,
            ))  # fmt: skip
    return changes
