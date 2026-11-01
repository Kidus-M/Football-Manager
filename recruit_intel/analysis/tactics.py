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


@dataclass
class SlotChange:
    slot: str
    before: str | None
    before_fit: float | None
    after: str | None
    after_fit: float | None
    severity_before: str | None
    severity_after: str | None


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
