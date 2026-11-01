"""Club and role configuration loading.

A club is one YAML file under config/clubs/. Nothing outside this module and the
YAML files knows which club is being served.
"""

from functools import cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from recruit_intel import CONFIG_DIR
from recruit_intel.features.metrics import METRICS

POSITIONS = {"GK", "DEF", "MID", "FWD"}


def _tbd_to_none(value: Any) -> Any:
    return None if value == "TBD" else value


class Recruitment(BaseModel):
    pool: list[str]
    max_age: int
    max_value_eur: int | None = None
    min_minutes: int = 900

    _tbd = field_validator("max_value_eur", mode="before")(_tbd_to_none)


class Theme(BaseModel):
    accent: str
    secondary: str | None = None
    ink: str | None = None


class ClubConfig(BaseModel):
    club: str
    competition: str
    season: str
    formation: str | None = None
    roles: dict[str, str]
    recruitment: Recruitment
    peer_group: str
    source_names: dict[str, str | int] = Field(default_factory=dict)
    theme: Theme | None = None

    _tbd = field_validator("formation", mode="before")(_tbd_to_none)

    @property
    def season_start_year(self) -> int:
        """'2026-27' -> 2026."""
        return int(self.season.split("-")[0])

    def source_name(self, source: str) -> str | int:
        try:
            return self.source_names[source]
        except KeyError:
            raise KeyError(f"{self.club} has no source_names.{source} in its config") from None


class Role(BaseModel):
    key: str
    label: str
    positions: list[str]
    weights: dict[str, float]

    @model_validator(mode="after")
    def _check(self) -> "Role":
        unknown = set(self.weights) - set(METRICS)
        if unknown:
            raise ValueError(f"role {self.key}: unknown metrics {sorted(unknown)}")
        bad_pos = set(self.positions) - POSITIONS
        if bad_pos:
            raise ValueError(f"role {self.key}: unknown positions {sorted(bad_pos)}")
        return self


def _read_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


@cache
def load_roles(path: Path = CONFIG_DIR / "roles.yaml") -> dict[str, Role]:
    raw = _read_yaml(path)
    return {key: Role(key=key, **body) for key, body in raw.items()}


def load_club(name: str, clubs_dir: Path = CONFIG_DIR / "clubs") -> ClubConfig:
    """Load a club by file stem, e.g. load_club('man_utd')."""
    path = clubs_dir / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in clubs_dir.glob("*.yaml"))
        raise FileNotFoundError(f"no club config {path.name}; available: {available}")
    club = ClubConfig(**_read_yaml(path))
    roles = load_roles()
    missing = {r for r in club.roles.values() if r not in roles}
    if missing:
        raise ValueError(f"{club.club}: roles not in roles.yaml: {sorted(missing)}")
    return club
