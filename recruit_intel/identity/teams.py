"""Map each source's team names onto Transfermarkt clubs.

Fuzzy matching on names with club-type tokens stripped ("FC", "AFC", "1.")
handles most clubs; config/team_aliases.yaml covers the rest ("Man Utd",
"Spurs"). Matching is one-to-one within a competition, so "Man City" can
never land on Manchester United just because it scored second best there.
"""

from functools import cache
from pathlib import Path

import yaml
from rapidfuzz import fuzz

from recruit_intel import CONFIG_DIR
from recruit_intel.identity.names import normalize

# Tokens that say what kind of club it is, not which club.
_CLUB_TOKENS = {
    "fc", "afc", "cf", "sc", "ac", "as", "ss", "ssc", "us", "sv", "vfb", "vfl", "tsg", "rc",
    "rcd", "ogc", "cd", "ud", "sd", "ca", "fsv", "bc", "club", "calcio", "de", "the",
    "1", "04", "05", "1846", "1899", "1900", "1907", "1909", "1910", "1913",
}  # fmt: skip

MIN_SCORE = 80


def team_key(name: str) -> str:
    """'1. FC Union Berlin' -> 'union berlin'."""
    words = normalize(name.replace(".", " ")).split()
    kept = [w for w in words if w not in _CLUB_TOKENS]
    return " ".join(kept or words)


@cache
def load_aliases(path: Path = CONFIG_DIR / "team_aliases.yaml") -> dict[str, str]:
    """Alias key -> canonical Transfermarkt name key."""
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {team_key(alias): team_key(tm) for tm, aliases in raw.items() for alias in aliases}


def match_teams(
    names: list[str], tm_names: list[str], aliases: dict[str, str] | None = None
) -> dict[str, str]:
    """Source team name -> Transfermarkt team name, one-to-one."""
    aliases = load_aliases() if aliases is None else aliases
    tm_by_key = {team_key(t): t for t in tm_names}
    out: dict[str, str] = {}

    pairs = []
    for name in names:
        key = team_key(name)
        alias = aliases.get(key)
        if alias in tm_by_key:
            out[name] = tm_by_key[alias]
            continue
        for tm_key, tm in tm_by_key.items():
            score = max(fuzz.token_set_ratio(key, tm_key), fuzz.ratio(key, tm_key))
            if score >= MIN_SCORE:
                pairs.append((score, name, tm))

    taken = set(out.values())
    for _score, name, tm in sorted(pairs, reverse=True):
        if name not in out and tm not in taken:
            out[name] = tm
            taken.add(tm)
    return out
