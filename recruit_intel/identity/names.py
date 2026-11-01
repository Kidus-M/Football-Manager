"""Name normalisation shared by every cross-source matcher."""

import re
import unicodedata

# Letters NFKD does not decompose into ASCII.
_SPECIAL = str.maketrans({"ı": "i", "ø": "o", "æ": "ae", "ß": "ss", "đ": "d", "ł": "l"})


def normalize(name: str) -> str:
    """'Altay Bayındır' -> 'altay bayindir'."""
    name = name.lower().translate(_SPECIAL)
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]+", "", ascii_name.replace("-", " ")).strip()


def tokens(name: str) -> set[str]:
    return set(normalize(name).split())
