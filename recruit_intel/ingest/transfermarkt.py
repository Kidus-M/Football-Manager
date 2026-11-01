"""Transfermarkt adapter: market value, contract end, birth date, position, foot.

Scrapes two page types:
- a league's club list   /<slug>/startseite/wettbewerb/<code>/saison_id/<year>
- a club's full squad    /<slug>/kader/verein/<id>/saison_id/<year>/plus/1

Layout changes break this first of all the sources; parsing is kept in pure
functions with fixture tests so a break shows up in pytest, not mid-pipeline.
"""

import re
from datetime import date, datetime

import pandas as pd
from bs4 import BeautifulSoup, Tag

from recruit_intel.ingest.http import DAY, CachedClient

SOURCE = "transfermarkt"
BASE = "https://www.transfermarkt.com"

# Country code used in club configs -> (Transfermarkt competition code, competition_id)
LEAGUES = {
    "ENG": ("GB1", "ENG-Premier League"),
    "ESP": ("ES1", "ESP-La Liga"),
    "ITA": ("IT1", "ITA-Serie A"),
    "GER": ("L1", "GER-Bundesliga"),
    "FRA": ("FR1", "FRA-Ligue 1"),
    "NED": ("NL1", "NED-Eredivisie"),
    "POR": ("PO1", "POR-Liga Portugal"),
    "BEL": ("BE1", "BEL-Pro League"),
}

POSITIONS = {"Goalkeeper": "GK", "Defender": "DEF", "Midfield": "MID", "Attack": "FWD"}

_MULTIPLIERS = {"k": 1e3, "m": 1e6, "bn": 1e9}


def parse_value(text: str) -> int | None:
    """'€35M' -> 35_000_000, '€200K' -> 200_000, '-' -> None."""
    m = re.search(r"([\d.,]+)\s*(bn|m|k)?", text.lower())
    if not m:
        return None
    number = float(m.group(1).replace(",", "."))
    return int(round(number * _MULTIPLIERS.get(m.group(2) or "", 1)))


def parse_date(text: str) -> date | None:
    """'07/07/2002 (24)' or 'Jul 7, 2002 (24)' -> date; '-' -> None."""
    text = re.sub(r"\(.*?\)", "", text).strip()
    for fmt in ("%d/%m/%Y", "%b %d, %Y", "%d.%m.%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def parse_height(text: str) -> int | None:
    """'1,93m' -> 193."""
    m = re.search(r"(\d)[,.](\d{2})", text)
    return int(m.group(1) + m.group(2)) if m else None


def _id_from_href(href: str, kind: str) -> str:
    m = re.search(rf"/{kind}/(\d+)", href)
    if not m:
        raise ValueError(f"no {kind} id in {href!r}")
    return m.group(1)


def _rows(soup: BeautifulSoup) -> list[Tag]:
    table = soup.select_one("table.items")
    if table is None:
        raise ValueError("no table.items on page (layout change or block page?)")
    return table.select(":scope > tbody > tr")


def parse_league(html: str, competition_id: str, season: int) -> pd.DataFrame:
    """Club list of a league -> teams rows (source_team_id is the TM club id)."""
    clubs = []
    for row in _rows(BeautifulSoup(html, "lxml")):
        link = row.select_one("td.hauptlink a[href*='/verein/']")
        if link is None:
            continue
        clubs.append(
            {
                "source": SOURCE,
                "source_team_id": _id_from_href(link["href"], "verein"),
                "name": link.get_text(strip=True),
                "competition_id": competition_id,
                "season": season,
            }
        )
    return pd.DataFrame(clubs)


def parse_squad(html: str, snapshot: date) -> dict[str, pd.DataFrame]:
    """Full squad page -> source_players and market rows."""
    soup = BeautifulSoup(html, "lxml")
    team = soup.select_one("h1").get_text(" ", strip=True)
    players, market = [], []
    for row in _rows(soup):
        tds = row.find_all("td", recursive=False)
        if len(tds) < 10:
            continue
        link = tds[1].select_one("td.hauptlink a")
        pid = _id_from_href(link["href"], "spieler")
        detail = tds[1].select("table.inline-table tr")
        flags = [img["title"] for img in tds[3].select("img[title]")]
        players.append(
            {
                "source": SOURCE,
                "source_player_id": pid,
                "name": link.get_text(strip=True),
                "team_name": team,
                "position": POSITIONS.get(tds[0].get("title", "")),
                "detailed_position": detail[-1].get_text(strip=True) if len(detail) > 1 else None,
                "birth_date": parse_date(tds[2].get_text(strip=True)),
                "nationality": flags[0] if flags else None,
                "height_cm": parse_height(tds[4].get_text(strip=True)),
                "foot": tds[5].get_text(strip=True) or None,
            }
        )
        market.append(
            {
                "source": SOURCE,
                "source_player_id": pid,
                "value_eur": parse_value(tds[9].get_text(strip=True)),
                "contract_end": parse_date(tds[8].get_text(strip=True)),
                "joined_date": parse_date(tds[6].get_text(strip=True)),
                "snapshot_date": snapshot,
            }
        )
    for df_rows in (players, market):
        if not df_rows:
            raise ValueError(f"{team}: squad table parsed to zero players")
    return {"source_players": pd.DataFrame(players), "market": pd.DataFrame(market)}


def make_client(**kw) -> CachedClient:
    # TM is the most block-happy source: keep well under one request per 3s.
    return CachedClient(SOURCE, min_interval=kw.pop("min_interval", 4.0), **kw)


def league_url(code: str, season: int) -> str:
    # The slug segment is cosmetic; TM routes on the competition code.
    return f"{BASE}/x/startseite/wettbewerb/{code}/saison_id/{season}"


def squad_url(club_id: str | int, season: int) -> str:
    return f"{BASE}/x/kader/verein/{club_id}/saison_id/{season}/plus/1"


def fetch_league_clubs(client: CachedClient, country: str, season: int) -> pd.DataFrame:
    code, competition_id = LEAGUES[country]
    return parse_league(client.get_text(league_url(code, season)), competition_id, season)


def fetch_squad(client: CachedClient, club_id: str | int, season: int) -> dict[str, pd.DataFrame]:
    out = parse_squad(client.get_text(squad_url(club_id, season)), date.today())
    out["source_players"]["team_id"] = str(club_id)
    return out


def ingest_club(club_id: str | int, season: int, client=None) -> dict[str, pd.DataFrame]:
    own = client is None
    client = client or make_client(ttl=7 * DAY)
    try:
        return fetch_squad(client, club_id, season)
    finally:
        if own:
            client.close()
