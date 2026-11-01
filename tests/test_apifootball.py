import json

import httpx
import pytest

from recruit_intel.ingest import apifootball as af
from recruit_intel.ingest.http import CachedClient


def _stat(team, minutes, tackles, interceptions, past):
    return {
        "team": {"id": 1, "name": team},
        "league": {"id": 39, "season": 2026},
        "games": {"minutes": minutes, "position": "Defender"},
        "tackles": {"total": tackles, "blocks": 1, "interceptions": interceptions},
        "duels": {"total": 20, "won": 12},
        "dribbles": {"attempts": 2, "success": 1, "past": past},
        "fouls": {"drawn": 3, "committed": 2},
        "passes": {"total": 400, "key": 3, "accuracy": "88"},
    }


PAGE = {
    "errors": [],
    "paging": {"current": 1, "total": 1},
    "response": [
        {
            "player": {"id": 5, "name": "L. Yoro", "birth": {"date": "2005-11-13"},
                       "nationality": "France"},
            # Moved mid-season: both clubs' entries count.
            "statistics": [
                _stat("Lille", 450, 10, 4, 2),
                _stat("Manchester United", 450, 8, None, 1),
            ],
        },
        {"player": {"id": 6, "name": "Unused"}, "statistics": [_stat("X", None, 0, 0, 0)]},
    ],
}  # fmt: skip


def test_parse_sums_entries_and_skips_players_without_minutes():
    out = af.parse([PAGE], "ENG-Premier League", 2026)
    players = out["source_players"]
    assert list(players.source_player_id) == ["5"]
    assert players.iloc[0].team_name == "Manchester United"
    assert str(players.iloc[0].birth_date) == "2005-11-13"
    stats = out["player_season_stats"].set_index("metric")
    assert stats.loc["tackles"].value == 18 and stats.loc["tackles"].minutes == 900
    assert stats.loc["interceptions"].value == 4  # None counts as 0
    assert stats.loc["dribbled_past"].value == 3


def _client(tmp_path, body):
    client = CachedClient("af", cache_dir=tmp_path, min_interval=0)
    client._client = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body))
    )
    return client


def test_quota_error_raises_and_is_not_cached(tmp_path):
    limit = {"errors": {"requests": "You have reached the request limit for the day"}}
    client = _client(tmp_path, limit)
    with pytest.raises(af.QuotaExhausted):
        af.fetch_page(client, 39, 2026, 1)
    assert not any(tmp_path.rglob("*.cache"))


def test_ok_page_is_cached(tmp_path):
    client = _client(tmp_path, PAGE)
    assert af.fetch_page(client, 39, 2026, 1)["paging"]["total"] == 1
    assert len(list(tmp_path.rglob("*.cache"))) == 1


def test_api_key_from_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)
    monkeypatch.setattr(af, "ROOT", tmp_path)
    assert af.api_key() is None
    (tmp_path / ".env").write_text('API_FOOTBALL_KEY="abc123"\n')
    assert af.api_key() == "abc123"
    assert json.dumps(PAGE)  # fixture stays JSON-serialisable
