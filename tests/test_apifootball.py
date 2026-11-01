import json

import httpx
import pytest

from recruit_intel.ingest import apifootball as af
from recruit_intel.ingest.http import CachedClient


def _stat(team, minutes, tackles, interceptions, past, team_id=1, league_id=39):
    return {
        "team": {"id": team_id, "name": team},
        "league": {"id": league_id, "season": 2026},
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
                _stat("Lille", 450, 10, 4, 2, team_id=79),
                _stat("Manchester United", 450, 8, None, 1, team_id=33),
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
    assert stats.loc["interceptions"].value == 4  # the unrecorded entry adds nothing
    assert stats.loc["dribbled_past"].value == 3


def test_only_this_leagues_entries_count_and_duplicates_merge():
    cup = _stat("Manchester United", 270, 50, 50, 50, team_id=33, league_id=45)
    page = json.loads(json.dumps(PAGE))
    page["response"][0]["statistics"].append(cup)
    # The same player, seen again from his other club's squad pages.
    again = {"response": [page["response"][0]]}
    stats = af.parse([page, again], "ENG-Premier League", 2026, league_id=39)["player_season_stats"]
    row = stats.set_index("metric").loc["tackles"]
    assert row.value == 18 and row.minutes == 900  # cup excluded, no double count


def test_ingest_league_fetches_club_by_club(tmp_path):
    def handler(request):
        path, params = request.url.path, request.url.params
        if path.endswith("/teams"):
            return httpx.Response(
                200,
                json={
                    "errors": [],
                    "response": [
                        {"team": {"id": 79, "name": "Lille"}},
                        {"team": {"id": 33, "name": "Man Utd"}},
                    ],
                },
            )
        assert "league" not in params  # never the page-capped league listing
        return httpx.Response(200, json={**PAGE, "paging": {"current": 1, "total": 1}})

    client = CachedClient("af", cache_dir=tmp_path, min_interval=0)
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    out = af.ingest_league(client, "ENG", 2026)
    assert list(out["source_players"].source_player_id) == ["5"]


def test_stat_line_duplicated_across_clubs_is_dropped():
    # The same league stat line under two clubs: more minutes than a season.
    dup = {"player": {"id": 1165, "name": "Youth Player"}, "statistics": [
        _stat("Manchester United", 2600, 30, 20, 5, team_id=33),
        _stat("Wolves", 2603, 30, 20, 5, team_id=39),
    ]}  # fmt: skip
    out = af.parse([{"response": [dup, *PAGE["response"]]}], "ENG-Premier League", 2024,
                   league_id=39)  # fmt: skip
    assert "1165" not in set(out["source_players"].source_player_id)
    assert "5" in set(out["source_players"].source_player_id)  # a genuine move is kept


def test_field_never_recorded_is_missing_not_zero():
    page = json.loads(json.dumps(PAGE))
    for s in page["response"][0]["statistics"]:
        s["tackles"]["blocks"] = None
    metrics = set(af.parse([page], "ENG-Premier League", 2026)["player_season_stats"].metric)
    assert "blocks" not in metrics and "tackles" in metrics


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


PLAN = {"errors": {"plan": "Free plans do not have access to this season, try from 2022 to 2024."}}


def test_plan_restriction_falls_back_to_newest_allowed_season(tmp_path):
    seen = []

    def handler(request):
        seen.append(request.url.params["season"])
        return httpx.Response(200, json=PLAN if request.url.params["season"] == "2026" else PAGE)

    client = CachedClient("af", cache_dir=tmp_path, min_interval=0)
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    assert af.usable_season(client, 2026) == 2024
    assert af.usable_season(client, 2026) == 2024  # remembered: no second probe
    assert seen == ["2026"]
    assert not list(tmp_path.rglob("*.cache"))  # the refusal itself isn't cached


def test_plan_error_is_its_own_exception(tmp_path):
    with pytest.raises(af.PlanRestricted) as info:
        af.fetch_page(_client(tmp_path, PLAN), 39, 2026, 1)
    assert info.value.latest == 2024


def test_http_429_daily_limit_is_quota_exhausted(tmp_path):
    body = {"errors": {"rateLimit": "You have reached your daily request limit."}}
    client = CachedClient("af", cache_dir=tmp_path, min_interval=0)
    client._client = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(429, json=body))
    )
    with pytest.raises(af.QuotaExhausted):
        af.fetch_page(client, 39, 2024, 1)
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
