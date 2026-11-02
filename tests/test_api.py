"""API smoke tests against the local database (skipped when it isn't built)."""

import math

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from recruit_intel import db  # noqa: E402

pytestmark = [
    pytest.mark.skipif(not db.DEFAULT_DB.exists(), reason="no local database; run `recruit ingest`"),
    pytest.mark.uses_trained_models,
]


@pytest.fixture(scope="module")
def client():
    from recruit_intel.api.main import app

    return TestClient(app)


def _no_nan(obj):
    if isinstance(obj, float):
        assert not math.isnan(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _no_nan(v)
    elif isinstance(obj, list):
        for v in obj:
            _no_nan(v)


def test_club_has_theme_and_formation(client):
    body = client.get("/api/club").json()
    assert body["formation"] in body["formations"]
    assert body["theme"]["accent"].startswith("#")


def test_lineup_has_eleven_positioned_slots(client):
    body = client.get("/api/lineup").json()
    assert len(body["slots"]) == 11
    assert all(0 <= s["x"] <= 100 and 0 <= s["y"] <= 100 for s in body["slots"])
    _no_nan(body)


def test_lineup_rejects_unknown_formation(client):
    assert client.get("/api/lineup", params={"formation": "9-0-1"}).status_code == 400


def test_player_lookup_and_detail(client):
    hits = client.get("/api/players", params={"q": "Mainoo"}).json()
    assert hits, "expected Kobbie Mainoo in the pool"
    detail = client.get(f"/api/players/{hits[0]['player_id']}").json()
    assert detail["percentiles"] and detail["roles"]
    _no_nan(detail)
    assert client.get("/api/players/nobody").status_code == 404


def test_search_returns_ranked_candidates(client):
    body = client.get("/api/search", params={"slot": "RCB", "limit": 5}).json()
    scores = [c["score"] for c in body["candidates"]]
    assert scores == sorted(scores, reverse=True)
    _no_nan(body)
    assert client.get("/api/search", params={"slot": "XX"}).status_code == 404


def test_tactics_with_what_if(client):
    body = client.get("/api/tactics", params={"press": "high"}).json()
    assert {lv["lever"] for lv in body["levers"]} >= {"press", "buildup"}
    assert "press" in body["settings"]
    _no_nan(body)
    assert client.get("/api/tactics", params={"press": "extreme"}).status_code == 400
