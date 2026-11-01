import httpx

from recruit_intel.ingest.http import CachedClient


def test_second_call_is_served_from_cache(tmp_path):
    calls = []

    def handler(request):
        calls.append(request.url)
        return httpx.Response(200, json={"ok": True})

    client = CachedClient("t", cache_dir=tmp_path, min_interval=0)
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    assert client.get_json("https://x.test/a") == {"ok": True}
    assert client.get_json("https://x.test/a") == {"ok": True}
    assert len(calls) == 1


def test_transient_errors_are_retried(tmp_path, monkeypatch):
    import recruit_intel.ingest.http as http

    monkeypatch.setattr(http, "BACKOFF_S", 0)
    attempts = []

    def handler(request):
        attempts.append(1)
        if len(attempts) < 3:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json={"ok": True})

    client = CachedClient("t", cache_dir=tmp_path, min_interval=0)
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    assert client.get_json("https://x.test/a") == {"ok": True}
    assert len(attempts) == 3


def test_retries_give_up_eventually(tmp_path, monkeypatch):
    import pytest

    import recruit_intel.ingest.http as http

    monkeypatch.setattr(http, "BACKOFF_S", 0)

    def handler(request):
        raise httpx.ConnectTimeout("down", request=request)

    client = CachedClient("t", cache_dir=tmp_path, min_interval=0)
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.ConnectTimeout):
        client.get_text("https://x.test/a")
