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
