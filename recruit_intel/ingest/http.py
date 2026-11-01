"""Cached, rate-limited HTTP for all scrapers.

Every response body is cached on disk under data/cache/<source>/ so re-running
the pipeline costs no requests until the cache entry expires. Requests to the
same host are spaced at least `min_interval` seconds apart.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from recruit_intel import DATA_DIR

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)
DAY = 24 * 3600


class CachedClient:
    def __init__(
        self,
        source: str,
        *,
        ttl: float = DAY,
        min_interval: float = 2.0,
        cache_dir: Path | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.source = source
        self.ttl = ttl
        self.min_interval = min_interval
        self.cache_dir = (cache_dir or DATA_DIR / "cache") / source
        self._client = httpx.Client(
            headers={"User-Agent": USER_AGENT, **(headers or {})},
            timeout=30,
            follow_redirects=True,
        )
        self._last_request: dict[str, float] = {}

    def _path(self, url: str, params: dict[str, Any] | None) -> Path:
        key = url + "?" + json.dumps(params or {}, sort_keys=True)
        return self.cache_dir / (hashlib.sha256(key.encode()).hexdigest()[:24] + ".cache")

    def _throttle(self, url: str) -> None:
        host = urlparse(url).netloc
        wait = self._last_request.get(host, 0) + self.min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_request[host] = time.monotonic()

    def get_text(self, url: str, params: dict[str, Any] | None = None, **kw: Any) -> str:
        path = self._path(url, params)
        if path.exists() and time.time() - path.stat().st_mtime < self.ttl:
            return path.read_text(encoding="utf-8")
        self._throttle(url)
        resp = self._client.get(url, params=params, **kw)
        resp.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(resp.text, encoding="utf-8")
        return resp.text

    def get_json(self, url: str, params: dict[str, Any] | None = None, **kw: Any) -> Any:
        return json.loads(self.get_text(url, params, **kw))

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "CachedClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
