"""SearXNG search provider — the default, self-hosted metasearch engine.

Queries the SearXNG JSON API. No API key needed — SearXNG anonymously
aggregates results from Google, Bing, DuckDuckGo, etc.
"""

import logging
from typing import Optional

import httpx

from .base import SearchProvider, SearchResult

logger = logging.getLogger(__name__)


class SearXNGSearcher(SearchProvider):
    """Search adapter for a self-hosted SearXNG instance."""

    def __init__(self, base_url: str, timeout: float = 10.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def name(self) -> str:
        return f"searxng ({self._base_url})"

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        client = await self._get_client()
        try:
            response = await client.get(
                f"{self._base_url}/search",
                params={
                    "q": query,
                    "format": "json",
                    "categories": "general",
                    "pageno": 1,
                },
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for i, entry in enumerate(data.get("results", [])[:num_results]):
                results.append(
                    SearchResult(
                        title=entry.get("title", "Untitled"),
                        url=entry.get("url", ""),
                        snippet=entry.get("content", ""),
                        position=i + 1,
                    )
                )

            logger.debug(
                "SearXNG search for %r returned %d results (asked for %d)",
                query,
                len(results),
                num_results,
            )
            return results

        except httpx.HTTPError as exc:
            logger.error("SearXNG search failed for query %r: %s", query, exc)
            raise

    async def health_check(self) -> bool:
        """Check if SearXNG is reachable (without triggering search engine queries)."""
        client = await self._get_client()
        try:
            # Use /healthz — a lightweight endpoint that doesn't query search engines
            response = await client.get(f"{self._base_url}/healthz")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
