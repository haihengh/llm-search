"""SerpAPI provider — optional alternative to SearXNG.

Requires SEARCH_API_KEY set to a SerpAPI key.
Free tier: 100 queries/month. https://serpapi.com/
"""

import logging
from typing import Optional
from urllib.parse import urlencode

import httpx

from .base import SearchProvider, SearchResult

logger = logging.getLogger(__name__)

SERPAPI_URL = "https://serpapi.com/search"


class SerpAPIProvider(SearchProvider):
    """Search adapter for SerpAPI (Google Search API wrapper)."""

    def __init__(self, api_key: str, timeout: float = 10.0):
        self._api_key = api_key
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def name(self) -> str:
        return "serpapi"

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        client = await self._get_client()
        try:
            params = {
                "api_key": self._api_key,
                "q": query,
                "engine": "google",
                "num": str(min(num_results, 10)),
            }
            response = await client.get(f"{SERPAPI_URL}?{urlencode(params)}")
            response.raise_for_status()
            data = response.json()

            results = []
            organic = data.get("organic_results", [])
            for i, entry in enumerate(organic[:num_results]):
                results.append(
                    SearchResult(
                        title=entry.get("title", "Untitled"),
                        url=entry.get("link", ""),
                        snippet=entry.get("snippet", ""),
                        position=i + 1,
                    )
                )

            logger.debug(
                "SerpAPI search for %r returned %d results", query, len(results)
            )
            return results

        except httpx.HTTPError as exc:
            logger.error("SerpAPI search failed for query %r: %s", query, exc)
            raise

    async def health_check(self) -> bool:
        """Check if the SerpAPI key is valid."""
        client = await self._get_client()
        try:
            response = await client.get(
                f"{SERPAPI_URL}?{urlencode({'api_key': self._api_key, 'q': 'test', 'engine': 'google', 'num': '1'})}"
            )
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
