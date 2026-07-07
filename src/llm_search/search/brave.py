"""Brave Search API provider — optional alternative to SearXNG.

Requires SEARCH_API_KEY set to a Brave Search API key.
Free tier: 2,000 queries/month. https://brave.com/search/api/
"""

import logging
from typing import Optional

import httpx

from .base import SearchProvider, SearchResult

logger = logging.getLogger(__name__)

BRAVE_API_URL = "https://api.search.brave.com/res/v1/web/search"


class BraveSearchProvider(SearchProvider):
    """Search adapter for the Brave Search API."""

    def __init__(self, api_key: str, timeout: float = 10.0):
        self._api_key = api_key
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def name(self) -> str:
        return "brave"

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": self._api_key,
                },
            )
        return self._client

    async def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        client = await self._get_client()
        try:
            response = await client.get(
                BRAVE_API_URL,
                params={
                    "q": query,
                    "count": min(num_results, 20),
                },
            )
            response.raise_for_status()
            data = response.json()

            results = []
            web_results = data.get("web", {}).get("results", [])
            for i, entry in enumerate(web_results[:num_results]):
                results.append(
                    SearchResult(
                        title=entry.get("title", "Untitled"),
                        url=entry.get("url", ""),
                        snippet=entry.get("description", ""),
                        position=i + 1,
                    )
                )

            logger.debug(
                "Brave search for %r returned %d results", query, len(results)
            )
            return results

        except httpx.HTTPError as exc:
            logger.error("Brave search failed for query %r: %s", query, exc)
            raise

    async def health_check(self) -> bool:
        """Check if the Brave API key is valid."""
        client = await self._get_client()
        try:
            response = await client.get(
                BRAVE_API_URL,
                params={"q": "test", "count": 1},
            )
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
