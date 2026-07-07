"""Tests for search providers using mocked HTTP responses."""

import pytest
from pytest_httpx import HTTPXMock

from llm_search.search.base import SearchResult, format_results_for_llm
from llm_search.search.brave import BraveSearchProvider
from llm_search.search.searxng import SearXNGSearcher
from llm_search.search.serpapi import SerpAPIProvider


class TestSearchResult:
    """Tests for SearchResult formatting."""

    def test_to_llm_format(self):
        sr = SearchResult(
            title="Test Title",
            url="https://example.com",
            snippet="This is a test snippet.",
            position=1,
        )
        formatted = sr.to_llm_format()
        assert "[1]" in formatted
        assert "Test Title" in formatted
        assert "https://example.com" in formatted
        assert "test snippet" in formatted

    def test_format_results_for_llm_empty(self):
        result = format_results_for_llm([])
        assert result == "No search results found."

    def test_format_results_for_llm_multiple(self):
        results = [
            SearchResult("A", "https://a.com", "Snippet A", 1),
            SearchResult("B", "https://b.com", "Snippet B", 2),
        ]
        formatted = format_results_for_llm(results)
        assert "[1]" in formatted
        assert "[2]" in formatted
        assert "A" in formatted
        assert "B" in formatted


class TestSearXNGProvider:
    """Tests for the SearXNG search adapter."""

    @pytest.mark.asyncio
    async def test_search_parses_results(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            url="http://searxng:8080/search?q=test+query&format=json&categories=general&pageno=1",
            json={
                "results": [
                    {
                        "title": "First Result",
                        "url": "https://example.com/1",
                        "content": "Content of first result",
                    },
                    {
                        "title": "Second Result",
                        "url": "https://example.com/2",
                        "content": "Content of second result",
                    },
                    {
                        "title": "Third — Should be cut",
                        "url": "https://example.com/3",
                        "content": "This one exceeds num_results=2",
                    },
                ]
            },
        )

        provider = SearXNGSearcher(base_url="http://searxng:8080")
        results = await provider.search("test query", num_results=2)

        assert len(results) == 2
        assert results[0].title == "First Result"
        assert results[0].url == "https://example.com/1"
        assert results[1].title == "Second Result"

    @pytest.mark.asyncio
    async def test_search_empty_results(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            url="http://searxng:8080/search?q=nothing&format=json&categories=general&pageno=1",
            json={"results": []},
        )

        provider = SearXNGSearcher(base_url="http://searxng:8080")
        results = await provider.search("nothing")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_missing_fields(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            url="http://searxng:8080/search?q=minimal&format=json&categories=general&pageno=1",
            json={
                "results": [
                    {"title": "Only Title"},
                ]
            },
        )

        provider = SearXNGSearcher(base_url="http://searxng:8080")
        results = await provider.search("minimal")
        assert len(results) == 1
        assert results[0].title == "Only Title"
        assert results[0].url == ""
        assert results[0].snippet == ""

    @pytest.mark.asyncio
    async def test_health_check_ok(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            url="http://searxng:8080/search?q=test&format=json",
            status_code=200,
        )

        provider = SearXNGSearcher(base_url="http://searxng:8080")
        ok = await provider.health_check()
        assert ok is True

    @pytest.mark.asyncio
    async def test_health_check_fail(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            url="http://searxng:8080/search?q=test&format=json",
            status_code=500,
        )

        provider = SearXNGSearcher(base_url="http://searxng:8080")
        ok = await provider.health_check()
        assert ok is False


class TestBraveProvider:
    """Tests for the Brave Search adapter."""

    @pytest.mark.asyncio
    async def test_search_parses_results(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            url="https://api.search.brave.com/res/v1/web/search?q=test&count=5",
            json={
                "web": {
                    "results": [
                        {
                            "title": "Brave Result",
                            "url": "https://brave.com/1",
                            "description": "A brave search result",
                        },
                    ]
                }
            },
        )

        provider = BraveSearchProvider(api_key="test-key")
        results = await provider.search("test", num_results=5)

        assert len(results) == 1
        assert results[0].title == "Brave Result"
        assert results[0].url == "https://brave.com/1"
        assert results[0].snippet == "A brave search result"

    @pytest.mark.asyncio
    async def test_search_empty_results(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            url="https://api.search.brave.com/res/v1/web/search?q=nothing&count=5",
            json={"web": {"results": []}},
        )

        provider = BraveSearchProvider(api_key="test-key")
        results = await provider.search("nothing")
        assert results == []


class TestSerpAPIProvider:
    """Tests for the SerpAPI adapter."""

    @pytest.mark.asyncio
    async def test_search_parses_results(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            url__regex=r"https://serpapi\.com/search\?.*q=test.*",
            json={
                "organic_results": [
                    {
                        "title": "Google Result",
                        "link": "https://google.com/1",
                        "snippet": "A google result snippet",
                    },
                ]
            },
        )

        provider = SerpAPIProvider(api_key="test-key")
        results = await provider.search("test", num_results=5)

        assert len(results) == 1
        assert results[0].title == "Google Result"
        assert results[0].url == "https://google.com/1"
        assert results[0].snippet == "A google result snippet"

    @pytest.mark.asyncio
    async def test_search_empty_results(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            url__regex=r"https://serpapi\.com/search\?.*q=nothing.*",
            json={"organic_results": []},
        )

        provider = SerpAPIProvider(api_key="test-key")
        results = await provider.search("nothing")
        assert results == []
