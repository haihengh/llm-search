"""Tests for the search result cache."""

import time

import pytest

from llm_search.cache import SearchCache


class TestSearchCache:
    """Tests for the TTL search cache."""

    def test_get_miss_returns_none(self):
        cache = SearchCache(ttl_seconds=300)
        result = cache.get("nonexistent query", 5)
        assert result is None

    def test_set_and_get(self):
        cache = SearchCache(ttl_seconds=300)
        results = [{"title": "Test", "url": "https://example.com", "snippet": "..."}]
        cache.set("test query", 5, results)
        cached = cache.get("test query", 5)
        assert cached == results

    def test_same_query_different_num_results_is_separate_key(self):
        cache = SearchCache(ttl_seconds=300)
        r5 = [{"title": "Five"}]
        r10 = [{"title": "Ten"}]
        cache.set("query", 5, r5)
        cache.set("query", 10, r10)
        assert cache.get("query", 5) == r5
        assert cache.get("query", 10) == r10

    def test_cache_is_case_insensitive(self):
        cache = SearchCache(ttl_seconds=300)
        results = [{"title": "Test"}]
        cache.set("Hello World", 5, results)
        assert cache.get("hello world", 5) == results
        assert cache.get("HELLO WORLD", 5) == results

    def test_cache_whitespace_insensitive(self):
        cache = SearchCache(ttl_seconds=300)
        results = [{"title": "Test"}]
        cache.set("  hello world  ", 5, results)
        assert cache.get("hello world", 5) == results

    def test_hit_rate(self):
        cache = SearchCache(ttl_seconds=300)
        cache.set("q", 5, [{}])
        cache.get("q", 5)  # hit
        cache.get("q2", 5)  # miss
        cache.get("q3", 5)  # miss
        assert cache.hits == 1
        assert cache.misses == 2
        assert cache.hit_rate == 1 / 3

    def test_clear_resets_counters(self):
        cache = SearchCache(ttl_seconds=300)
        cache.set("q", 5, [{}])
        cache.get("q", 5)
        cache.clear()
        assert cache.hits == 0
        assert cache.misses == 0
        assert cache.hit_rate == 0.0
        assert cache.get("q", 5) is None

    def test_ttl_expiry(self):
        cache = SearchCache(ttl_seconds=0.1)  # 100ms TTL
        cache.set("q", 5, [{"title": "ephemeral"}])
        assert cache.get("q", 5) is not None
        time.sleep(0.15)
        assert cache.get("q", 5) is None

    def test_zero_hits_hit_rate_is_zero(self):
        cache = SearchCache(ttl_seconds=300)
        assert cache.hit_rate == 0.0
