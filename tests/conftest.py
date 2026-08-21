"""Shared pytest fixtures.

The search cache is a module-level singleton, so a search executed by one
test can leak into the next (cache hit → provider never called). Clear it
between tests so each test gets a clean slate.
"""

import pytest

from llm_search.cache import cache


@pytest.fixture(autouse=True)
def _clear_search_cache():
    cache.clear()
    yield
    cache.clear()
