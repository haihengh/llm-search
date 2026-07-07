"""Search provider adapters."""

from .base import SearchProvider, SearchResult
from .brave import BraveSearchProvider
from .searxng import SearXNGSearcher
from .serpapi import SerpAPIProvider

__all__ = [
    "SearchProvider",
    "SearchResult",
    "SearXNGSearcher",
    "BraveSearchProvider",
    "SerpAPIProvider",
]
