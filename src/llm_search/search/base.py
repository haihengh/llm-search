"""Abstract search provider interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SearchResult:
    """A single search result from any provider."""

    title: str
    url: str
    snippet: str
    position: int = 0

    def to_llm_format(self) -> str:
        """Format as compact text for LLM consumption."""
        return f'[{self.position}] "{self.title}"\n{self.url}\n{self.snippet}\n'


class SearchProvider(ABC):
    """Abstract base for search backends."""

    @abstractmethod
    async def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        """Execute a search and return parsed results."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name for health checks."""
        ...

    async def health_check(self) -> bool:
        """Check if the provider is reachable. Override for real checks."""
        return True


def format_results_for_llm(results: list[SearchResult]) -> str:
    """Format a list of search results as a text block for the LLM.

    Compact and token-efficient: just position, title, URL, snippet.
    """
    if not results:
        return "No search results found."

    formatted = [r.to_llm_format() for r in results]
    return "Search results:\n\n" + "\n".join(formatted)
