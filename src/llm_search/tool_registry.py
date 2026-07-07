"""Tool definitions and execution dispatch.

The web_search and fetch_page tools are auto-injected into every request
so clients don't need to define them. Additional client-provided tools
are preserved alongside them.
"""

import logging
from typing import Any

from .cache import cache
from .fetch_page import fetch_page_text
from .search.base import SearchProvider, format_results_for_llm

logger = logging.getLogger(__name__)

# ── Tool Definition ───────────────────────────────────────────

WEB_SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the internet for current, up-to-date information. "
            "Use this whenever you need facts, news, or knowledge beyond "
            "your training cutoff date."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (1-10)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}

FETCH_PAGE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "fetch_page",
        "description": (
            "Fetch the full text content of a web page by its URL. "
            "Use this after web_search to read a specific page in detail — "
            "for example, to get full release notes, documentation, or article text. "
            "Returns clean readable text (max ~8000 chars)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL of the page to fetch (must start with http:// or https://)",
                },
            },
            "required": ["url"],
        },
    },
}

# Tool name constants
WEB_SEARCH = "web_search"
FETCH_PAGE = "fetch_page"


def inject_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Ensure web_search and fetch_page are present in the tools list.

    If the client provided tools, both are added alongside them.
    If no tools were provided, returns a list with both built-in tools.
    """
    tools = list(tools) if tools else []
    existing_names = {t.get("function", {}).get("name") for t in tools}

    if WEB_SEARCH not in existing_names:
        tools.append(WEB_SEARCH_TOOL)
    if FETCH_PAGE not in existing_names:
        tools.append(FETCH_PAGE_TOOL)

    return tools


# Backward-compatible alias
def inject_web_search_tool(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Deprecated: use inject_tools() instead."""
    return inject_tools(tools)


# ── Tool Execution ────────────────────────────────────────────

async def execute_web_search(
    query: str,
    search_provider: SearchProvider,
    num_results: int = 5,
) -> str:
    """Execute a web search and return LLM-formatted results.

    Checks the cache first; on miss, queries the search provider.
    The result is formatted as compact text for LLM consumption.
    """
    # Clamp num_results
    num_results = max(1, min(num_results, 10))

    # Check cache
    cached = cache.get(query, num_results)
    if cached is not None:
        logger.debug("Cache hit for %r (n=%d)", query, num_results)
        return format_results_for_llm(cached)

    # Execute search
    logger.info("Searching for %r (n=%d)", query, num_results)
    results = await search_provider.search(query, num_results)

    # Cache the results
    cache.set(query, num_results, results)

    return format_results_for_llm(results)


# ── Tool Dispatch ─────────────────────────────────────────────

async def execute_fetch_page(
    url: str,
    **kwargs,
) -> str:
    """Fetch a web page and return its readable text content.

    Validates the URL, fetches the page, extracts text, and truncates
    to a reasonable size for LLM consumption.
    """
    logger.info("Fetching page: %s", url)
    return await fetch_page_text(url)


# ── Tool Dispatch ─────────────────────────────────────────────

# Map of tool name → executor function.
# Extensible: add more tools here and they become available to the LLM.
TOOL_EXECUTORS: dict[str, Any] = {
    WEB_SEARCH: execute_web_search,
    FETCH_PAGE: execute_fetch_page,
}


async def dispatch_tool(
    tool_name: str,
    arguments: dict[str, Any],
    search_provider: SearchProvider,
) -> str:
    """Execute a tool by name and return its result as a string."""
    executor = TOOL_EXECUTORS.get(tool_name)
    if executor is None:
        return f"Error: unknown tool '{tool_name}'"

    try:
        return await executor(**arguments, search_provider=search_provider)
    except Exception as exc:
        logger.error("Tool %r execution failed: %s", tool_name, exc)
        return f"Error executing {tool_name}: {exc}"
