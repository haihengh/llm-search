"""Tool definitions and execution dispatch.

The web_search tool is auto-injected into every request so clients
don't need to define it themselves. Additional client-provided tools
are preserved alongside it.
"""

import logging
from typing import Any

from .cache import cache
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

# Tool name constant
WEB_SEARCH = "web_search"


def inject_web_search_tool(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Ensure web_search is present in the tools list.

    If the client provided tools, web_search is added alongside them.
    If no tools were provided, returns a list with just web_search.
    """
    tools = list(tools) if tools else []

    # Check if web_search is already present
    for tool in tools:
        if tool.get("function", {}).get("name") == WEB_SEARCH:
            return tools  # Already there — don't duplicate

    tools.append(WEB_SEARCH_TOOL)
    return tools


# ── Tool Execution ────────────────────────────────────────────

async def execute_web_search(
    query: str,
    num_results: int,
    search_provider: SearchProvider,
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

# Map of tool name → executor function.
# Extensible: add more tools here and they become available to the LLM.
TOOL_EXECUTORS: dict[str, Any] = {
    WEB_SEARCH: execute_web_search,
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
