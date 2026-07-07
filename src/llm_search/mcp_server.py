"""MCP (Model Context Protocol) server for llm-search.

Exposes web_search and fetch_page as MCP tools over stdio transport.
MCP clients (Claude Desktop, etc.) can connect to this server to
discover and call search/fetch tools directly.

Usage:
    llm-search --mcp
    python -m llm_search --mcp
"""

import asyncio
import json
import logging
from typing import Any

from .config import settings
from .fetch_page import fetch_page_text
from .search.base import SearchProvider, format_results_for_llm
from .search.brave import BraveSearchProvider
from .search.searxng import SearXNGSearcher
from .search.serpapi import SerpAPIProvider

logger = logging.getLogger(__name__)

# ── Search Provider (shared with the API server) ───────────────

_search_provider: SearchProvider | None = None


def _get_provider() -> SearchProvider:
    """Lazy-init the search provider from config."""
    global _search_provider
    if _search_provider is not None:
        return _search_provider

    if settings.search_provider == "brave":
        _search_provider = BraveSearchProvider(api_key=settings.search_api_key)
    elif settings.search_provider == "serpapi":
        _search_provider = SerpAPIProvider(api_key=settings.search_api_key)
    else:
        _search_provider = SearXNGSearcher(base_url=settings.searxng_url)

    return _search_provider


# ── Tool Definitions ───────────────────────────────────────────

WEB_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The search query",
        },
        "num_results": {
            "type": "integer",
            "description": "Number of results to return (1-10, default 5)",
            "default": 5,
        },
    },
    "required": ["query"],
}

FETCH_PAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "url": {
            "type": "string",
            "description": "The full URL of the page to fetch (https://...)",
        },
    },
    "required": ["url"],
}


# ── MCP Server ─────────────────────────────────────────────────

async def run_mcp_server():
    """Start the MCP server over stdio transport.

    Uses the official mcp Python SDK. Must be installed via:
        pip install llm-search[mcp]
    """
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        import mcp.types as types
    except ImportError:
        print(
            "MCP SDK not installed. Run: pip install llm-search[mcp]",
            flush=True,
        )
        raise

    # Setup logging (stdio is the transport; use stderr for logs)
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=None,  # stderr
    )

    server = Server("llm-search")

    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        """Return the list of available tools."""
        return [
            types.Tool(
                name="web_search",
                description=(
                    "Search the internet for current, up-to-date information. "
                    "Use this whenever you need facts, news, or knowledge beyond "
                    "your training cutoff date."
                ),
                inputSchema=WEB_SEARCH_SCHEMA,
            ),
            types.Tool(
                name="fetch_page",
                description=(
                    "Fetch the full text content of a web page by its URL. "
                    "Use this after web_search to read a specific page in detail."
                ),
                inputSchema=FETCH_PAGE_SCHEMA,
            ),
        ]

    @server.call_tool()
    async def handle_call_tool(
        name: str,
        arguments: dict[str, Any],
    ) -> list[types.TextContent]:
        """Execute a tool and return the result."""
        try:
            if name == "web_search":
                query = arguments.get("query", "")
                num_results = max(1, min(int(arguments.get("num_results", 5)), 10))
                logger.info("MCP web_search: %r (n=%d)", query, num_results)

                provider = _get_provider()
                results = await provider.search(query, num_results)
                text = format_results_for_llm(results)

            elif name == "fetch_page":
                url = arguments.get("url", "")
                logger.info("MCP fetch_page: %s", url)
                text = await fetch_page_text(url)

            else:
                text = f"Unknown tool: {name}"

        except Exception as exc:
            logger.error("MCP tool %r failed: %s", name, exc)
            text = f"Error executing {name}: {exc}"

        return [types.TextContent(type="text", text=text)]

    # Run over stdio
    logger.info("MCP server starting (stdio transport)")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )
