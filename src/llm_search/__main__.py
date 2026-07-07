"""Entry point: python -m llm_search or llm-search command.

Starts the middleware server by default. All configuration is via environment
variables or .env file. SearXNG should be running separately
(typically via docker compose).

Modes:
    llm-search          — Start the OpenAI-compatible API server (default)
    llm-search --mcp    — Start as an MCP server (Model Context Protocol)
"""

import argparse
import sys

import uvicorn

from .config import settings


def main():
    """Main entry point — parse args and start the appropriate server."""
    parser = argparse.ArgumentParser(
        prog="llm-search",
        description="Give your local LLM internet search capability.",
    )
    parser.add_argument(
        "--mcp",
        action="store_true",
        help="Run as an MCP server (Model Context Protocol) over stdio",
    )
    parser.add_argument(
        "--host",
        default=None,
        help=f"Bind address (default: {settings.middleware_host})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"Bind port (default: {settings.middleware_port})",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["debug", "info", "warning", "error"],
        help=f"Log level (default: {settings.log_level.lower()})",
    )

    args = parser.parse_args()

    if args.mcp:
        # MCP server mode (stdio transport)
        from .mcp_server import run_mcp_server
        run_mcp_server()
    else:
        # OpenAI-compatible API server
        host = args.host or settings.middleware_host
        port = args.port or settings.middleware_port
        log_level = (args.log_level or settings.log_level).lower()

        uvicorn.run(
            "llm_search.server:app",
            host=host,
            port=port,
            log_level=log_level,
            reload=False,
        )


if __name__ == "__main__":
    main()
