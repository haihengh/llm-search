"""Entry point: python -m llm_search

Starts the middleware server. All configuration is via environment
variables or .env file. SearXNG should be running separately
(typically via docker compose).
"""

import uvicorn

from .config import settings


def main():
    """Start the LLM Search middleware server."""
    uvicorn.run(
        "llm_search.server:app",
        host=settings.middleware_host,
        port=settings.middleware_port,
        log_level=settings.log_level.lower(),
        reload=False,
    )


if __name__ == "__main__":
    main()
