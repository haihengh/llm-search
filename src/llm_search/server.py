"""FastAPI server — OpenAI-compatible API with tool-call interception.

Exposes /v1/chat/completions, /health, and /stats.
The tool loop runs server-side: the client makes one request and
gets back the final answer after all searches are complete.
"""

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .cache import cache as search_cache
from .config import settings
from .rate_limiter import rate_limiter
from .search.base import SearchProvider
from .search.brave import BraveSearchProvider
from .search.searxng import SearXNGSearcher
from .search.serpapi import SerpAPIProvider
from .tool_loop import (
    LMStudioError,
    ToolLoopExhaustedError,
    run_tool_loop,
    run_tool_loop_streaming,
)

logger = logging.getLogger(__name__)

# ── Search Provider Factory ───────────────────────────────────

_search_provider: Optional[SearchProvider] = None


def create_search_provider() -> SearchProvider:
    """Build the search provider from configuration."""
    if settings.search_provider == "brave":
        if not settings.search_api_key:
            raise ValueError("SEARCH_API_KEY is required when using Brave Search")
        logger.info("Using Brave Search API")
        return BraveSearchProvider(api_key=settings.search_api_key)

    if settings.search_provider == "serpapi":
        if not settings.search_api_key:
            raise ValueError("SEARCH_API_KEY is required when using SerpAPI")
        logger.info("Using SerpAPI")
        return SerpAPIProvider(api_key=settings.search_api_key)

    # Default: SearXNG
    logger.info("Using SearXNG at %s", settings.searxng_url)
    return SearXNGSearcher(base_url=settings.searxng_url)


def get_search_provider() -> SearchProvider:
    """Get the current search provider singleton."""
    assert _search_provider is not None, "Search provider not initialized"
    return _search_provider


# ── App Lifespan ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown for the FastAPI app."""
    global _search_provider

    # Startup
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("LLM Search v0.1.0 starting")
    logger.info("LM Studio URL: %s", settings.lm_studio_url)
    logger.info("Search provider: %s", settings.search_provider)

    _search_provider = create_search_provider()

    # Verify SearXNG/LM Studio connectivity (non-fatal)
    try:
        if await _search_provider.health_check():
            logger.info("Search provider: OK")
        else:
            logger.warning("Search provider health check failed")
    except Exception as exc:
        logger.warning("Search provider health check error: %s", exc)

    yield

    # Shutdown
    logger.info("LLM Search shutting down")
    if hasattr(_search_provider, "close"):
        await _search_provider.close()


# ── FastAPI App ───────────────────────────────────────────────

app = FastAPI(
    title="LLM Search",
    description="Give your local LLM internet search capability. No API keys, no rate limits.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response Models ───────────────────────────────────

class Message(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ChatRequest(BaseModel):
    model: str = "local-model"
    messages: list[Message]
    tools: Optional[list[dict[str, Any]]] = None
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class ChatResponseChoice(BaseModel):
    index: int = 0
    message: dict[str, Any]
    finish_reason: str = "stop"


class ChatResponseUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatResponseChoice]
    usage: ChatResponseUsage


class HealthResponse(BaseModel):
    status: str
    lm_studio_url: str
    search_provider: str
    searxng_ok: Optional[bool] = None
    lm_studio_ok: bool = False
    cache_hit_rate: float
    total_searches: int
    uptime_seconds: float


class StatsResponse(BaseModel):
    total_requests: int
    total_searches: int
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    rate_limits_hit: int


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


# ── Global counters ───────────────────────────────────────────

_start_time = time.time()
_request_count = 0
_total_searches = 0


# ── Helper: Extract Client IP ─────────────────────────────────

def get_client_id(request: Request) -> str:
    """Get a client identifier for rate limiting.

    Uses X-Forwarded-For if behind a proxy, otherwise the client IP.
    """
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ── Routes ────────────────────────────────────────────────────

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: Request, body: ChatRequest):
    """OpenAI-compatible chat completions with automatic web search.

    The web_search tool is auto-injected. If the LLM decides to search,
    the middleware executes the search and feeds results back to the LLM
    automatically — the client just waits for the final answer.
    """
    global _request_count, _total_searches

    # Rate limit
    client_id = get_client_id(request)
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please slow down.",
        )

    _request_count += 1

    # Convert messages to dicts
    messages = [msg.model_dump(exclude_none=True) for msg in body.messages]

    # ── Streaming path ──────────────────────────────────────
    if body.stream:
        chatcmpl_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        async def sse_safe_wrapper():
            """Wrap the streaming generator with fallback error handling."""
            try:
                async for chunk in run_tool_loop_streaming(
                    messages=messages,
                    search_provider=get_search_provider(),
                    chatcmpl_id=chatcmpl_id,
                    created=created,
                    tools=body.tools,
                    model=body.model,
                ):
                    yield chunk
            except Exception:
                logger.exception("Unhandled error during streaming")
                error_data = json.dumps(
                    {"error": {"message": "Internal server error", "type": "internal_error"}}
                )
                yield f"data: {error_data}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            sse_safe_wrapper(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ── Non-streaming path ──────────────────────────────────
    try:
        result = await run_tool_loop(
            messages=messages,
            search_provider=get_search_provider(),
            tools=body.tools,
            model=body.model,
        )
    except LMStudioError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except ToolLoopExhaustedError as exc:
        # Return a 200 but signal that we hit the loop limit
        return JSONResponse(
            status_code=200,
            content={
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": str(exc),
                    },
                    "finish_reason": "tool_loop_max",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
        )

    _total_searches += result.get("searches", 0)

    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
    created=int(time.time()),
    model=body.model,
    choices=[
        ChatResponseChoice(
            message={
                "role": "assistant",
                "content": result["content"],
            },
        )
    ],
        usage=ChatResponseUsage(),
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — returns connectivity status for all components."""
    provider = get_search_provider()
    searxng_ok = False
    lm_studio_ok = False

    try:
        searxng_ok = await provider.health_check()
    except Exception:
        pass

    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.lm_studio_url.rstrip('/')}/models")
            lm_studio_ok = resp.status_code == 200
    except Exception:
        pass

    status = "ok" if (searxng_ok and lm_studio_ok) else "degraded"
    if not searxng_ok and not lm_studio_ok:
        status = "error"

    return HealthResponse(
        status=status,
        lm_studio_url=settings.lm_studio_url,
        search_provider=get_search_provider().name,
        searxng_ok=searxng_ok,
        lm_studio_ok=lm_studio_ok,
        cache_hit_rate=round(search_cache.hit_rate, 3),
        total_searches=_total_searches,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.get("/stats", response_model=StatsResponse)
async def stats():
    """Usage statistics."""
    return StatsResponse(
        total_requests=_request_count,
        total_searches=_total_searches,
        cache_hits=search_cache.hits,
        cache_misses=search_cache.misses,
        cache_hit_rate=round(search_cache.hit_rate, 3),
        rate_limits_hit=rate_limiter.hits,
    )


# ── Error Handlers ────────────────────────────────────────────

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )
