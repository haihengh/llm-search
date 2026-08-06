"""FastAPI server — OpenAI + Anthropic API with tool-call interception.

OpenAI endpoint:   POST /v1/chat/completions
Anthropic endpoint: POST /v1/messages
Health:            GET  /health
Stats:             GET  /stats

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
from .config import runtime_config, settings
from .rate_limiter import rate_limiter
from .search.base import SearchProvider
from .search.brave import BraveSearchProvider
from .search.searxng import SearXNGSearcher
from .search.serpapi import SerpAPIProvider
from . import __version__ as _version
from .anthropic_adapter import (
    anthropic_request_to_openai,
    anthropic_stream_from_openai,
    openai_response_to_anthropic,
)
from .responses_adapter import (
    openai_result_to_responses,
    responses_request_to_openai,
    responses_stream_from_tool_loop,
)
from .tool_loop import (
    LMStudioError,
    ToolLoopExhaustedError,
    is_context_overflow,
    run_tool_loop,
    run_tool_loop_streaming,
)

logger = logging.getLogger(__name__)

# ── Search Provider Factory ───────────────────────────────────

_search_provider: Optional[SearchProvider] = None


def create_search_provider() -> SearchProvider:
    """Build the search provider from *runtime* configuration."""
    if runtime_config.search_provider == "brave":
        if not runtime_config.search_api_key:
            raise ValueError("SEARCH_API_KEY is required when using Brave Search")
        logger.info("Using Brave Search API")
        return BraveSearchProvider(api_key=runtime_config.search_api_key)

    if runtime_config.search_provider == "serpapi":
        if not runtime_config.search_api_key:
            raise ValueError("SEARCH_API_KEY is required when using SerpAPI")
        logger.info("Using SerpAPI")
        return SerpAPIProvider(api_key=runtime_config.search_api_key)

    # Default: SearXNG
    logger.info("Using SearXNG at %s", runtime_config.searxng_url)
    return SearXNGSearcher(base_url=runtime_config.searxng_url)


def recreate_search_provider() -> Optional[SearchProvider]:
    """Recreate the search provider (called after runtime config change).

    Returns the new provider or None if recreation failed (e.g. missing API key).
    On failure the old provider is kept.
    """
    global _search_provider
    logger.info("Recreating search provider — config changed")
    try:
        new_provider = create_search_provider()
    except ValueError as exc:
        logger.warning("Cannot recreate search provider: %s — keeping old one", exc)
        return None
    if _search_provider is not None and hasattr(_search_provider, "close"):
        try:
            import asyncio
            asyncio.ensure_future(_search_provider.close())
        except Exception:
            pass
    _search_provider = new_provider
    return _search_provider


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
    logger.info("LLM Search v%s starting", _version)
    logger.info("LM Studio URL: %s", runtime_config.lm_studio_url)
    logger.info("Search provider: %s", runtime_config.search_provider)

    _search_provider = create_search_provider()

    # Register hook: recreate search provider when config changes
    def _on_config_change(key: str, old_val: Any, new_val: Any) -> None:
        provider_keys = {"search_provider", "searxng_url", "search_api_key"}
        if key in provider_keys:
            logger.info("Config %r changed — recreating search provider", key)
            recreate_search_provider()

    runtime_config.on_change(_on_config_change)

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
    version=_version,
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
    reasoning: bool = True  # if False, skip reasoning_content, relay only content


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
    total_fetch_pages: int = 0
    total_hallucinated_tools: int = 0
    total_passthrough_tools: int = 0
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    rate_limits_hit: int
    llm_call_count: int = 0
    llm_avg_ms: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    tokens_per_second: float = 0.0
    prompt_tokens_per_second: float = 0.0
    uptime_seconds: float = 0.0


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


# ── Global counters ───────────────────────────────────────────

_start_time = time.time()
_request_count = 0
_total_searches = 0
_total_fetch_pages = 0
_total_hallucinated = 0
_total_passthrough = 0
_llm_call_count = 0
_llm_total_ms = 0.0
_total_prompt_tokens = 0
_total_completion_tokens = 0
_total_prompt_ms = 0.0
_total_generation_ms = 0.0


def _ingest_request_stats(stats_dict: dict[str, Any]) -> None:
    """Merge per-request tool-loop stats into global counters."""
    global _total_fetch_pages, _total_hallucinated, _total_passthrough
    global _llm_call_count, _llm_total_ms
    global _total_prompt_tokens, _total_completion_tokens
    global _total_prompt_ms, _total_generation_ms
    _total_fetch_pages += stats_dict.get("fetch_page_count", 0)
    _total_hallucinated += stats_dict.get("hallucinated_tool_count", 0)
    _total_passthrough += stats_dict.get("passthrough_tool_count", 0)
    _llm_call_count += stats_dict.get("llm_call_count", 0)
    _llm_total_ms += stats_dict.get("llm_total_ms", 0)
    _total_prompt_tokens += stats_dict.get("prompt_tokens", 0)
    _total_completion_tokens += stats_dict.get("completion_tokens", 0)
    _total_prompt_ms += stats_dict.get("prompt_total_ms", 0)
    _total_generation_ms += stats_dict.get("generation_total_ms", 0)


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
        stats_container: list[dict[str, Any]] = []

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
                    stats_out=stats_container,
                    reasoning=body.reasoning,
                ):
                    yield chunk
            except Exception:
                logger.exception("Unhandled error during streaming")
                error_data = json.dumps(
                    {"error": {"message": "Internal server error", "type": "internal_error"}}
                )
                yield f"data: {error_data}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                if stats_container:
                    _ingest_request_stats(stats_container[0])

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
            reasoning=body.reasoning,
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
    if result.get("stats"):
        _ingest_request_stats(result["stats"])

    # Build the assistant message, including any passthrough tool calls
    assistant_msg: dict[str, Any] = {
        "role": "assistant",
        "content": result["content"],
    }
    if result.get("tool_calls"):
        assistant_msg["tool_calls"] = result["tool_calls"]

    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=body.model,
        choices=[
            ChatResponseChoice(
                message=assistant_msg,
                finish_reason=result.get("finish_reason", "stop"),
            )
        ],
        usage=ChatResponseUsage(),
    )


# ── Anthropic Messages API ─────────────────────────────────────

class AnthropicErrorResponse(BaseModel):
    type: str = "error"
    error: dict[str, Any]


@app.post("/v1/messages")
async def messages(request: Request):
    """Anthropic Messages API endpoint.

    Translates Anthropic-format requests to OpenAI internally,
    runs the tool loop, and translates responses back.

    Supports both non-streaming and streaming (stream=True in body),
    though streaming is sent as a single SSE text_delta for simplicity.
    """
    global _request_count, _total_searches

    # Rate limit
    client_id = get_client_id(request)
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(status_code=429, detail="Too many requests.")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    _request_count += 1

    # Translate Anthropic → OpenAI
    openai_req = anthropic_request_to_openai(body)
    use_stream = body.get("stream", False)
    model = body.get("model", "local-model")

    logger.info(
        "Anthropic request: model=%r stream=%s messages=%d",
        model, use_stream, len(openai_req.get("messages", [])),
    )

    try:
        if use_stream:
            # Streaming path — translate OpenAI SSE to Anthropic SSE.
            #
            # We peek at the FIRST chunk from the stream *before* wrapping
            # it in StreamingResponse so that context-overflow errors
            # (which LM Studio returns as 400s) can be surfaced as proper
            # HTTP 400 responses.  Claude Code only triggers auto-compaction
            # on HTTP-level errors, not on in-stream SSE error events.
            chatcmpl_id = f"msg_{uuid.uuid4().hex[:12]}"
            created = int(time.time())
            stats_container: list[dict[str, Any]] = []

            tool_loop_gen = run_tool_loop_streaming(
                messages=openai_req["messages"],
                search_provider=get_search_provider(),
                chatcmpl_id=chatcmpl_id,
                created=created,
                tools=openai_req["tools"],
                model=openai_req["model"],
                stats_out=stats_container,
            )
            stream_gen = anthropic_stream_from_openai(
                tool_loop_gen, model=model, request_id=chatcmpl_id,
            )

            # Grab the first SSE event without sending HTTP headers yet.
            try:
                first_chunk: str | None = await stream_gen.__anext__()
            except StopAsyncIteration:
                first_chunk = None
            except Exception as exc:
                # Any unexpected error (timeout, network, etc.) during the
                # first streaming iteration — surface as a proper API error
                # instead of a 500 so Claude Code can react gracefully.
                logger.exception("Error during first streaming chunk")
                status_code = 400 if is_context_overflow(LMStudioError(str(exc))) else 502
                return JSONResponse(
                    status_code=status_code,
                    content={
                        "type": "error",
                        "error": {
                            "type": "invalid_request_error" if status_code == 400 else "api_error",
                            "message": str(exc),
                        },
                    },
                )

            # If the very first event is an error, return it as a plain
            # HTTP error so Claude Code can react (auto-compact, etc.).
            if first_chunk and first_chunk.startswith("event: error"):
                try:
                    # Extract the JSON payload from the SSE data line.
                    data_line = first_chunk.split("\ndata: ")[1].split("\n")[0]
                    error_data = json.loads(data_line)
                    error_info = error_data.get("error", {})
                    error_type: str = error_info.get("type", "api_error")
                    error_msg: str = error_info.get("message", "Unknown error")
                except (IndexError, json.JSONDecodeError):
                    error_type, error_msg = "api_error", "Unknown error"

                status_code = 400 if error_type == "invalid_request_error" else 502
                return JSONResponse(
                    status_code=status_code,
                    content={
                        "type": "error",
                        "error": {"type": error_type, "message": error_msg},
                    },
                )

            # Normal path — stream the remaining events (first_chunk
            # prepended so nothing is lost).
            async def sse_wrapper():
                if first_chunk is not None:
                    yield first_chunk
                try:
                    async for chunk in stream_gen:
                        yield chunk
                except Exception:
                    logger.exception("Error in Anthropic streaming")
                    yield (
                        "event: message_stop\n"
                        'data: {"type":"message_stop"}\n\n'
                    )
                finally:
                    if stats_container:
                        _ingest_request_stats(stats_container[0])

            return StreamingResponse(
                sse_wrapper(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        # Non-streaming path
        result = await run_tool_loop(
            messages=openai_req["messages"],
            search_provider=get_search_provider(),
            tools=openai_req["tools"],
            model=openai_req["model"],
        )

        _total_searches += result.get("searches", 0)
        if result.get("stats"):
            _ingest_request_stats(result["stats"])

        return openai_response_to_anthropic(result, model)

    except LMStudioError as exc:
        # Anthropic-shaped error body so Claude Code can react properly —
        # a context overflow becomes "prompt is too long", which triggers
        # Claude Code's auto-compaction instead of an opaque failure.
        if is_context_overflow(exc):
            status_code, error_type = 400, "invalid_request_error"
            message = f"prompt is too long: {exc}"
        else:
            status_code, error_type = 502, "api_error"
            message = str(exc)
        return JSONResponse(
            status_code=status_code,
            content={"type": "error", "error": {"type": error_type, "message": message}},
        )
    except ToolLoopExhaustedError as exc:
        return openai_response_to_anthropic(
            {"content": str(exc), "tool_calls_count": 0, "iterations": 5, "searches": 0},
            model,
        )



# ── OpenAI Responses API (Codex / GPT-5.x clients) ────────────────

@app.post("/v1/responses")
async def responses_api(request: Request):
    """OpenAI Responses API endpoint for Codex and GPT-5.x clients.

    Translates Responses API format ↔ Chat Completions internally.
    Supports both streaming and non-streaming.
    """
    global _request_count, _total_searches

    client_id = get_client_id(request)
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(status_code=429, detail="Too many requests.")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    _request_count += 1

    # Translate Responses → Chat Completions
    openai_req = responses_request_to_openai(body)
    model = body.get("model", "local-model")
    use_stream = body.get("stream", False)

    logger.info(
        "Responses request: model=%r messages=%d tools=%d stream=%s",
        model,
        len(openai_req.get("messages", [])),
        len(openai_req.get("tools") or []),
        use_stream,
    )

    # ── Streaming path ──────────────────────────────────────────
    if use_stream:
        resp_id = f"resp_{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        chatcmpl_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        stats_container: list[dict[str, Any]] = []

        tool_loop_gen = run_tool_loop_streaming(
            messages=openai_req["messages"],
            search_provider=get_search_provider(),
            chatcmpl_id=chatcmpl_id,
            created=created,
            tools=openai_req["tools"],
            model=openai_req["model"],
            stats_out=stats_container,
        )
        stream_gen = responses_stream_from_tool_loop(
            tool_loop_gen, model=model, resp_id=resp_id,
        )

        # Peek at the first chunk to catch early errors
        try:
            first_chunk: str | None = await stream_gen.__anext__()
        except StopAsyncIteration:
            first_chunk = None
        except Exception as exc:
            logger.exception("Error during first streaming chunk (Responses)")
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "code": "server_error",
                        "message": str(exc),
                    }
                },
            )

        if first_chunk and first_chunk.startswith("event: error"):
            try:
                data_line = first_chunk.split("\ndata: ")[1].split("\n")[0]
                error_data = json.loads(data_line)
                error_info = error_data.get("error", {})
                msg = error_info.get("message", "Unknown error")
            except (IndexError, json.JSONDecodeError):
                msg = "Unknown error"
            return JSONResponse(
                status_code=502,
                content={"error": {"code": "server_error", "message": msg}},
            )

        async def sse_wrapper():
            if first_chunk is not None:
                yield first_chunk
            try:
                async for chunk in stream_gen:
                    yield chunk
            except Exception:
                logger.exception("Error in Responses streaming")
                yield (
                    "event: response.completed\n"
                    f'data: {{"type":"response.completed","response":{{"id":"{resp_id}","object":"response","status":"incomplete","model":"{model}","output":[]}}}}\n\n'
                )
            finally:
                if stats_container:
                    _ingest_request_stats(stats_container[0])

        return StreamingResponse(
            sse_wrapper(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ── Non-streaming path ──────────────────────────────────────
    try:
        result = await run_tool_loop(
            messages=openai_req["messages"],
            search_provider=get_search_provider(),
            tools=openai_req["tools"],
            model=openai_req["model"],
        )
    except LMStudioError as exc:
        # Responses-shaped error
        status_code = 400 if is_context_overflow(exc) else 502
        return JSONResponse(
            status_code=status_code,
            content={
                "error": {
                    "code": "invalid_request_error" if status_code == 400 else "server_error",
                    "message": str(exc),
                }
            },
        )
    except ToolLoopExhaustedError as exc:
        result = {"content": str(exc), "tool_calls_count": 0, "iterations": 5, "searches": 0}

    _total_searches += result.get("searches", 0)
    if result.get("stats"):
        _ingest_request_stats(result["stats"])

    return openai_result_to_responses(result, model)


@app.get("/v1/models")
async def list_models():
    """Proxy the models list from the LLM backend.

    Chat clients call this to discover available models. The middleware
    proxies the request to LM Studio / Ollama so clients see the full
    model catalog, no matter what machine the LLM is running on.
    """
    import httpx

    url = f"{runtime_config.lm_studio_url.rstrip('/')}/models"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()
    except httpx.ConnectError:
        raise HTTPException(
            status_code=502,
            detail=f"LLM backend not reachable at {runtime_config.lm_studio_url}",
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM backend returned {exc.response.status_code}",
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
            resp = await client.get(f"{runtime_config.lm_studio_url.rstrip('/')}/models")
            lm_studio_ok = resp.status_code == 200
    except Exception:
        pass

    status = "ok" if (searxng_ok and lm_studio_ok) else "degraded"
    if not searxng_ok and not lm_studio_ok:
        status = "error"

    return HealthResponse(
        status=status,
        lm_studio_url=runtime_config.lm_studio_url,
        search_provider=get_search_provider().name,
        searxng_ok=searxng_ok,
        lm_studio_ok=lm_studio_ok,
        cache_hit_rate=round(search_cache.hit_rate, 3),
        total_searches=_total_searches,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.get("/stats", response_model=StatsResponse)
async def stats():
    """Usage statistics with LLM performance and tool-call breakdowns."""
    avg_ms = round(_llm_total_ms / _llm_call_count, 1) if _llm_call_count > 0 else 0.0
    # Prefer backend-reported timing; fall back to wall-clock LLM total time.
    gen_ms = _total_generation_ms if _total_generation_ms > 0 else _llm_total_ms
    prompt_ms = _total_prompt_ms if _total_prompt_ms > 0 else _llm_total_ms
    tps = round(_total_completion_tokens / (gen_ms / 1000), 1) if gen_ms > 0 and _total_completion_tokens > 0 else 0.0
    pps = round(_total_prompt_tokens / (prompt_ms / 1000), 1) if prompt_ms > 0 and _total_prompt_tokens > 0 else 0.0
    return StatsResponse(
        total_requests=_request_count,
        total_searches=_total_searches,
        total_fetch_pages=_total_fetch_pages,
        total_hallucinated_tools=_total_hallucinated,
        total_passthrough_tools=_total_passthrough,
        cache_hits=search_cache.hits,
        cache_misses=search_cache.misses,
        cache_hit_rate=round(search_cache.hit_rate, 3),
        rate_limits_hit=rate_limiter.hits,
        llm_call_count=_llm_call_count,
        llm_avg_ms=avg_ms,
        total_prompt_tokens=_total_prompt_tokens,
        total_completion_tokens=_total_completion_tokens,
        tokens_per_second=tps,
        prompt_tokens_per_second=pps,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


# ── Runtime Config API ─────────────────────────────────────

class ConfigUpdateRequest(BaseModel):
    """Fields for updating runtime config. All are optional."""
    lm_studio_url: Optional[str] = None
    search_provider: Optional[str] = None
    searxng_url: Optional[str] = None
    search_api_key: Optional[str] = None
    max_tool_loop_iterations: Optional[int] = None
    max_client_tools: Optional[int] = None
    lm_studio_timeout: Optional[float] = None
    max_search_results: Optional[int] = None


@app.get("/v1/config")
async def get_config():
    """Return current runtime configuration (editable fields only)."""
    return {
        "config": runtime_config.to_dict(),
        "immutable": {
            "middleware_host": settings.middleware_host,
            "middleware_port": settings.middleware_port,
            "rate_limit_per_minute": settings.rate_limit_per_minute,
            "search_cache_ttl_seconds": settings.search_cache_ttl_seconds,
            "log_level": settings.log_level,
        },
    }


@app.put("/v1/config")
async def update_config(body: ConfigUpdateRequest):
    """Update runtime configuration. Only provided fields are changed."""
    updates = body.model_dump(exclude_none=True)
    if not updates:
        return {"changed": {}, "config": runtime_config.to_dict()}

    changed = runtime_config.update(**updates)
    logger.info("Runtime config updated: %s", changed)
    return {"changed": changed, "config": runtime_config.to_dict()}


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
