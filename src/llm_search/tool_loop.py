"""The core tool-call intercept loop.

This is the engine: it sends the user's messages to LM Studio,
intercepts any web_search tool calls, executes them, feeds results
back to the LLM, and loops until the LLM produces a plain text answer.
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional

import httpx

from .config import runtime_config
from .search.base import SearchProvider
from .tool_registry import (
    FETCH_PAGE,
    FETCH_PAGE_TOOL,
    TOOL_EXECUTORS,
    WEB_SEARCH,
    WEB_SEARCH_TOOL,
    dispatch_tool,
    inject_tools,
)

logger = logging.getLogger(__name__)


class ToolLoopExhaustedError(Exception):
    """Raised when the tool-call loop exceeds max iterations."""


class LMStudioError(Exception):
    """Raised when LM Studio returns an error or is unreachable."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


_CONTEXT_OVERFLOW_MARKERS = (
    "n_ctx",
    "n_keep",
    "context length",
    "context_length",
    "context window",
    "context size",
    "exceed",
    "too long",
)


def is_context_overflow(exc: LMStudioError) -> bool:
    """True when LM Studio rejected the request because the prompt
    exceeds the loaded model's context window.

    llama.cpp phrases this as e.g. "The number of tokens to keep from
    the initial prompt is greater than the context length (n_keep: X
    >= n_ctx: Y)", or LM Studio's "request (N tokens) exceeds the
    available context size (M tokens)". MLX and other backends use
    similar wording.
    """
    if exc.status_code is not None and exc.status_code != 400:
        return False
    msg = str(exc).lower()
    return any(marker in msg for marker in _CONTEXT_OVERFLOW_MARKERS)


# ── Reasoning Toggle ──────────────────────────────────────────

_NO_REASONING_PROMPT = (
    "You are in direct-answer mode. Do NOT use reasoning, chain of thought, "
    "or step-by-step thinking. Answer concisely and directly — give only "
    "the final answer with no preamble, no thinking process, and no "
    "explanation of your reasoning."
)


def _inject_no_reasoning_prompt(conversation: list[dict[str, Any]]) -> None:
    """Prepend or extend a system message to disable reasoning."""
    if conversation and conversation[0].get("role") == "system":
        conversation[0]["content"] = (
            conversation[0]["content"] + "\n\n" + _NO_REASONING_PROMPT
        )
    else:
        conversation.insert(0, {"role": "system", "content": _NO_REASONING_PROMPT})


# ── Stats Collector ────────────────────────────────────────────

@dataclass
class ToolLoopStats:
    """Per-request statistics collected during the tool loop."""

    llm_call_count: int = 0
    llm_total_ms: float = 0.0
    web_search_count: int = 0
    fetch_page_count: int = 0
    hallucinated_tool_count: int = 0
    passthrough_tool_count: int = 0
    total_iterations: int = 0

    # Token-level timing (extracted from LM Studio usage + timings)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_total_ms: float = 0.0          # time spent processing the prompt
    generation_total_ms: float = 0.0       # time spent generating tokens

    # Timestamps for the last LLM call
    last_llm_ms: float = 0.0
    _llm_timings: list[float] = field(default_factory=list)

    def record_llm_call(self, elapsed_ms: float) -> None:
        self.llm_call_count += 1
        self.llm_total_ms += elapsed_ms
        self.last_llm_ms = elapsed_ms
        self._llm_timings.append(elapsed_ms)

    def record_token_usage(
        self,
        usage: dict[str, Any] | None,
        timings: dict[str, Any] | None = None,
    ) -> None:
        """Ingest usage + timings from an LM Studio / llama.cpp response."""
        if usage:
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
        if timings:
            self.prompt_total_ms += timings.get("prompt_ms", 0)
            self.generation_total_ms += timings.get("predicted_ms", 0)

    @property
    def llm_avg_ms(self) -> float:
        if self.llm_call_count == 0:
            return 0.0
        return self.llm_total_ms / self.llm_call_count

    @property
    def tokens_per_second(self) -> float:
        """Estimated completion tokens per second.

        Uses generation_total_ms if the backend provides it (llama.cpp
        ``timings.predicted_ms``), otherwise falls back to wall-clock
        total LLM time (which is a slight underestimate since it
        includes prompt processing).
        """
        if self.generation_total_ms > 0:
            return self.completion_tokens / (self.generation_total_ms / 1000)
        # Fallback: wall-clock time (prompt processing is typically fast
        # enough that this is a reasonable approximation).
        if self.completion_tokens > 0 and self.llm_total_ms > 0:
            return self.completion_tokens / (self.llm_total_ms / 1000)
        return 0.0

    @property
    def prompt_tokens_per_second(self) -> float:
        """Estimated prompt processing speed (tokens/sec).

        Uses prompt_total_ms if available, otherwise wall-clock fallback.
        """
        if self.prompt_total_ms > 0:
            return self.prompt_tokens / (self.prompt_total_ms / 1000)
        if self.prompt_tokens > 0 and self.llm_total_ms > 0:
            return self.prompt_tokens / (self.llm_total_ms / 1000)
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "llm_call_count": self.llm_call_count,
            "llm_total_ms": round(self.llm_total_ms, 1),
            "llm_avg_ms": round(self.llm_avg_ms, 1),
            "last_llm_ms": round(self.last_llm_ms, 1),
            "web_search_count": self.web_search_count,
            "fetch_page_count": self.fetch_page_count,
            "hallucinated_tool_count": self.hallucinated_tool_count,
            "passthrough_tool_count": self.passthrough_tool_count,
            "total_iterations": self.total_iterations,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "prompt_total_ms": round(self.prompt_total_ms, 1),
            "generation_total_ms": round(self.generation_total_ms, 1),
            "tokens_per_second": round(self.tokens_per_second, 1),
            "prompt_tokens_per_second": round(self.prompt_tokens_per_second, 1),
        }


# ── LM Studio Chat Client ─────────────────────────────────────

async def call_lm_studio(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    model: str,
    lm_studio_url: str,
    max_tokens: Optional[int] = None,
) -> dict[str, Any]:
    """Send a chat completion request to LM Studio.

    Returns the full response JSON. Raises LMStudioError on failure.
    """
    url = f"{lm_studio_url.rstrip('/')}/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
    if max_tokens:
        payload["max_tokens"] = max_tokens

    timeout = runtime_config.lm_studio_timeout
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.ConnectError:
            raise LMStudioError(f"LM Studio not reachable at {lm_studio_url}")
        except (httpx.TimeoutException, httpx.ReadTimeout):
            raise LMStudioError(
                f"LM Studio request timed out after {timeout}s "
                f"(model may be slow with many tools — try increasing "
                f"LM_STUDIO_TIMEOUT or reducing the number of client tools)"
            )
        except httpx.HTTPStatusError as exc:
            raise LMStudioError(
                f"LM Studio returned {exc.response.status_code}: "
                f"{exc.response.text[:500]}",
                status_code=exc.response.status_code,
            )


# ── Streaming LM Studio Client ─────────────────────────────────

async def call_lm_studio_streaming(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    model: str,
    lm_studio_url: str,
    max_tokens: Optional[int] = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Send a streaming chat completion request to LM Studio.

    Yields each parsed SSE data event as a dict. The caller should
    reconstruct content or tool calls from the delta chunks.
    Stops when `data: [DONE]` is received.

    Raises LMStudioError on connection failure or HTTP error.
    """
    url = f"{lm_studio_url.rstrip('/')}/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    if tools:
        payload["tools"] = tools
    if max_tokens:
        payload["max_tokens"] = max_tokens

    timeout = runtime_config.lm_studio_timeout
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise LMStudioError(
                        f"LM Studio returned {response.status_code}: "
                        f"{body.decode(errors='replace')[:500]}",
                        status_code=response.status_code,
                    )
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data = line[6:]  # Strip "data: " prefix
                        if data == "[DONE]":
                            return
                        try:
                            chunk_data = json.loads(data)
                        except json.JSONDecodeError:
                            logger.debug("Skipping unparseable SSE line: %s", line[:100])
                            continue
                        # LM Studio streams HTTP-level failures as an SSE
                        # "event: error" chunk with HTTP 200 (observed for
                        # context overflow: "request (N tokens) exceeds the
                        # available context size"). A chunk carrying "error"
                        # is not a generation chunk — raise so the tool
                        # loop's normal LMStudioError handling applies.
                        if isinstance(chunk_data, dict) and "error" in chunk_data:
                            err = chunk_data["error"]
                            if isinstance(err, dict):
                                raise LMStudioError(
                                    err.get("message", "LM Studio streaming error"),
                                    status_code=err.get("code") or 400,
                                )
                        yield chunk_data
        except httpx.ConnectError:
            raise LMStudioError(f"LM Studio not reachable at {lm_studio_url}")
        except (httpx.TimeoutException, httpx.ReadTimeout):
            raise LMStudioError(
                f"LM Studio streaming request timed out after "
                f"{timeout}s per read — the model may be "
                f"generating too slowly or hung"
            )
        except httpx.HTTPStatusError as exc:
            raise LMStudioError(
                f"LM Studio returned {exc.response.status_code}: "
                f"{exc.response.text[:500]}",
                status_code=exc.response.status_code,
            )


# ── Response Parsing ──────────────────────────────────────────

def extract_assistant_message(
    response: dict[str, Any],
) -> tuple[Optional[str], list[dict[str, Any]]]:
    """Extract text content and tool calls from an LM Studio response.

    Returns (content, tool_calls). One or both may be present.
    """
    choices = response.get("choices", [])
    if not choices:
        return None, []

    message = choices[0].get("message", {})
    content = message.get("content")
    tool_calls = message.get("tool_calls", [])

    return content, tool_calls


# ── The Tool Loop ─────────────────────────────────────────────

async def run_tool_loop(
    messages: list[dict[str, Any]],
    search_provider: SearchProvider,
    *,
    tools: Optional[list[dict[str, Any]]] = None,
    model: str = "local-model",
    lm_studio_url: Optional[str] = None,
    reasoning: bool = True,
    max_tokens: Optional[int] = None,
) -> dict[str, Any]:
    """Execute the tool-call intercept loop.

    1. Send messages + all tools (client tools + search tools) to LM Studio
    2. If LLM calls web_search / fetch_page → execute, feed back, repeat
    3. If LLM calls a client-provided tool → return it as passthrough
    4. If LLM hallucinates an unknown tool → feed error, let it recover
    5. Return the final assistant message as an OpenAI-format dict

    Args:
        messages: Chat messages (OpenAI format)
        search_provider: Where to execute searches
        tools: Optional client-provided tools (web_search + fetch_page
               are auto-injected alongside them)
        model: Model name to pass to LM Studio
        lm_studio_url: URL of the LM Studio API

    Returns:
        Dict with keys: content, tool_calls (passthrough), tool_calls_count,
        iterations, searches, finish_reason

    Raises:
        ToolLoopExhaustedError: Max iterations exceeded
        LMStudioError: LM Studio is unreachable or errors
    """
    url = lm_studio_url or runtime_config.lm_studio_url
    max_iter = runtime_config.max_tool_loop_iterations

    # Merge client-provided tools with auto-injected search tools.
    # The LLM sees client tools + web_search + fetch_page. To prevent
    # overwhelming small local models, we cap the number of client tools
    # via max_client_tools (env: MAX_CLIENT_TOOLS, default 4). Set to 0
    # to disable client tool passthrough entirely.
    all_tools = inject_tools(tools)
    client_tool_names: set[str] = set()
    if tools:
        for t in tools:
            func = t.get("function") if isinstance(t, dict) else None
            if isinstance(func, dict):
                name = func.get("name", "")
                if name:
                    client_tool_names.add(name)

    # Trim client tools to the configured limit (search tools are always sent)
    if runtime_config.max_client_tools >= 0 and len(client_tool_names) > runtime_config.max_client_tools:
        overflow = len(client_tool_names) - runtime_config.max_client_tools
        logger.info(
            "Trimming %d client tools (limit: %d) — set MAX_CLIENT_TOOLS to "
            "increase or 0 to disable passthrough",
            overflow, runtime_config.max_client_tools,
        )
        # Keep only the first N client tools by preserving order from all_tools
        kept_client: set[str] = set()
        trimmed_tools: list[dict[str, Any]] = []
        for t in all_tools:
            name = (t.get("function") or {}).get("name", "")
            if name in TOOL_EXECUTORS:
                # Always keep search tools
                trimmed_tools.append(t)
            elif name in client_tool_names:
                if len(kept_client) < runtime_config.max_client_tools:
                    kept_client.add(name)
                    trimmed_tools.append(t)
                # else: skip this client tool (trimmed)
            else:
                trimmed_tools.append(t)
        all_tools = trimmed_tools
        # Update client_tool_names to only include tools actually sent
        client_tool_names = kept_client

    conversation = list(messages)  # Copy — we'll mutate this

    # When reasoning is disabled, inject a system prompt telling the
    # model to skip chain-of-thought and answer directly.
    if not reasoning:
        _inject_no_reasoning_prompt(conversation)

    total_searches = 0
    total_tool_calls = 0
    stats = ToolLoopStats()

    # Track seen queries / URLs so we can tell the model when it's
    # repeating itself instead of silently re-executing the same search.
    seen_queries: set[str] = set()
    seen_urls: set[str] = set()

    for iteration in range(1, max_iter + 1):
        logger.debug("Tool loop iteration %d/%d", iteration, max_iter)

        # Nudge the LLM to synthesize an answer instead of searching
        # again. Small local models often get stuck in a search→search→search
        # loop without these reminders.
        #   - Iteration 3: gentle hint (still early, may need more searches)
        #   - Last 2 iterations: forceful "stop searching now"
        if total_searches > 0:
            if iteration == 3 and iteration < max_iter - 1:
                conversation.append({
                    "role": "user",
                    "content": (
                        "You now have search results. If the information is "
                        "sufficient to answer the user's question, please "
                        "respond directly instead of searching again. Only "
                        "search again if you need completely different information."
                    ),
                })
            elif iteration >= max_iter - 1:
                conversation.append({
                    "role": "user",
                    "content": (
                        "You now have search results. Please synthesize a "
                        "final answer based on what you found. Do NOT call "
                        "web_search again — answer the user's question now."
                    ),
                })

        # Send to LM Studio
        t0 = time.monotonic()
        response = await call_lm_studio(
            messages=conversation,
            tools=all_tools,
            model=model,
            lm_studio_url=url,
            max_tokens=max_tokens,
        )
        stats.record_llm_call((time.monotonic() - t0) * 1000)
        stats.record_token_usage(
            response.get("usage"),
            response.get("timings"),
        )

        content, tool_calls = extract_assistant_message(response)

        # No tool calls → LLM is done, return the answer
        if not tool_calls:
            finish_reason = (response.get("choices") or [{}])[0].get("finish_reason")
            if not content and finish_reason == "length":
                # Generation was truncated before any answer text was
                # produced — the prompt (or the model's chain-of-thought)
                # exhausted the context window. Reasoning models burn the
                # whole budget on reasoning_content and yield empty content
                # when they hit the wall, which looks identical to a dead
                # model. Surface it as a context overflow so clients that
                # auto-compact on "prompt is too long" can recover.
                raise LMStudioError(
                    "prompt is too long: the model exhausted its context "
                    "window before producing an answer "
                    "(finish_reason=length, no content, no tool calls)",
                    status_code=400,
                )
            stats.total_iterations = iteration
            return {
                "content": content or "",
                "tool_calls_count": total_tool_calls,
                "iterations": iteration,
                "searches": total_searches,
                "stats": stats.to_dict(),
            }

        # Build the assistant message with tool_calls to append to conversation
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
        }
        conversation.append(assistant_message)

        # Three-way classification:
        #   search tools  → execute server-side, feed result, continue loop
        #   client tools  → passthrough to caller for execution
        #   hallucinations → names match nothing — feed error, let LLM recover
        search_tool_calls = []
        passthrough_tool_calls = []
        hallucinated_tool_calls = []
        for tc in tool_calls:
            tool_name = tc.get("function", {}).get("name", "")
            if tool_name in TOOL_EXECUTORS:
                search_tool_calls.append(tc)
            elif tool_name in client_tool_names:
                passthrough_tool_calls.append(tc)
                stats.passthrough_tool_count += 1
                logger.info("Client tool %r — will passthrough to caller", tool_name)
            elif tool_name:
                hallucinated_tool_calls.append(tc)
                stats.hallucinated_tool_count += 1
                logger.info("Hallucinated tool %r — will block", tool_name)
            else:
                logger.warning(
                    "Dropping malformed tool call with empty name: %s",
                    json.dumps(tc)[:200],
                )

        # Execute search tools and feed results back to conversation
        for tc in search_tool_calls:
            tool_name = tc.get("function", {}).get("name", "")

            # Parse arguments
            try:
                raw_args = tc.get("function", {}).get("arguments", "{}")
                if isinstance(raw_args, str):
                    arguments = json.loads(raw_args)
                else:
                    arguments = raw_args
            except json.JSONDecodeError:
                arguments = {}

            logger.info("Tool call: %s(%s)", tool_name, arguments)

            # ── Duplicate detection ──────────────────────────
            # If the model repeats the exact same query or URL, tell it
            # the results are already in the conversation instead of
            # re-executing (and returning byte-identical cached results).
            dedup_key: str | None = None
            if tool_name == WEB_SEARCH:
                dedup_key = arguments.get("query", "").strip().lower()
            elif tool_name == FETCH_PAGE:
                dedup_key = arguments.get("url", "").strip().lower()

            if dedup_key and (
                (tool_name == WEB_SEARCH and dedup_key in seen_queries)
                or (tool_name == FETCH_PAGE and dedup_key in seen_urls)
            ):
                result_text = (
                    f"You already searched for this — the results are "
                    f"earlier in this conversation. Use them to answer "
                    f"the user. If you need DIFFERENT information, try "
                    f"a more specific or alternative query."
                )
                logger.info(
                    "Duplicate %s blocked: %r — fed hint to LLM",
                    tool_name, dedup_key,
                )
            else:
                # Track the query / URL before executing
                if dedup_key and tool_name == WEB_SEARCH:
                    seen_queries.add(dedup_key)
                elif dedup_key and tool_name == FETCH_PAGE:
                    seen_urls.add(dedup_key)

                # Track searches
                if tool_name == WEB_SEARCH:
                    total_searches += 1
                    stats.web_search_count += 1
                elif tool_name == FETCH_PAGE:
                    stats.fetch_page_count += 1

                # Execute the tool
                result_text = await dispatch_tool(
                    tool_name=tool_name,
                    arguments=arguments,
                    search_provider=search_provider,
                )
            total_tool_calls += 1

            # Append tool result to conversation
            tool_message = {
                "role": "tool",
                "tool_call_id": tc.get("id", f"call_{total_tool_calls}"),
                "name": tool_name,
                "content": result_text,
            }
            conversation.append(tool_message)

        # Hallucinated tools — names the model invented that match
        # neither our search tools nor the client's tools. Feed an error
        # back to the LLM so it can recover and try a different approach.
        for tc in hallucinated_tool_calls:
            tool_name = tc.get("function", {}).get("name", "unknown")
            tool_message = {
                "role": "tool",
                "tool_call_id": tc.get("id", f"call_{total_tool_calls}"),
                "name": tool_name,
                "content": (
                    f"Error: the '{tool_name}' tool is not available. "
                    "You only have web_search and fetch_page. "
                    "Please use web_search to find the information you need."
                ),
            }
            conversation.append(tool_message)
            total_tool_calls += 1
            logger.info(
                "Blocked hallucinated tool %r — fed error back to LLM", tool_name
            )

        if hallucinated_tool_calls:
            # Some tools were hallucinations. Continue the loop so the
            # LLM can adjust its approach.
            continue

        # Client tools — the LLM called a tool the client provided.
        # Stop the loop and return the tool calls for the client to
        # execute (e.g. Claude Code's bash, read, write).
        if passthrough_tool_calls:
            logger.info(
                "Passthrough %d client tool(s) to caller: %s",
                len(passthrough_tool_calls),
                [tc.get("function", {}).get("name") for tc in passthrough_tool_calls],
            )
            stats.total_iterations = iteration
            return {
                "content": content or "",
                "tool_calls": passthrough_tool_calls,
                "tool_calls_count": total_tool_calls,
                "iterations": iteration,
                "searches": total_searches,
                "finish_reason": "tool_use",
                "stats": stats.to_dict(),
            }

        # Loop continues — LLM sees the search results and responds

    # Max iterations exceeded — build a graceful fallback from
    # accumulated search results rather than throwing an error.
    # Claude Code and other clients can still make use of the raw
    # search results even if the LLM didn't produce a final answer.
    search_result_texts: list[str] = []
    for msg in conversation:
        if msg.get("role") == "tool" and msg.get("name") == WEB_SEARCH:
            search_result_texts.append(msg.get("content", ""))

    if search_result_texts:
        fallback_content = (
            "I searched multiple times but was unable to synthesize a final answer. "
            "Here are the raw search results:\n\n" +
            "\n---\n".join(search_result_texts)
        )
    else:
        fallback_content = (
            f"Tool loop exceeded maximum iterations ({max_iter}). "
            f"No search results were collected across {max_iter} iterations. "
            f"Last response had {total_tool_calls} tool calls."
        )

    stats.total_iterations = max_iter
    return {
        "content": fallback_content,
        "tool_calls_count": total_tool_calls,
        "iterations": max_iter,
        "searches": total_searches,
        "finish_reason": "tool_loop_max",
        "stats": stats.to_dict(),
    }


# ── Streaming Tool Loop ────────────────────────────────────────

async def run_tool_loop_streaming(
    messages: list[dict[str, Any]],
    search_provider: SearchProvider,
    *,
    chatcmpl_id: str = "",
    created: int = 0,
    tools: Optional[list[dict[str, Any]]] = None,
    model: str = "local-model",
    lm_studio_url: Optional[str] = None,
    stats_out: Optional[list[dict[str, Any]]] = None,
    reasoning: bool = True,
    relay_reasoning: Optional[bool] = None,
    max_tokens: Optional[int] = None,
) -> AsyncGenerator[str, None]:
    """Execute the tool-call loop with single-pass streaming on every turn.

    The LLM is called with stream=True from the first call. Text deltas
    are relayed live to the caller. Tool-call fragments are accumulated
    across chunks — search tools (web_search / fetch_page) are executed
    server-side, results are appended to the conversation, and a new
    streaming call begins. Client-provided tools are emitted as SSE delta
    chunks for passthrough to the caller. Hallucinated tools are blocked
    with an error fed back to the LLM so it can recover.

    Unlike the old "check + re-generate" approach this issues exactly
    one LM Studio call per iteration — no second generation that can
    diverge from the first one.

    If *stats_out* is provided, the final stats dict is appended to it
    so the caller can aggregate per-request metrics without parsing the
    stream.

    Two independent reasoning switches:
      *reasoning*        — whether the model should think at all. When
                           False a system prompt tells it to skip
                           chain-of-thought entirely.
      *relay_reasoning*  — whether any ``reasoning_content`` the model
                           does emit is streamed to the caller as text.
                           Defaults to *reasoning*. API clients (Anthropic
                           / Responses) pass False: chain-of-thought is
                           deliberation, not assistant speech, and showing
                           it as the reply makes the model's private
                           "let me step back and rethink" monologue look
                           like its answer.

    Chain-of-thought is never written into the conversation history
    regardless of these flags — feeding it back makes the model treat
    its own thinking as statements it already made out loud.

    Yields:
        SSE-formatted strings: ``data: {json}\\n\\n`` per chunk.
        Terminates with ``data: [DONE]\\n\\n``.
    """
    url = lm_studio_url or runtime_config.lm_studio_url
    max_iter = runtime_config.max_tool_loop_iterations
    if relay_reasoning is None:
        relay_reasoning = reasoning

    # Merge client-provided tools with auto-injected search tools.
    # The LLM sees client tools + web_search + fetch_page. To prevent
    # overwhelming small local models, we cap the number of client tools
    # via max_client_tools (env: MAX_CLIENT_TOOLS, default 4). Set to 0
    # to disable client tool passthrough entirely.
    all_tools = inject_tools(tools)
    client_tool_names: set[str] = set()
    if tools:
        for t in tools:
            func = t.get("function") if isinstance(t, dict) else None
            if isinstance(func, dict):
                name = func.get("name", "")
                if name:
                    client_tool_names.add(name)

    # Trim client tools to the configured limit (search tools are always sent)
    if runtime_config.max_client_tools >= 0 and len(client_tool_names) > runtime_config.max_client_tools:
        overflow = len(client_tool_names) - runtime_config.max_client_tools
        logger.info(
            "Trimming %d client tools (limit: %d) — set MAX_CLIENT_TOOLS to "
            "increase or 0 to disable passthrough",
            overflow, runtime_config.max_client_tools,
        )
        kept_client: set[str] = set()
        trimmed_tools: list[dict[str, Any]] = []
        for t in all_tools:
            name = (t.get("function") or {}).get("name", "")
            if name in TOOL_EXECUTORS:
                trimmed_tools.append(t)
            elif name in client_tool_names:
                if len(kept_client) < runtime_config.max_client_tools:
                    kept_client.add(name)
                    trimmed_tools.append(t)
            else:
                trimmed_tools.append(t)
        all_tools = trimmed_tools
        client_tool_names = kept_client

    conversation = list(messages)

    # When reasoning is disabled, inject a system prompt telling the
    # model to skip chain-of-thought and answer directly.
    if not reasoning:
        _inject_no_reasoning_prompt(conversation)

    total_searches = 0
    total_tool_calls = 0
    relayed_text = False  # True once any text content has been sent to caller
    stats = ToolLoopStats()

    # Track seen queries / URLs so we can tell the model when it's
    # repeating itself instead of silently re-executing the same search.
    seen_queries: set[str] = set()
    seen_urls: set[str] = set()

    def _sse(data: dict[str, Any]) -> str:
        """Format a dict as an SSE data event."""
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    def _stats_sse() -> str:
        """Format current stats as an SSE named event."""
        return f"event: stats\ndata: {json.dumps(stats.to_dict(), ensure_ascii=False)}\n\n"

    def _error_sse(message: str, error_type: str) -> str:
        """Format an error as an SSE data event."""
        return _sse({"error": {"message": message, "type": error_type}})

    def _chunk_sse(delta: dict[str, Any], finish_reason: Optional[str] = None) -> str:
        """Build an OpenAI-compatible streaming chunk."""
        return _sse({
            "id": chatcmpl_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }],
        })

    try:
        for iteration in range(1, max_iter + 1):
            logger.debug("Tool loop (streaming) iteration %d/%d", iteration, max_iter)

            # Nudge the LLM to synthesize an answer instead of searching
            # again. Small local models often get stuck in a search→search→search
            # loop without these reminders.
            #   - Iteration 3: gentle hint (still early, may need more searches)
            #   - Last 2 iterations: forceful "stop searching now"
            if total_searches > 0:
                if iteration == 3 and iteration < max_iter - 1:
                    conversation.append({
                        "role": "user",
                        "content": (
                            "You now have search results. If the information is "
                            "sufficient to answer the user's question, please "
                            "respond directly instead of searching again. Only "
                            "search again if you need completely different information."
                        ),
                    })
                elif iteration >= max_iter - 1:
                    conversation.append({
                        "role": "user",
                        "content": (
                            "You now have search results. Please synthesize a "
                            "final answer based on what you found. Do NOT call "
                            "web_search again — answer the user's question now."
                        ),
                    })

            # ── Single-pass streaming call ──────────────────────
            had_role = False
            content_parts: list[str] = []      # real answer text
            reasoning_parts: list[str] = []    # chain-of-thought, never history
            tool_fragments: dict[int, dict[str, Any]] = {}
            chunk_count = 0
            last_finish_reason: Optional[str] = None
            t0 = time.monotonic()

            async for chunk in call_lm_studio_streaming(
                messages=conversation,
                tools=all_tools,
                model=model,
                lm_studio_url=url,
                max_tokens=max_tokens,
            ):
                chunk_count += 1
                choices = chunk.get("choices", [])
                if not choices:
                    logger.debug("Streaming chunk %d: no choices, skipping", chunk_count)
                    continue
                delta = choices[0].get("delta", {})
                finish_reason = choices[0].get("finish_reason")

                # ── Text content — relay immediately ────────────
                # Reasoning models emit reasoning_content (chain-of-thought)
                # alongside or instead of content. The two are kept strictly
                # apart: only `content` is the assistant's actual reply, and
                # only `content` is eligible to enter the conversation history.
                raw_content = delta.get("content") or ""
                raw_reasoning = delta.get("reasoning_content") or ""

                # Chain-of-thought — relayed only when the caller opted in.
                if raw_reasoning:
                    reasoning_parts.append(raw_reasoning)
                    if relay_reasoning:
                        if not had_role:
                            had_role = True
                            if relayed_text:
                                # Previous iteration already emitted content;
                                # insert a separator so answers don't run together.
                                yield _chunk_sse({"content": "\n\n"})
                            else:
                                yield _chunk_sse({"role": "assistant"})
                        yield _chunk_sse({"content": raw_reasoning})

                # Actual answer text — always relayed.
                if raw_content:
                    if not had_role:
                        had_role = True
                        if relayed_text:
                            yield _chunk_sse({"content": "\n\n"})
                        else:
                            yield _chunk_sse({"role": "assistant"})
                    content_parts.append(raw_content)
                    yield _chunk_sse({"content": raw_content})

                # ── Tool-call fragments — accumulate ─────────────
                delta_tcs = delta.get("tool_calls")
                if delta_tcs:
                    for tc in delta_tcs:
                        idx = tc.get("index", 0)
                        entry = tool_fragments.setdefault(idx, {})
                        # ID — first fragment carries it
                        if "id" in tc and "id" not in entry:
                            entry["id"] = tc["id"]
                        # Function name — first fragment carries it
                        func = tc.get("function", {})
                        if "name" in func and "function" not in entry:
                            entry["function"] = {"name": func["name"], "arguments": ""}
                        # Arguments — concatenate across fragments
                        if "arguments" in func:
                            if "function" not in entry:
                                entry["function"] = {"name": "", "arguments": ""}
                            entry["function"]["arguments"] += func["arguments"]

                # ── Token usage (final chunk carries usage + timings) ─
                if "usage" in chunk:
                    stats.record_token_usage(
                        chunk["usage"],
                        chunk.get("timings"),
                    )

                if finish_reason is not None:
                    last_finish_reason = finish_reason
                    break

            stats.record_llm_call((time.monotonic() - t0) * 1000)

            # LM Studio doesn't stream `usage` — estimate tokens from
            # accumulated text (chars ÷ 4 ≈ tokens is standard heuristic).
            # Reasoning counts toward generation cost even when it is not
            # relayed, so both go into the token estimate.
            answer_text = "".join(content_parts)
            thinking_text = "".join(reasoning_parts)
            generated_text = answer_text + thinking_text
            if generated_text:
                stats.completion_tokens += max(1, len(generated_text) // 4)
            # Estimate prompt tokens from the conversation sent to the LLM.
            # Rough but gives useful prompt_tokens_per_second even when the
            # backend doesn't return usage in streaming mode.
            if stats.prompt_tokens == 0:
                prompt_chars = sum(
                    len(str(m.get("content", ""))) for m in conversation
                )
                stats.prompt_tokens = max(1, prompt_chars // 4)

            logger.info(
                "Streaming iteration %d done: %d chunks, %d answer chars, "
                "%d reasoning chars (relayed=%s), %d tool-call fragments",
                iteration, chunk_count,
                len(answer_text),
                len(thinking_text),
                relay_reasoning,
                len(tool_fragments),
            )

            # ── Assemble accumulated tool calls ─────────────────
            # Reconstruct full tool-call objects from fragments
            assembled_tool_calls: list[dict[str, Any]] = []
            for idx in sorted(tool_fragments):
                frag = tool_fragments[idx]
                func = frag.get("function", {})
                tool_name = func.get("name", "").strip() if isinstance(func, dict) else ""
                if not tool_name:
                    continue
                assembled_tool_calls.append({
                    "id": frag.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": func.get("arguments", "{}"),
                    },
                })

            # ── No tool calls — answer is complete ──────────────
            if not assembled_tool_calls:
                if content_parts or had_role:
                    # Something was already relayed (answer text, or
                    # chain-of-thought the caller asked to see).
                    yield _chunk_sse({}, "stop")
                elif last_finish_reason == "length":
                    # Generation was truncated before ANY answer text or
                    # tool call was produced — the prompt (or the model's
                    # chain-of-thought) exhausted the context window.
                    # Reasoning models burn their whole budget on
                    # reasoning_content and emit empty content when they
                    # hit the wall, which looks identical to a dead model.
                    #
                    # Surface this as a context-overflow error instead of
                    # relaying the half-finished thinking or a fake "empty
                    # response" message: clients that auto-compact on
                    # "prompt is too long" (Claude Code) then shrink the
                    # conversation and retry successfully. Nothing has been
                    # relayed in this iteration, so the error becomes the
                    # first SSE event and the server turns it into an HTTP
                    # 400 before headers are sent.
                    logger.warning(
                        "Generation truncated (finish_reason=length) with "
                        "no content and no tool calls (%d reasoning chars, "
                        "%d chunks) — reporting context overflow",
                        len(thinking_text), chunk_count,
                    )
                    stats.total_iterations = iteration
                    if stats_out is not None:
                        stats_out.append(stats.to_dict())
                    yield _error_sse(
                        "prompt is too long: the model exhausted its "
                        "context window mid-generation before producing "
                        "an answer (finish_reason=length) — compact or "
                        "shorten the conversation and retry",
                        "context_overflow",
                    )
                    yield _stats_sse()
                    yield "data: [DONE]\n\n"
                    return
                elif thinking_text:
                    # The model produced only chain-of-thought and we
                    # suppressed it, so nothing has been sent. A blank reply
                    # is worse than the thinking itself — relay it, but say
                    # so in the logs since it means the model never closed
                    # its reasoning block into a real answer.
                    logger.warning(
                        "Model produced %d reasoning chars but no content and "
                        "no tool calls — relaying reasoning as the answer",
                        len(thinking_text),
                    )
                    yield _chunk_sse({"role": "assistant"})
                    yield _chunk_sse({"content": thinking_text}, "stop")
                else:
                    if total_searches > 0:
                        # Model didn't synthesise an answer after searching.
                        # Stream the raw search results as a graceful fallback
                        # so the caller sees SOMETHING instead of an empty response.
                        search_texts: list[str] = []
                        for msg in conversation:
                            if msg.get("role") == "tool" and msg.get("name") == WEB_SEARCH:
                                search_texts.append(msg.get("content", ""))
                        if search_texts:
                            fallback = (
                                "I found the following information:\n\n"
                                + "\n---\n".join(search_texts)
                            )
                            logger.warning(
                                "Model returned empty text after %d search(es) — "
                                "falling back to raw search results", total_searches
                            )
                            yield _chunk_sse({"role": "assistant"})
                            yield _chunk_sse({"content": fallback}, "stop")
                        else:
                            yield _chunk_sse({"role": "assistant"})
                            yield _chunk_sse({}, "stop")
                    else:
                        # Completely empty response — no text, no tool calls,
                        # no prior searches.  Yield a diagnostic message so
                        # the caller sees something actionable instead of
                        # "request completed without producing content".
                        logger.warning(
                            "Model returned completely empty response "
                            "(%d chunks, 0 text, 0 tool calls, 0 searches)",
                            chunk_count,
                        )
                        yield _chunk_sse({"role": "assistant"})
                        yield _chunk_sse(
                            {"content": (
                                "The model returned an empty response. "
                                "This may indicate the prompt was too long, "
                                "the model is not loaded, or the model does "
                                "not support the requested task."
                            )},
                            "stop",
                        )
                stats.total_iterations = iteration
                # Record stats BEFORE yielding — consumers (including our
                # own adapters) stop consuming at [DONE], so lines after
                # that yield never run and stats would silently never be
                # ingested.
                if stats_out is not None:
                    stats_out.append(stats.to_dict())
                yield _stats_sse()
                yield "data: [DONE]\n\n"
                return

            if had_role:
                relayed_text = True

            # ── Build assistant message for conversation ────────
            conversation.append({
                "role": "assistant",
                "content": "".join(content_parts) if content_parts else None,
                "tool_calls": assembled_tool_calls,
            })

            # Three-way classification:
            #   search tools  → execute server-side, feed result, continue
            #   client tools  → passthrough to caller via SSE delta chunks
            #   hallucinations → feed error back to LLM, let it recover
            search_tool_calls: list[dict[str, Any]] = []
            passthrough_tool_calls: list[dict[str, Any]] = []
            hallucinated_tool_calls: list[dict[str, Any]] = []
            for tc in assembled_tool_calls:
                tool_name = tc.get("function", {}).get("name", "")
                if tool_name in TOOL_EXECUTORS:
                    search_tool_calls.append(tc)
                elif tool_name in client_tool_names:
                    passthrough_tool_calls.append(tc)
                    stats.passthrough_tool_count += 1
                    logger.info(
                        "Client tool %r — will passthrough to caller", tool_name
                    )
                elif tool_name:
                    hallucinated_tool_calls.append(tc)
                    stats.hallucinated_tool_count += 1
                    logger.info(
                        "Hallucinated tool %r — will block", tool_name
                    )
                else:
                    logger.warning(
                        "Dropping malformed tool call with empty name: %s",
                        json.dumps(tc)[:200],
                    )

            # Execute search tools and feed results back to conversation
            for tc in search_tool_calls:
                tool_name = tc.get("function", {}).get("name", "")

                try:
                    raw_args = tc.get("function", {}).get("arguments", "{}")
                    if isinstance(raw_args, str):
                        arguments = json.loads(raw_args)
                    else:
                        arguments = raw_args
                except json.JSONDecodeError:
                    arguments = {}

                logger.info("Tool call: %s(%s)", tool_name, arguments)

                # ── Duplicate detection ──────────────────────────
                dedup_key: str | None = None
                if tool_name == WEB_SEARCH:
                    dedup_key = arguments.get("query", "").strip().lower()
                elif tool_name == FETCH_PAGE:
                    dedup_key = arguments.get("url", "").strip().lower()

                if dedup_key and (
                    (tool_name == WEB_SEARCH and dedup_key in seen_queries)
                    or (tool_name == FETCH_PAGE and dedup_key in seen_urls)
                ):
                    result_text = (
                        f"You already searched for this — the results are "
                        f"earlier in this conversation. Use them to answer "
                        f"the user. If you need DIFFERENT information, try "
                        f"a more specific or alternative query."
                    )
                    logger.info(
                        "Duplicate %s blocked: %r — fed hint to LLM",
                        tool_name, dedup_key,
                    )
                else:
                    if dedup_key and tool_name == WEB_SEARCH:
                        seen_queries.add(dedup_key)
                    elif dedup_key and tool_name == FETCH_PAGE:
                        seen_urls.add(dedup_key)

                    if tool_name == WEB_SEARCH:
                        total_searches += 1
                        stats.web_search_count += 1
                    elif tool_name == FETCH_PAGE:
                        stats.fetch_page_count += 1

                    result_text = await dispatch_tool(
                        tool_name=tool_name,
                        arguments=arguments,
                        search_provider=search_provider,
                    )
                total_tool_calls += 1

                conversation.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", f"call_{total_tool_calls}"),
                    "name": tool_name,
                    "content": result_text,
                })

            # Hallucinated tools — names the model invented. Feed error
            # back to the LLM so it can recover and try a different approach.
            for tc in hallucinated_tool_calls:
                tool_name = tc.get("function", {}).get("name", "unknown")
                conversation.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", f"call_{total_tool_calls}"),
                    "name": tool_name,
                    "content": (
                        f"Error: the '{tool_name}' tool is not available. "
                        "You only have web_search and fetch_page. "
                        "Please use web_search to find the information you need."
                    ),
                })
                total_tool_calls += 1
                logger.info(
                    "Blocked hallucinated tool %r — fed error back to LLM", tool_name
                )

            if hallucinated_tool_calls:
                # Continue the loop — LLM gets error feedback and retries
                continue

            # Client tools — the LLM called a tool the client provided.
            # Emit as SSE delta chunks so the adapters capture them,
            # then stop the stream. The client will execute the tools
            # and make another request.
            if passthrough_tool_calls:
                logger.info(
                    "Passthrough %d client tool(s) to caller: %s",
                    len(passthrough_tool_calls),
                    [tc.get("function", {}).get("name") for tc in passthrough_tool_calls],
                )
                # Emit role preamble if not already sent in this iteration
                if not had_role:
                    yield _chunk_sse({"role": "assistant"})
                # Emit each client tool call as a delta chunk. Only the
                # last chunk carries finish_reason per OpenAI SSE spec.
                for i, tc in enumerate(passthrough_tool_calls):
                    is_last = (i == len(passthrough_tool_calls) - 1)
                    yield _chunk_sse(
                        {"tool_calls": [tc]},
                        "tool_use" if is_last else None,
                    )
                stats.total_iterations = iteration
                # Record stats BEFORE yielding — consumers (including our
                # own adapters) stop consuming at [DONE], so lines after
                # that yield never run and stats would silently never be
                # ingested.
                if stats_out is not None:
                    stats_out.append(stats.to_dict())
                yield _stats_sse()
                yield "data: [DONE]\n\n"
                return

            # Loop continues — LLM sees search results and responds

        # Max iterations exceeded — stream accumulated search results
        # as a graceful fallback instead of an error.
        search_result_texts: list[str] = []
        for msg in conversation:
            if msg.get("role") == "tool" and msg.get("name") == WEB_SEARCH:
                search_result_texts.append(msg.get("content", ""))

        if search_result_texts:
            fallback = (
                "I searched multiple times but was unable to synthesize a final answer. "
                "Here are the raw search results:\n\n" +
                "\n---\n".join(search_result_texts)
            )
        else:
            fallback = (
                f"Tool loop exceeded maximum iterations ({max_iter}). "
                f"No search results were collected. "
                f"Last response had {total_tool_calls} tool calls."
            )

        stats.total_iterations = max_iter
        if stats_out is not None:
            stats_out.append(stats.to_dict())
        yield _chunk_sse({"role": "assistant"})
        yield _chunk_sse({"content": fallback}, "tool_loop_max")
        yield _stats_sse()
        yield "data: [DONE]\n\n"

    except LMStudioError as exc:
        logger.error("LM Studio error during streaming: %s", exc)
        err_type = "context_overflow" if is_context_overflow(exc) else "lm_studio_error"
        if stats_out is not None:
            stats_out.append(stats.to_dict())
        yield _error_sse(str(exc), err_type)
        yield _stats_sse()
        yield "data: [DONE]\n\n"
