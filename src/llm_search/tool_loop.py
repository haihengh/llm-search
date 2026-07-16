"""The core tool-call intercept loop.

This is the engine: it sends the user's messages to LM Studio,
intercepts any web_search tool calls, executes them, feeds results
back to the LLM, and loops until the LLM produces a plain text answer.
"""

import json
import logging
import uuid
from typing import Any, AsyncGenerator, Optional

import httpx

from .config import settings
from .search.base import SearchProvider
from .tool_registry import (
    FETCH_PAGE,
    FETCH_PAGE_TOOL,
    TOOL_EXECUTORS,
    WEB_SEARCH,
    WEB_SEARCH_TOOL,
    dispatch_tool,
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
    "too long",
)


def is_context_overflow(exc: LMStudioError) -> bool:
    """True when LM Studio rejected the request because the prompt
    exceeds the loaded model's context window.

    llama.cpp phrases this as e.g. "The number of tokens to keep from
    the initial prompt is greater than the context length (n_keep: X
    >= n_ctx: Y)". MLX and other backends use similar wording.
    """
    if exc.status_code is not None and exc.status_code != 400:
        return False
    msg = str(exc).lower()
    return any(marker in msg for marker in _CONTEXT_OVERFLOW_MARKERS)


# ── LM Studio Chat Client ─────────────────────────────────────

async def call_lm_studio(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    model: str,
    lm_studio_url: str,
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

    async with httpx.AsyncClient(timeout=settings.lm_studio_timeout) as client:
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.ConnectError:
            raise LMStudioError(f"LM Studio not reachable at {lm_studio_url}")
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

    async with httpx.AsyncClient(timeout=settings.lm_studio_timeout) as client:
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
                            yield json.loads(data)
                        except json.JSONDecodeError:
                            logger.debug("Skipping unparseable SSE line: %s", line[:100])
        except httpx.ConnectError:
            raise LMStudioError(f"LM Studio not reachable at {lm_studio_url}")
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
) -> dict[str, Any]:
    """Execute the tool-call intercept loop.

    1. Send messages + tools to LM Studio
    2. If LLM calls web_search → execute, feed back, repeat
    3. Return the final assistant message as an OpenAI-format dict

    Args:
        messages: Chat messages (OpenAI format)
        search_provider: Where to execute searches
        tools: Optional client-provided tools (web_search is auto-injected)
        model: Model name to pass to LM Studio
        lm_studio_url: URL of the LM Studio API

    Returns:
        Dict with keys: content, tool_calls_count, iterations, searches

    Raises:
        ToolLoopExhaustedError: Max iterations exceeded
        LMStudioError: LM Studio is unreachable or errors
    """
    url = lm_studio_url or settings.lm_studio_url
    max_iter = settings.max_tool_loop_iterations

    # Only pass search tools to the LLM — client tools (bash, read, etc.)
    # are handled by the client itself (e.g. Claude Code's agentic loop).
    # Injecting all client tools confuses small local models.
    all_tools = [WEB_SEARCH_TOOL, FETCH_PAGE_TOOL]

    conversation = list(messages)  # Copy — we'll mutate this
    total_searches = 0
    total_tool_calls = 0

    for iteration in range(1, max_iter + 1):
        logger.debug("Tool loop iteration %d/%d", iteration, max_iter)

        # On later iterations, nudge the LLM to synthesize an answer
        # instead of searching again. Small local models sometimes get
        # stuck in a search → search → search loop without this reminder.
        if iteration >= max_iter - 1 and total_searches > 0:
            conversation.append({
                "role": "user",
                "content": (
                    "You now have search results. Please synthesize a "
                    "final answer based on what you found. Do NOT call "
                    "web_search again — answer the user's question now."
                ),
            })

        # Send to LM Studio
        response = await call_lm_studio(
            messages=conversation,
            tools=all_tools,
            model=model,
            lm_studio_url=url,
        )

        content, tool_calls = extract_assistant_message(response)

        # No tool calls → LLM is done, return the answer
        if not tool_calls:
            return {
                "content": content or "",
                "tool_calls_count": total_tool_calls,
                "iterations": iteration,
                "searches": total_searches,
            }

        # Build the assistant message with tool_calls to append to conversation
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
        }
        conversation.append(assistant_message)

        # Split tool calls: recognised (we handle) vs unrecognised (passthrough to client)
        our_tool_calls = []
        their_tool_calls = []
        for tc in tool_calls:
            tool_name = tc.get("function", {}).get("name", "")
            if tool_name in TOOL_EXECUTORS:
                our_tool_calls.append(tc)
            elif tool_name:
                their_tool_calls.append(tc)
                logger.info("Unrecognised tool %r — will passthrough to client", tool_name)
            else:
                logger.warning(
                    "Dropping malformed tool call with empty name: %s",
                    json.dumps(tc)[:200],
                )

        # Execute recognised tools and feed results back to conversation
        for tc in our_tool_calls:
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

            # Track searches
            if tool_name == WEB_SEARCH:
                total_searches += 1
            total_tool_calls += 1

            # Execute the tool
            result_text = await dispatch_tool(
                tool_name=tool_name,
                arguments=arguments,
                search_provider=search_provider,
            )

            # Append tool result to conversation
            tool_message = {
                "role": "tool",
                "tool_call_id": tc.get("id", f"call_{total_tool_calls}"),
                "name": tool_name,
                "content": result_text,
            }
            conversation.append(tool_message)

        # Unrecognised tools are almost certainly hallucinations from
        # Claude-distilled models that know about Bash/Read/Write from
        # training. Passing them to the client causes "invalid tool
        # parameters" errors. Instead, feed an error back to the LLM so
        # it can recover and try a different approach.
        for tc in their_tool_calls:
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

        if their_tool_calls:
            # Some unrecognised tools were blocked above. Continue the loop
            # so the LLM can adjust its approach.
            continue

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

    return {
        "content": fallback_content,
        "tool_calls_count": total_tool_calls,
        "iterations": max_iter,
        "searches": total_searches,
        "finish_reason": "tool_loop_max",
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
) -> AsyncGenerator[str, None]:
    """Execute the tool-call loop with single-pass streaming on every turn.

    The LLM is called with stream=True from the first call. Text deltas
    are relayed live to the caller. Tool-call fragments are accumulated
    across chunks — if the model finishes with recognized tool calls
    (web_search / fetch_page), they are executed, results are appended
    to the conversation, and a new streaming call begins. Hallucinated
    client tools (bash, read, write, etc.) are blocked server-side with
    an error fed back to the LLM so it can recover.

    Unlike the old "check + re-generate" approach this issues exactly
    one LM Studio call per iteration — no second generation that can
    diverge from the first one.

    Yields:
        SSE-formatted strings: ``data: {json}\\n\\n`` per chunk.
        Terminates with ``data: [DONE]\\n\\n``.
    """
    url = lm_studio_url or settings.lm_studio_url
    max_iter = settings.max_tool_loop_iterations

    # Only pass search tools to the LLM (see run_tool_loop for rationale)
    all_tools = [WEB_SEARCH_TOOL, FETCH_PAGE_TOOL]
    conversation = list(messages)
    total_searches = 0
    total_tool_calls = 0
    relayed_text = False  # True once any text content has been sent to caller

    def _sse(data: dict[str, Any]) -> str:
        """Format a dict as an SSE data event."""
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

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

            # On later iterations, nudge the LLM to answer
            if iteration >= max_iter - 1 and total_searches > 0:
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
            content_parts: list[str] = []
            tool_fragments: dict[int, dict[str, Any]] = {}
            chunk_count = 0

            async for chunk in call_lm_studio_streaming(
                messages=conversation,
                tools=all_tools,
                model=model,
                lm_studio_url=url,
            ):
                chunk_count += 1
                choices = chunk.get("choices", [])
                if not choices:
                    logger.debug("Streaming chunk %d: no choices, skipping", chunk_count)
                    continue
                delta = choices[0].get("delta", {})
                finish_reason = choices[0].get("finish_reason")

                # ── Text content — relay immediately ────────────
                text = delta.get("content") or ""
                if text:
                    if not had_role:
                        had_role = True
                        if relayed_text:
                            # Previous iteration already emitted content;
                            # insert a separator so answers don't run together.
                            yield _chunk_sse({"content": "\n\n"})
                        else:
                            yield _chunk_sse({"role": "assistant"})
                    content_parts.append(text)
                    yield _chunk_sse({"content": text})

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

                if finish_reason is not None:
                    break

            logger.info(
                "Streaming iteration %d done: %d chunks, %d text chars, "
                "%d tool-call fragments, finish_reason not None",
                iteration, chunk_count,
                sum(len(p) for p in content_parts),
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
                if not content_parts:
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
                            yield _chunk_sse({"content": fallback}, "stop")
                        else:
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
                        yield _chunk_sse(
                            {"content": (
                                "The model returned an empty response. "
                                "This may indicate the prompt was too long, "
                                "the model is not loaded, or the model does "
                                "not support the requested task."
                            )},
                            "stop",
                        )
                else:
                    yield _chunk_sse({}, "stop")
                yield "data: [DONE]\n\n"
                return

            if content_parts:
                relayed_text = True

            # ── Build assistant message for conversation ────────
            conversation.append({
                "role": "assistant",
                "content": "".join(content_parts) if content_parts else None,
                "tool_calls": assembled_tool_calls,
            })

            # Split into recognised (we handle) vs unrecognised (block)
            our_tool_calls: list[dict[str, Any]] = []
            their_tool_calls: list[dict[str, Any]] = []
            for tc in assembled_tool_calls:
                tool_name = tc.get("function", {}).get("name", "")
                if tool_name in TOOL_EXECUTORS:
                    our_tool_calls.append(tc)
                elif tool_name:
                    their_tool_calls.append(tc)
                    logger.info(
                        "Unrecognised tool %r — blocked (fed back to LLM)", tool_name
                    )
                else:
                    logger.warning(
                        "Dropping malformed tool call with empty name: %s",
                        json.dumps(tc)[:200],
                    )

            for tc in our_tool_calls:
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

                if tool_name == WEB_SEARCH:
                    total_searches += 1
                total_tool_calls += 1

                result_text = await dispatch_tool(
                    tool_name=tool_name,
                    arguments=arguments,
                    search_provider=search_provider,
                )

                conversation.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", f"call_{total_tool_calls}"),
                    "name": tool_name,
                    "content": result_text,
                })

            # Unrecognised tools = hallucination. Feed error to LLM.
            for tc in their_tool_calls:
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

            if their_tool_calls:
                # Continue the loop — LLM gets error feedback and retries
                continue

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

        yield _chunk_sse({"role": "assistant"})
        yield _chunk_sse({"content": fallback}, "tool_loop_max")
        yield "data: [DONE]\n\n"

    except LMStudioError as exc:
        logger.error("LM Studio error during streaming: %s", exc)
        err_type = "context_overflow" if is_context_overflow(exc) else "lm_studio_error"
        yield _error_sse(str(exc), err_type)
        yield "data: [DONE]\n\n"
