"""The core tool-call intercept loop.

This is the engine: it sends the user's messages to LM Studio,
intercepts any web_search tool calls, executes them, feeds results
back to the LLM, and loops until the LLM produces a plain text answer.
"""

import json
import logging
from typing import Any, Optional

import httpx

from .config import settings
from .search.base import SearchProvider
from .tool_registry import dispatch_tool, inject_web_search_tool, WEB_SEARCH

logger = logging.getLogger(__name__)


class ToolLoopExhaustedError(Exception):
    """Raised when the tool-call loop exceeds max iterations."""


class LMStudioError(Exception):
    """Raised when LM Studio returns an error or is unreachable."""


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

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.ConnectError:
            raise LMStudioError(f"LM Studio not reachable at {lm_studio_url}")
        except httpx.HTTPStatusError as exc:
            raise LMStudioError(
                f"LM Studio returned {exc.response.status_code}: "
                f"{exc.response.text[:500]}"
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

    # Inject web_search tool
    all_tools = inject_web_search_tool(tools)

    conversation = list(messages)  # Copy — we'll mutate this
    total_searches = 0
    total_tool_calls = 0

    for iteration in range(1, max_iter + 1):
        logger.debug("Tool loop iteration %d/%d", iteration, max_iter)

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

        # Execute each tool call
        for tc in tool_calls:
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

        # Loop continues — LLM sees the search results and responds

    # Max iterations exceeded
    raise ToolLoopExhaustedError(
        f"Tool loop exceeded maximum iterations ({max_iter}). "
        f"Last response had {total_tool_calls} tool calls across {max_iter} iterations."
    )
