"""OpenAI Responses API ↔ Chat Completions adapter.

Translates between the two formats so the Codex desktop app
(which requires ``wire_api = "responses"``) can use the middleware.

Endpoint:  POST /v1/responses  →  internal Chat Completions tool loop
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator

from .tool_registry import FETCH_PAGE_TOOL, WEB_SEARCH_TOOL

logger = logging.getLogger(__name__)

# ── Tiny helpers ──────────────────────────────────────────────────

def _maybe_loads(s: str | dict[str, Any] | Any) -> Any:
    """Parse a JSON string, pass through a dict, or convert to string."""
    if isinstance(s, str):
        try:
            return json.loads(s)
        except (json.JSONDecodeError, TypeError):
            return s
    if isinstance(s, dict):
        return s
    return str(s)


def _extract_text(content: str | list[dict[str, Any]] | Any) -> str:
    """Pull plain text out of Responses-API content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            t = block.get("type", "")
            if t in ("input_text", "output_text", "text"):
                parts.append(str(block.get("text", "")))
        return "\n".join(parts)
    return str(content or "")


def _wrap_tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    """Ensure a tool is in Chat-Completions (nested-function) format.

    Responses API allows a flat ``{type, name, description, parameters}``
    shape; Chat Completions requires ``{type:"function", function:{...}}``.
    If the tool already has a ``function`` key we leave it alone.

    Also fixes common schema issues that cause LM Studio to reject tools:
    missing ``type: "object"`` in parameters, missing ``additionalProperties``.
    """
    def _sanitize_params(p: Any) -> dict[str, Any]:
        """Normalize parameters to a valid JSON Schema object."""
        if not isinstance(p, dict):
            return {"type": "object", "properties": {}, "additionalProperties": False}
        p.setdefault("type", "object")
        p.setdefault("properties", {})
        p.setdefault("additionalProperties", False)
        return p

    if "function" in tool:
        # Already wrapped (or compat format) — patch inner parameters
        inner = tool["function"]
        inner["parameters"] = _sanitize_params(inner.get("parameters"))
        return tool

    # Flattened Responses shape → wrap in Chat Completions format
    return {
        "type": "function",
        "function": {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": _sanitize_params(tool.get("parameters")),
        },
    }


# ── Request: Responses API → OpenAI Chat ──────────────────────────

def responses_request_to_openai(body: dict[str, Any]) -> dict[str, Any]:
    """Convert a Responses API request to Chat Completions format.

    Returns a dict with keys ``messages``, ``tools``, ``model`` suitable
    for passing directly to ``run_tool_loop`` / ``run_tool_loop_streaming``.
    """
    openai_messages: list[dict[str, Any]] = []

    # 1. ``instructions`` → system message
    instructions: str | None = body.get("instructions")
    if instructions:
        openai_messages.append({"role": "system", "content": instructions})

    # 2. Convert ``input`` items → messages
    raw_input = body.get("input", [])

    if isinstance(raw_input, str):
        # Bare string → single user message
        openai_messages.append({"role": "user", "content": raw_input})
        raw_input = []  # fall through to empty list below

    for item in raw_input:
        item_type = item.get("type", "")

        if item_type == "message":
            role = item.get("role", "user")
            content = _extract_text(item.get("content", ""))
            openai_messages.append({"role": role, "content": content})

        elif item_type == "function_call":
            # Assistant emitted a tool call in prior turn
            tc_id = item.get("id") or item.get("call_id") or f"fc_{uuid.uuid4().hex[:8]}"
            raw_args = _maybe_loads(item.get("arguments", "{}"))
            args_str = json.dumps(raw_args, ensure_ascii=False) if not isinstance(raw_args, str) else raw_args
            openai_messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": args_str,
                    },
                }],
            })

        elif item_type == "function_call_output":
            # Tool result from prior turn
            call_id = item.get("call_id", "")
            output = item.get("output", "")
            if isinstance(output, dict):
                output = json.dumps(output, ensure_ascii=False)
            elif not isinstance(output, str):
                output = str(output)
            openai_messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": output,
            })

        # Unknown item types are silently skipped

    # 3. Convert tools
    # The middleware's tool_loop auto-injects web_search + fetch_page
    # so we pass along any tools the client specified *in addition*.
    # The tool_loop will merge them.
    raw_tools: list[dict[str, Any]] = body.get("tools", []) or []
    openai_tools = [_wrap_tool_schema(t) for t in raw_tools] if raw_tools else None

    return {
        "messages": openai_messages,
        "tools": openai_tools,
        "model": body.get("model", "local-model"),
    }


# ── Response: Tool-loop result → Responses API ────────────────────

def openai_result_to_responses(
    result: dict[str, Any],
    model: str,
    *,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Convert a tool_loop result dict into a Responses API response shape.

    ``result`` is the dict returned by ``run_tool_loop``::

        {
            "content": str | None,
            "tool_calls": [...],        # passthrough (not recognized)
            "tool_calls_count": int,
            "iterations": int,
            "searches": int,
            "finish_reason": str,
        }
    """
    resp_id = request_id or f"resp_{uuid.uuid4().hex[:12]}"
    output_items: list[dict[str, Any]] = []

    content_text = result.get("content") or ""
    passthrough = result.get("tool_calls") or []

    # Text output
    if content_text:
        output_items.append({
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": content_text}
            ],
        })

    # Passthrough tool calls (unrecognized by middleware → forward to Codex)
    for tc in passthrough:
        func = tc.get("function", {})
        raw_args = func.get("arguments", "{}")
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = raw_args
        else:
            args = raw_args
        args_str = json.dumps(args, ensure_ascii=False) if not isinstance(args, str) else args

        tc_id = tc.get("id") or f"fc_{uuid.uuid4().hex[:8]}"
        output_items.append({
            "type": "function_call",
            "id": tc_id,
            "call_id": tc_id,
            "name": func.get("name", ""),
            "arguments": args_str,
        })

    # Status
    finish = result.get("finish_reason", "stop")
    status = "completed" if finish in ("stop", "tool_use") else "incomplete"

    return {
        "id": resp_id,
        "object": "response",
        "status": status,
        "created_at": int(time.time()),
        "model": model,
        "output": output_items,
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
    }


# ── Streaming: Tool-loop SSE → Responses API SSE ──────────────────

def _responses_sse(event: str, data: dict[str, Any]) -> str:
    """Format a Responses API SSE event."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def responses_stream_from_tool_loop(
    stream_generator: AsyncGenerator[str, None],
    *,
    model: str,
    resp_id: str,
) -> AsyncGenerator[str, None]:
    """Convert OpenAI SSE chunks from the tool loop into Responses API SSE.

    Emits the Responses streaming lifecycle:
      response.created → output_text.delta* → output_text.done → response.completed

    Passthrough tool calls (unrecognized by the middleware) are emitted
    as ``function_call`` output items between the text block and completed.
    """
    content_parts: list[str] = []
    pending_tool_calls: list[dict[str, Any]] = []
    msg_id = f"msg_{uuid.uuid4().hex[:12]}"
    output_index: int = 0
    content_index: int = 0
    created_sent = False
    text_done_sent = False

    # ── Peek at stream for errors before emitting response.created ──
    async for line in stream_generator:
        line = line.strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data: "):
            continue

        data_str = line[6:]
        if data_str == "[DONE]":
            break

        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        # Error in stream
        if "error" in chunk:
            err = chunk["error"]
            msg = err.get("message", "Unknown error")
            yield _responses_sse("error", {
                "type": "error",
                "error": {"code": err.get("type", "server_error"), "message": msg},
            })
            return

        choices = chunk.get("choices", [])
        if not choices:
            continue

        delta = choices[0].get("delta", {})
        finish_reason = choices[0].get("finish_reason")

        # ── Text content ─────────────────────────────────────────
        if "role" in delta:
            # Delta with role="assistant" — just a preamble, no content
            continue

        text = delta.get("content") or ""
        if text:
            if not created_sent:
                created_sent = True
                yield _responses_sse("response.created", {
                    "type": "response.created",
                    "response": {
                        "id": resp_id,
                        "object": "response",
                        "status": "in_progress",
                        "model": model,
                        "output": [],
                    },
                })
                yield _responses_sse("response.in_progress", {
                    "type": "response.in_progress",
                    "response": {
                        "id": resp_id,
                        "object": "response",
                        "status": "in_progress",
                        "model": model,
                        "output": [],
                    },
                })
                # Codex requires these lifecycle events before any deltas:
                # output_item.added → content_part.added → output_text.delta
                yield _responses_sse("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": {
                        "type": "message",
                        "id": msg_id,
                        "role": "assistant",
                        "status": "in_progress",
                        "content": [],
                    },
                })
                yield _responses_sse("response.content_part.added", {
                    "type": "response.content_part.added",
                    "item_id": msg_id,
                    "output_index": output_index,
                    "content_index": content_index,
                    "part": {"type": "output_text", "text": ""},
                })
            content_parts.append(text)
            yield _responses_sse("response.output_text.delta", {
                "type": "response.output_text.delta",
                "item_id": msg_id,
                "output_index": output_index,
                "content_index": content_index,
                "delta": text,
            })

        # ── Passthrough tool calls ───────────────────────────────
        delta_tcs = delta.get("tool_calls")
        if delta_tcs:
            for tc in delta_tcs:
                func = tc.get("function", {})
                raw_args = func.get("arguments", "{}")
                if isinstance(raw_args, str):
                    try:
                        parsed = json.loads(raw_args)
                    except json.JSONDecodeError:
                        parsed = raw_args
                else:
                    parsed = raw_args
                args_str = json.dumps(parsed, ensure_ascii=False) if not isinstance(parsed, str) else parsed

                tc_id = tc.get("id") or f"fc_{uuid.uuid4().hex[:8]}"
                pending_tool_calls.append({
                    "type": "function_call",
                    "id": tc_id,
                    "call_id": tc_id,
                    "name": func.get("name", ""),
                    "arguments": args_str,
                })

        # finish_reason is relayed per-chunk but we handle it after the loop

    # ── Close text block ─────────────────────────────────────────
    full_text = "".join(content_parts)

    if created_sent and full_text:
        text_done_sent = True
        yield _responses_sse("response.output_text.done", {
            "type": "response.output_text.done",
            "item_id": msg_id,
            "output_index": output_index,
            "content_index": content_index,
            "text": full_text,
        })
        yield _responses_sse("response.content_part.done", {
            "type": "response.content_part.done",
            "item_id": msg_id,
            "output_index": output_index,
            "content_index": content_index,
            "part": {"type": "output_text", "text": full_text},
        })
        yield _responses_sse("response.output_item.done", {
            "type": "response.output_item.done",
            "output_index": output_index,
            "item": {
                "type": "message",
                "id": msg_id,
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": full_text}],
            },
        })

    # ── Build final output array ─────────────────────────────────
    output_items: list[dict[str, Any]] = []
    if full_text:
        output_items.append({
            "type": "message",
            "id": msg_id,
            "role": "assistant",
            "content": [{"type": "output_text", "text": full_text}],
        })
    for tc in pending_tool_calls:
        output_items.append(tc)

    # ── Emit response.completed ──────────────────────────────────
    if not created_sent:
        # No content produced at all — still emit lifecycle
        created_sent = True
        yield _responses_sse("response.created", {
            "type": "response.created",
            "response": {
                "id": resp_id,
                "object": "response",
                "status": "in_progress",
                "model": model,
                "output": [],
            },
        })

    yield _responses_sse("response.completed", {
        "type": "response.completed",
        "response": {
            "id": resp_id,
            "object": "response",
            "status": "completed",
            "created_at": int(time.time()),
            "model": model,
            "output": output_items,
            "usage": {
                "input_tokens": 0,
                "output_tokens": len(full_text.split()) if full_text else 0,
                "total_tokens": 0,
            },
        },
    })
