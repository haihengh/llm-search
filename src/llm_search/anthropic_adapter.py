"""Anthropic Messages API ↔ OpenAI Chat Completions adapter.

Translates between the two formats so Claude Code and other Anthropic
clients can use the middleware transparently.

Anthropic endpoint:  POST /v1/messages
OpenAI endpoint:     POST /v1/chat/completions (internal, to LM Studio)
"""

import json
import logging
import time
import uuid
from typing import Any, Optional

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import settings
from .tool_loop import (
    LMStudioError,
    ToolLoopExhaustedError,
    run_tool_loop,
    run_tool_loop_streaming,
)

logger = logging.getLogger(__name__)


# ── Anthropic → OpenAI Translation ─────────────────────────────

def _anthropic_content_to_openai(
    content: str | list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Convert Anthropic content to OpenAI text + tool_calls.

    Anthropic content can be:
      - A plain string (user message text)
      - A list of content blocks: [{"type": "text", "text": "..."}, ...]
      - Tool result blocks: [{"type": "tool_result", "tool_use_id": "...", "content": "..."}]

    Returns (text_content, tool_calls_list).
    """
    if isinstance(content, str):
        return content, []

    text_parts = []
    tool_calls = []

    for block in content:
        block_type = block.get("type", "")
        if block_type == "text":
            text_parts.append(block.get("text", ""))
        elif block_type == "tool_use":
            tool_calls.append({
                "id": block.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})) if isinstance(block.get("input"), dict) else str(block.get("input", "{}")),
                },
            })
        elif block_type == "tool_result":
            # Tool results are handled separately as role="tool" messages
            pass

    return "\n".join(text_parts) if text_parts else "", tool_calls


def _anthropic_tools_to_openai(anthropic_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic tool definitions to OpenAI format.

    Anthropic: {"name": "...", "description": "...", "input_schema": {...}}
    OpenAI:    {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
    """
    openai_tools = []
    for tool in anthropic_tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return openai_tools


def anthropic_request_to_openai(body: dict[str, Any]) -> dict[str, Any]:
    """Convert an Anthropic Messages request to OpenAI Chat Completions format.

    Returns a dict suitable for the internal tool_loop processing.
    """
    openai_messages = []

    # Anthropic "system" → prepend as system message
    system_text = body.get("system")
    if system_text:
        if isinstance(system_text, list):
            # System can be a list of text blocks in newer Anthropic API
            system_text = "\n".join(
                b.get("text", "") for b in system_text if b.get("type") == "text"
            )
        openai_messages.append({"role": "system", "content": str(system_text)})

    # Convert messages
    for msg in body.get("messages", []):
        role = msg.get("role", "user")

        if role == "user":
            content = msg.get("content", "")
            # Check if this is a tool_result message
            if isinstance(content, list):
                has_tool_results = any(
                    b.get("type") == "tool_result" for b in content
                )
                if has_tool_results:
                    # Split into individual tool result messages
                    for block in content:
                        if block.get("type") == "tool_result":
                            openai_messages.append({
                                "role": "tool",
                                "tool_call_id": block.get("tool_use_id", ""),
                                "content": _extract_tool_result_content(block),
                            })
                        elif block.get("type") == "text":
                            openai_messages.append({
                                "role": "user",
                                "content": block.get("text", ""),
                            })
                    continue

            text, tool_calls = _anthropic_content_to_openai(content)
            openai_messages.append({"role": "user", "content": text})

        elif role == "assistant":
            content = msg.get("content", "")
            text, tool_calls = _anthropic_content_to_openai(content)

            msg_dict: dict[str, Any] = {"role": "assistant"}
            if text:
                msg_dict["content"] = text
            if tool_calls:
                msg_dict["tool_calls"] = tool_calls
            openai_messages.append(msg_dict)

    # Convert tools
    anthropic_tools = body.get("tools", [])
    openai_tools = _anthropic_tools_to_openai(anthropic_tools) if anthropic_tools else None

    return {
        "messages": openai_messages,
        "tools": openai_tools,
        "model": body.get("model", "local-model"),
        "max_tokens": body.get("max_tokens", 4096),
    }


def _extract_tool_result_content(block: dict[str, Any]) -> str:
    """Extract text from an Anthropic tool_result block."""
    result_content = block.get("content", "")
    if isinstance(result_content, str):
        return result_content
    if isinstance(result_content, list):
        return "\n".join(
            b.get("text", "") for b in result_content if b.get("type") == "text"
        )
    return str(result_content)


# ── Tool-call validation ──────────────────────────────────────

def _validate_openai_tool_call(tc: dict[str, Any]) -> dict[str, Any] | None:
    """Validate and sanitize a single OpenAI-format tool call.

    Rejects tool calls with missing or empty function names, missing IDs,
    or unparseable arguments. Returns a cleaned dict or None.
    """
    func = tc.get("function")
    if not isinstance(func, dict):
        return None

    name = (func.get("name") or "").strip()
    if not name:
        return None

    # Parse + re-serialize arguments to guarantee valid JSON
    raw_args = func.get("arguments", "{}")
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError:
            return None  # unparseable JSON → drop
    elif isinstance(raw_args, dict):
        parsed = raw_args
    else:
        return None

    # Ensure ID is present; generate one if missing
    tool_id = tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"

    return {
        "id": tool_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(parsed, ensure_ascii=False),
        },
    }


# ── OpenAI → Anthropic Translation ─────────────────────────────

def _openai_tool_calls_to_anthropic_content(
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert OpenAI-format tool_calls to Anthropic tool_use content blocks.

    OpenAI:  {"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": "{...}"}}
    Anthropic: {"type": "tool_use", "id": "call_1", "name": "bash", "input": {...}}

    Tool calls should be pre-validated via _validate_openai_tool_call for the
    streaming path; for the non-streaming path, validation happens inline.
    """
    blocks: list[dict[str, Any]] = []
    for tc in tool_calls:
        func = tc.get("function", {})
        raw_args = func.get("arguments", "{}")
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
            except json.JSONDecodeError:
                parsed = raw_args
        else:
            parsed = raw_args

        blocks.append({
            "type": "tool_use",
            "id": tc.get("id", ""),
            "name": func.get("name", ""),
            "input": parsed,
        })
    return blocks


def openai_response_to_anthropic(
    openai_result: dict[str, Any],
    model: str,
    request_id: Optional[str] = None,
) -> dict[str, Any]:
    """Convert an OpenAI tool_loop result to Anthropic Messages format.

    OpenAI result: {"content": "...", "tool_calls_count": N, "iterations": N, "searches": N}
    When tool_calls are present (passthrough), returns stop_reason: "tool_use".
    """
    msg_id = request_id or f"msg_{uuid.uuid4().hex[:12]}"
    content_text = openai_result.get("content", "")
    passthrough_tool_calls = openai_result.get("tool_calls", [])

    # Validate and filter passthrough tool calls — malformed tool calls
    # from hallucinating models cause "tool call could not be parsed"
    # errors in Claude Code.
    valid_tool_calls = list(
        filter(None, (_validate_openai_tool_call(tc) for tc in passthrough_tool_calls))
    )

    # Build content blocks
    content_blocks: list[dict[str, Any]] = []
    if content_text:
        content_blocks.append({"type": "text", "text": content_text})

    if valid_tool_calls:
        content_blocks.extend(
            _openai_tool_calls_to_anthropic_content(valid_tool_calls)
        )

    stop_reason = "tool_use" if valid_tool_calls else "end_turn"

    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
        },
    }


# ── Streaming (OpenAI SSE → Anthropic SSE) ─────────────────────

async def anthropic_stream_from_openai(
    stream_generator,
    model: str,
    request_id: str,
) -> str:
    """Convert OpenAI SSE chunks to Anthropic SSE format.

    Handles both text content and tool_use content blocks. Tool calls that
    appear in delta chunks (from the middleware's passthrough) are captured
    and emitted as proper Anthropic tool_use content blocks.
    """
    import json as json_mod

    content_parts: list[str] = []
    finish_reason: str | None = None
    has_msg_start = False
    text_block_open = False
    text_block_index = -1   # assigned when text block starts
    next_block_index = 0
    pending_tool_calls: list[dict[str, Any]] = []

    async for line in stream_generator:
        line = line.strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json_mod.loads(data)
            except json_mod.JSONDecodeError:
                continue

            if "error" in chunk:
                err = chunk["error"]
                err_type: str = err.get("type", "api_error")
                err_msg: str = err.get("message", "Unknown error")

                # Map known LM Studio / llama.cpp error patterns to
                # Anthropic error types so Claude Code can react
                # appropriately (e.g. auto-compact on prompt-too-long).
                msg_lower = err_msg.lower()
                if err_type == "context_overflow" or any(
                    m in msg_lower
                    for m in ("n_ctx", "n_keep", "context length", "context_length", "context window")
                ):
                    err_type = "invalid_request_error"
                    err_msg = f"prompt is too long: {err_msg}"
                elif "not reachable" in msg_lower or "connect" in msg_lower:
                    err_type = "api_error"
                elif err_type not in ("invalid_request_error", "authentication_error",
                                        "permission_error", "not_found_error",
                                        "rate_limit_error", "api_error",
                                        "overloaded_error"):
                    err_type = "api_error"

                yield _sse_evt("error", {
                    "type": "error",
                    "error": {"type": err_type, "message": err_msg},
                })
                return

            choices = chunk.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta", {})
            fr = choices[0].get("finish_reason")

            # ── Text content ───────────────────────────────────
            content = delta.get("content") or ""
            if content:
                if not has_msg_start:
                    has_msg_start = True
                    yield _sse_evt("message_start", {
                        "type": "message_start",
                        "message": {
                            "id": request_id, "type": "message", "role": "assistant",
                            "model": model, "content": [],
                            "stop_reason": None, "stop_sequence": None,
                            "usage": {"input_tokens": 0, "output_tokens": 0},
                        },
                    })
                if not text_block_open:
                    text_block_open = True
                    text_block_index = next_block_index
                    next_block_index += 1
                    yield _sse_evt("content_block_start", {
                        "type": "content_block_start",
                        "index": text_block_index,
                        "content_block": {"type": "text", "text": ""},
                    })
                yield _sse_evt("content_block_delta", {
                    "type": "content_block_delta",
                    "index": text_block_index,
                    "delta": {"type": "text_delta", "text": content},
                })
                content_parts.append(content)

            # ── Tool calls in delta (passthrough) ──────────────
            delta_tool_calls = delta.get("tool_calls")
            if delta_tool_calls:
                for tc in delta_tool_calls:
                    validated = _validate_openai_tool_call(tc)
                    if validated:
                        pending_tool_calls.append(validated)

            if fr is not None:
                finish_reason = fr

    # ── Emit pending tool_use blocks ──────────────────────────────
    # These are unrecognised tool calls the middleware passed through
    # (e.g. a Claude-distilled model hallucinating read/bash/write).
    for tc in pending_tool_calls:
        if not has_msg_start:
            has_msg_start = True
            yield _sse_evt("message_start", {
                "type": "message_start",
                "message": {
                    "id": request_id, "type": "message", "role": "assistant",
                    "model": model, "content": [],
                    "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            })
        # Parse arguments for the tool_use input
        func = tc.get("function", {})
        raw_args = func.get("arguments", "{}")
        if isinstance(raw_args, str):
            try:
                tool_input = json_mod.loads(raw_args)
            except json_mod.JSONDecodeError:
                tool_input = {}
        else:
            tool_input = raw_args

        block = {
            "type": "tool_use",
            "id": tc.get("id", ""),
            "name": func.get("name", ""),
            "input": tool_input,
        }
        yield _sse_evt("content_block_start", {
            "type": "content_block_start",
            "index": next_block_index,
            "content_block": block,
        })
        yield _sse_evt("content_block_stop", {
            "type": "content_block_stop",
            "index": next_block_index,
        })
        next_block_index += 1
        finish_reason = "tool_calls"

    # ── Close the text block ──────────────────────────────────────
    if text_block_open:
        yield _sse_evt("content_block_stop", {
            "type": "content_block_stop",
            "index": text_block_index,
        })

    # ── Emit closing events ───────────────────────────────────────
    if has_msg_start:
        stop_reason = "end_turn" if finish_reason in (None, "stop") else "tool_use"
        yield _sse_evt("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason},
            "usage": {"output_tokens": len("".join(content_parts).split())},
        })
        yield _sse_evt("message_stop", {"type": "message_stop"})
    else:
        # No content produced — emit fallback text
        full_text = (
            "The request completed without producing content. "
            "This may indicate the model did not generate a response."
        )
        yield _sse_evt("message_start", {
            "type": "message_start",
            "message": {
                "id": request_id, "type": "message", "role": "assistant",
                "model": model, "content": [],
                "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })
        yield _sse_evt("content_block_start", {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        })
        yield _sse_evt("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": full_text},
        })
        yield _sse_evt("content_block_stop", {
            "type": "content_block_stop",
            "index": 0,
        })
        yield _sse_evt("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 0},
        })
        yield _sse_evt("message_stop", {"type": "message_stop"})


def _sse_evt(event: str, data: dict[str, Any]) -> str:
    """Format an Anthropic SSE event."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
