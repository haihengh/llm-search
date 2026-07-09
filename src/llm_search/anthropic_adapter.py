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


# ── OpenAI → Anthropic Translation ─────────────────────────────

def _openai_tool_calls_to_anthropic_content(
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert OpenAI-format tool_calls to Anthropic tool_use content blocks.

    OpenAI:  {"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": "{...}"}}
    Anthropic: {"type": "tool_use", "id": "call_1", "name": "bash", "input": {...}}
    """
    blocks: list[dict[str, Any]] = []
    for tc in tool_calls:
        func = tc.get("function", {})
        raw_args = func.get("arguments", "{}")
        if isinstance(raw_args, str):
            try:
                import json as _json
                parsed = _json.loads(raw_args)
            except _json.JSONDecodeError:
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
    finish_reason = openai_result.get("finish_reason", "end_turn")

    # Build content blocks
    content_blocks: list[dict[str, Any]] = []
    if content_text:
        content_blocks.append({"type": "text", "text": content_text})

    if passthrough_tool_calls:
        content_blocks.extend(
            _openai_tool_calls_to_anthropic_content(passthrough_tool_calls)
        )

    stop_reason = "tool_use" if passthrough_tool_calls else "end_turn"

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

    Anthropic streaming uses:
      event: message_start
      data: {"type": "message_start", "message": {...}}

      event: content_block_start
      data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}

      event: content_block_delta
      data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "..."}}

      event: content_block_stop
      data: {"type": "content_block_stop", "index": 0}

      event: message_delta
      data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {...}}

      event: message_stop
      data: {"type": "message_stop"}
    """
    import json as json_mod

    # We need to collect the full content from OpenAI SSE chunks,
    # then emit Anthropic-format SSE events.
    content_parts = []
    finish_reason = None
    started = False
    error_message = None

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

            # Check for error SSE chunks (e.g. tool_loop_exhausted)
            if "error" in chunk:
                error_message = chunk["error"].get("message", "Unknown error")
                continue

            choices = chunk.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta", {})
            fr = choices[0].get("finish_reason")

            # Extract content
            content = delta.get("content", "")
            if content:
                if not started:
                    # Emit message_start
                    yield (
                        "event: message_start\n"
                        f"data: {json_mod.dumps({'type': 'message_start', 'message': {'id': request_id, 'type': 'message', 'role': 'assistant', 'model': model, 'content': [], 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
                    )
                    # Emit content_block_start
                    yield (
                        "event: content_block_start\n"
                        f"data: {json_mod.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                    )
                    started = True

                # Emit content_block_delta
                yield (
                    "event: content_block_delta\n"
                    f"data: {json_mod.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': content}})}\n\n"
                )
                content_parts.append(content)

            if fr is not None:
                finish_reason = fr

    # Emit closing events
    if started:
        yield (
            "event: content_block_stop\n"
            f"data: {json_mod.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
        )

        stop_reason = "end_turn" if finish_reason == "stop" else "tool_use"
        yield (
            "event: message_delta\n"
            f"data: {json_mod.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason}, 'usage': {'output_tokens': len(''.join(content_parts).split())}})}\n\n"
        )

        yield (
            "event: message_stop\n"
            f"data: {json_mod.dumps({'type': 'message_stop'})}\n\n"
        )
    else:
        # No content was produced — emit the captured error or a generic fallback
        full_text = error_message or (
            "The request completed without producing content. "
            "This may indicate the model did not generate a response."
        )
        yield (
            "event: message_start\n"
            f"data: {json_mod.dumps({'type': 'message_start', 'message': {'id': request_id, 'type': 'message', 'role': 'assistant', 'model': model, 'content': [], 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
        )
        yield (
            "event: content_block_start\n"
            f"data: {json_mod.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
        )
        yield (
            "event: content_block_delta\n"
            f"data: {json_mod.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': full_text}})}\n\n"
        )
        yield (
            "event: content_block_stop\n"
            f"data: {json_mod.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
        )
        yield (
            "event: message_delta\n"
            f"data: {json_mod.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {'output_tokens': 0}})}\n\n"
        )
        yield (
            "event: message_stop\n"
            f"data: {json_mod.dumps({'type': 'message_stop'})}\n\n"
        )
