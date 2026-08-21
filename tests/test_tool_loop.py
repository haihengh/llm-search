"""Tests for the tool-call intercept loop."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_search.search.base import SearchProvider, SearchResult
from llm_search.fetch_page import extract_text_from_html, validate_url
from llm_search.tool_loop import (
    LMStudioError,
    ToolLoopExhaustedError,
    call_lm_studio_streaming,
    extract_assistant_message,
    is_context_overflow,
    run_tool_loop,
    run_tool_loop_streaming,
)


class TestExtractAssistantMessage:
    """Tests for response parsing."""

    def test_content_no_tool_calls(self):
        response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hello, how can I help?",
                }
            }]
        }
        content, tool_calls = extract_assistant_message(response)
        assert content == "Hello, how can I help?"
        assert tool_calls == []

    def test_tool_calls_no_content(self):
        response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": '{"query": "test"}',
                        },
                    }],
                }
            }]
        }
        content, tool_calls = extract_assistant_message(response)
        assert content is None
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "web_search"

    def test_empty_choices(self):
        response = {"choices": []}
        content, tool_calls = extract_assistant_message(response)
        assert content is None
        assert tool_calls == []


class FakeSearchProvider(SearchProvider):
    """Test double that returns canned results."""

    def __init__(self, results=None):
        self._results = results or []
        self._calls = []
        self._healthy = True

    @property
    def name(self) -> str:
        return "fake"

    async def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        self._calls.append((query, num_results))
        return self._results[:num_results]

    async def health_check(self) -> bool:
        return self._healthy


def make_mock_lm_response(content=None, tool_calls=None):
    """Build a mock LM Studio response dict."""
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message}]}


class TestToolLoop:
    """Tests for the tool-call intercept loop."""

    @pytest.mark.asyncio
    async def test_simple_answer_no_tool_calls(self):
        """LLM answers directly — no tool calls needed."""
        provider = FakeSearchProvider()
        mock_response = make_mock_lm_response(content="The answer is 42.")

        with patch("llm_search.tool_loop.call_lm_studio", new=AsyncMock(return_value=mock_response)):
            result = await run_tool_loop(
                messages=[{"role": "user", "content": "What is the answer?"}],
                search_provider=provider,
                model="test-model",
                lm_studio_url="http://localhost:1234/v1",
            )

        assert result["content"] == "The answer is 42."
        assert result["searches"] == 0
        assert result["iterations"] == 1
        assert result["tool_calls_count"] == 0

    @pytest.mark.asyncio
    async def test_length_truncation_without_content_raises_context_overflow(self):
        """finish_reason=length with no content and no tool calls → context
        overflow error, not a silently empty answer."""
        provider = FakeSearchProvider()
        mock_response = {
            "choices": [{
                "message": {"role": "assistant", "content": None},
                "finish_reason": "length",
            }],
            "usage": {"prompt_tokens": 90000, "completion_tokens": 8000},
        }

        with patch("llm_search.tool_loop.call_lm_studio", new=AsyncMock(return_value=mock_response)):
            with pytest.raises(LMStudioError) as excinfo:
                await run_tool_loop(
                    messages=[{"role": "user", "content": "Explain this."}],
                    search_provider=provider,
                    model="test-model",
                    lm_studio_url="http://localhost:1234/v1",
                )

        assert "context window" in str(excinfo.value)
        assert excinfo.value.status_code == 400

    @pytest.mark.asyncio
    async def test_single_search_then_answer(self):
        """LLM searches once, then answers."""
        provider = FakeSearchProvider(results=[
            SearchResult("Result 1", "https://ex.com/1", "Snippet 1", 1),
            SearchResult("Result 2", "https://ex.com/2", "Snippet 2", 2),
        ])

        # First call: LLM requests a search
        call1 = make_mock_lm_response(
            tool_calls=[{
                "id": "call_abc",
                "type": "function",
                "function": {
                    "name": "web_search",
                    "arguments": '{"query": "latest news"}',
                },
            }]
        )
        # Second call: LLM sees results and answers
        call2 = make_mock_lm_response(content="Based on the search results, the latest news is...")

        with patch(
            "llm_search.tool_loop.call_lm_studio",
            new=AsyncMock(side_effect=[call1, call2]),
        ):
            result = await run_tool_loop(
                messages=[{"role": "user", "content": "What's in the news?"}],
                search_provider=provider,
                model="test-model",
                lm_studio_url="http://localhost:1234/v1",
            )

        assert result["content"] == "Based on the search results, the latest news is..."
        assert result["searches"] == 1
        assert result["iterations"] == 2
        assert result["tool_calls_count"] == 1
        assert len(provider._calls) == 1
        assert provider._calls[0] == ("latest news", 5)

    @pytest.mark.asyncio
    async def test_multiple_searches(self):
        """LLM searches twice before answering."""
        provider = FakeSearchProvider(results=[
            SearchResult("R", "https://x.com", "S", 1),
        ])

        call1 = make_mock_lm_response(tool_calls=[{
            "id": "call_1",
            "type": "function",
            "function": {"name": "web_search", "arguments": '{"query": "first search"}'},
        }])
        call2 = make_mock_lm_response(tool_calls=[{
            "id": "call_2",
            "type": "function",
            "function": {"name": "web_search", "arguments": '{"query": "refined search"}'},
        }])
        call3 = make_mock_lm_response(content="Final answer after two searches.")

        with patch(
            "llm_search.tool_loop.call_lm_studio",
            new=AsyncMock(side_effect=[call1, call2, call3]),
        ):
            result = await run_tool_loop(
                messages=[{"role": "user", "content": "Complex question"}],
                search_provider=provider,
                model="test-model",
            )

        assert result["searches"] == 2
        assert result["iterations"] == 3
        assert len(provider._calls) == 2

    @pytest.mark.asyncio
    async def test_max_iterations_exceeded_returns_fallback(self):
        """When the LLM keeps searching, we return a graceful fallback
        with accumulated search results instead of raising an error."""
        provider = FakeSearchProvider(results=[
            SearchResult("R", "https://x.com", "S", 1),
        ])

        # Always return a tool call — never a plain answer. Each call uses a
        # NEW query: duplicate detection blocks repeated queries, so a
        # constant query would only ever search once.
        counter = {"n": 0}

        def _next_response(*args, **kwargs):
            counter["n"] += 1
            return make_mock_lm_response(tool_calls=[{
                "id": f"call_{counter['n']}",
                "type": "function",
                "function": {
                    "name": "web_search",
                    "arguments": json.dumps({"query": f"still searching {counter['n']}"}),
                },
            }])

        with patch(
            "llm_search.tool_loop.call_lm_studio",
            new=AsyncMock(side_effect=_next_response),
        ):
            result = await run_tool_loop(
                messages=[{"role": "user", "content": "Question"}],
                search_provider=provider,
                model="test-model",
                lm_studio_url="http://localhost:1234/v1",
            )

        # Should return a fallback, not raise an error
        assert result["finish_reason"] == "tool_loop_max"
        assert result["searches"] >= 5  # At least 5 searches happened
        assert "unable to synthesize" in result["content"].lower()
        assert "Search results:" in result["content"]

    @pytest.mark.asyncio
    async def test_lm_studio_unreachable(self):
        """LM Studio is down — should raise LMStudioError."""
        provider = FakeSearchProvider()

        with patch(
            "llm_search.tool_loop.call_lm_studio",
            new=AsyncMock(side_effect=LMStudioError("LM Studio not reachable")),
        ):
            with pytest.raises(LMStudioError, match="not reachable"):
                await run_tool_loop(
                    messages=[{"role": "user", "content": "Hello"}],
                    search_provider=provider,
                    model="test-model",
                )

    @pytest.mark.asyncio
    async def test_tool_with_json_object_arguments(self):
        """Some models send arguments as objects, not strings."""
        provider = FakeSearchProvider(results=[
            SearchResult("R", "https://x.com", "S", 1),
        ])

        call1 = make_mock_lm_response(tool_calls=[{
            "id": "call_obj",
            "type": "function",
            "function": {
                "name": "web_search",
                "arguments": {"query": "object args", "num_results": 3},
            },
        }])
        call2 = make_mock_lm_response(content="Answer.")

        with patch(
            "llm_search.tool_loop.call_lm_studio",
            new=AsyncMock(side_effect=[call1, call2]),
        ):
            result = await run_tool_loop(
                messages=[{"role": "user", "content": "Question"}],
                search_provider=provider,
                model="test-model",
            )

        assert result["searches"] == 1
        # num_results=3 was passed
        assert provider._calls[0] == ("object args", 3)

    @pytest.mark.asyncio
    async def test_client_tools_sent_to_llm_alongside_search(self):
        """Client tools are now sent to LM Studio alongside search tools.

        The LLM sees all available tools — both the client's tools and
        the auto-injected web_search + fetch_page. This preserves the
        local LLM's existing capabilities while adding search on top.
        """
        provider = FakeSearchProvider(results=[])
        mock_response = make_mock_lm_response(content="Done.")

        with patch("llm_search.tool_loop.call_lm_studio", new=AsyncMock(return_value=mock_response)) as mock_call:
            await run_tool_loop(
                messages=[{"role": "user", "content": "Hi"}],
                search_provider=provider,
                tools=[{"type": "function", "function": {"name": "calculator"}}],
                model="test-model",
            )

            tools_sent = mock_call.call_args.kwargs["tools"]
            tool_names = [t["function"]["name"] for t in tools_sent]
            assert "web_search" in tool_names
            assert "fetch_page" in tool_names
            assert "calculator" in tool_names  # Client tools are now included

    @pytest.mark.asyncio
    async def test_hallucinated_tool_blocked_not_passthrough(self):
        """LLM hallucinates a tool NOT in client tools → error fed back, loop continues."""
        provider = FakeSearchProvider()

        # LLM calls "bash" — not in TOOL_EXECUTORS and not in client tools
        call1 = make_mock_lm_response(
            content="Let me run a command.",
            tool_calls=[{
                "id": "call_bash",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": '{"command": "ls"}',
                },
            }]
        )
        # After getting error feedback, LLM produces a real answer
        call2 = make_mock_lm_response(content="I'll search instead.")

        with patch(
            "llm_search.tool_loop.call_lm_studio",
            new=AsyncMock(side_effect=[call1, call2]),
        ):
            result = await run_tool_loop(
                messages=[{"role": "user", "content": "List files"}],
                search_provider=provider,
                model="test-model",
                lm_studio_url="http://localhost:1234/v1",
            )

        # Should NOT passthrough — loop continues and gets a plain answer
        assert "tool_calls" not in result or not result.get("tool_calls")
        assert result["content"] == "I'll search instead."
        assert result["iterations"] == 2  # First blocked, second succeeded
        assert result["searches"] == 0

    @pytest.mark.asyncio
    async def test_mixed_search_executed_hallucination_blocked(self):
        """web_search is executed; hallucinated 'bash' (not in client tools) is blocked."""
        provider = FakeSearchProvider(results=[
            SearchResult("Result", "https://ex.com", "Snippet", 1),
        ])

        # LLM calls web_search AND "bash" in the same response
        # "bash" is NOT in client tools → hallucination
        call1 = make_mock_lm_response(
            content="Searching and then running a command...",
            tool_calls=[
                {
                    "id": "call_search",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": '{"query": "test"}'},
                },
                {
                    "id": "call_bash",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command": "ls"}'},
                },
            ],
        )
        # After bash is blocked, LLM sees search results and answers
        call2 = make_mock_lm_response(content="Based on search results: answer.")

        with patch(
            "llm_search.tool_loop.call_lm_studio",
            new=AsyncMock(side_effect=[call1, call2]),
        ):
            result = await run_tool_loop(
                messages=[{"role": "user", "content": "Search and list"}],
                search_provider=provider,
                model="test-model",
                lm_studio_url="http://localhost:1234/v1",
            )

        # web_search was executed, bash was blocked, loop continued
        assert "tool_calls" not in result or not result.get("tool_calls")
        assert result["content"] == "Based on search results: answer."
        assert result["searches"] == 1
        assert result["iterations"] == 2
        assert len(provider._calls) == 1  # web_search was executed

    @pytest.mark.asyncio
    async def test_all_recognized_tools_no_passthrough(self):
        """When all tool calls are recognised, loop continues normally."""
        provider = FakeSearchProvider(results=[
            SearchResult("R", "https://x.com", "S", 1),
        ])

        call1 = make_mock_lm_response(tool_calls=[{
            "id": "call_1",
            "type": "function",
            "function": {"name": "web_search", "arguments": '{"query": "test"}'},
        }])
        call2 = make_mock_lm_response(content="Final answer.")

        with patch(
            "llm_search.tool_loop.call_lm_studio",
            new=AsyncMock(side_effect=[call1, call2]),
        ):
            result = await run_tool_loop(
                messages=[{"role": "user", "content": "Search please"}],
                search_provider=provider,
                model="test-model",
                lm_studio_url="http://localhost:1234/v1",
            )

        # Normal loop — no passthrough needed
        assert "tool_calls" not in result or result.get("tool_calls") is None
        assert result["content"] == "Final answer."
        assert result["searches"] == 1
        assert result["iterations"] == 2

    # ── Client tool passthrough tests ──────────────────────────

    @pytest.mark.asyncio
    async def test_client_tool_passthrough(self):
        """LLM calls a client-provided tool → loop stops and returns it for passthrough."""
        provider = FakeSearchProvider()

        # LLM calls "read_file" — it IS in the client's tools list
        call1 = make_mock_lm_response(
            content="Let me read that file.",
            tool_calls=[{
                "id": "call_read",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": '{"path": "/tmp/test.txt"}',
                },
            }]
        )

        with patch(
            "llm_search.tool_loop.call_lm_studio",
            new=AsyncMock(return_value=call1),
        ):
            result = await run_tool_loop(
                messages=[{"role": "user", "content": "Read /tmp/test.txt"}],
                search_provider=provider,
                tools=[{"type": "function", "function": {"name": "read_file"}}],
                model="test-model",
                lm_studio_url="http://localhost:1234/v1",
            )

        # Should return the tool call for passthrough — NOT block it
        assert result.get("tool_calls") is not None
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "read_file"
        assert result["finish_reason"] == "tool_use"
        assert result["content"] == "Let me read that file."
        assert result["searches"] == 0
        assert result["iterations"] == 1

    @pytest.mark.asyncio
    async def test_mixed_search_executed_client_tool_passthrough(self):
        """web_search is executed; client tool is returned for passthrough."""
        provider = FakeSearchProvider(results=[
            SearchResult("Result", "https://ex.com", "Snippet", 1),
        ])

        # LLM calls web_search AND "read_file" (a client-provided tool)
        call1 = make_mock_lm_response(
            content="Searching and reading...",
            tool_calls=[
                {
                    "id": "call_search",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": '{"query": "test"}'},
                },
                {
                    "id": "call_read",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "/tmp/x.txt"}'},
                },
            ],
        )

        with patch(
            "llm_search.tool_loop.call_lm_studio",
            new=AsyncMock(return_value=call1),
        ):
            result = await run_tool_loop(
                messages=[{"role": "user", "content": "Search and read"}],
                search_provider=provider,
                tools=[{"type": "function", "function": {"name": "read_file"}}],
                model="test-model",
                lm_studio_url="http://localhost:1234/v1",
            )

        # web_search was executed (server-side), read_file is passthrough
        assert result.get("tool_calls") is not None
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "read_file"
        assert result["finish_reason"] == "tool_use"
        assert result["searches"] == 1
        assert len(provider._calls) == 1  # web_search was executed

    @pytest.mark.asyncio
    async def test_multiple_client_tools_all_passthrough(self):
        """LLM calls multiple client tools — all returned for passthrough."""
        provider = FakeSearchProvider()

        call1 = make_mock_lm_response(
            content=None,
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "/a.txt"}'},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command": "ls"}'},
                },
            ],
        )

        with patch(
            "llm_search.tool_loop.call_lm_studio",
            new=AsyncMock(return_value=call1),
        ):
            result = await run_tool_loop(
                messages=[{"role": "user", "content": "Read and list"}],
                search_provider=provider,
                tools=[
                    {"type": "function", "function": {"name": "read_file"}},
                    {"type": "function", "function": {"name": "bash"}},
                ],
                model="test-model",
                lm_studio_url="http://localhost:1234/v1",
            )

        assert result.get("tool_calls") is not None
        assert len(result["tool_calls"]) == 2
        names = [tc["function"]["name"] for tc in result["tool_calls"]]
        assert "read_file" in names
        assert "bash" in names
        assert result["finish_reason"] == "tool_use"

    @pytest.mark.asyncio
    async def test_hallucination_blocked_even_with_client_tools(self):
        """Tools NOT in client list are still blocked, even when client provides other tools."""
        provider = FakeSearchProvider()

        # LLM calls "write_file" — NOT in TOOL_EXECUTORS and NOT in client tools
        # (client only provides "read_file")
        call1 = make_mock_lm_response(
            tool_calls=[{
                "id": "call_write",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": '{"path": "/tmp/out.txt", "content": "test"}',
                },
            }]
        )
        call2 = make_mock_lm_response(content="I'll use the available tools instead.")

        with patch(
            "llm_search.tool_loop.call_lm_studio",
            new=AsyncMock(side_effect=[call1, call2]),
        ):
            result = await run_tool_loop(
                messages=[{"role": "user", "content": "Write a file"}],
                search_provider=provider,
                tools=[{"type": "function", "function": {"name": "read_file"}}],
                model="test-model",
                lm_studio_url="http://localhost:1234/v1",
            )

        # Hallucinated "write_file" was blocked — loop continued to plain answer
        assert "tool_calls" not in result or not result.get("tool_calls")
        assert result["content"] == "I'll use the available tools instead."
        assert result["iterations"] == 2


def make_sse_chunk(delta: dict, finish_reason=None) -> dict:
    """Build a single SSE data dict as LM Studio would stream it."""
    return {
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }


async def async_gen_from(items: list):
    """Helper: convert a list into an async generator."""
    for item in items:
        yield item


class TestCallLMStudioStreaming:
    """Tests for the streaming LM Studio client."""

    @pytest.mark.asyncio
    async def test_streams_content_chunks(self):
        """Streaming client yields parsed SSE data dicts."""
        sse_lines = [
            'data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"content":" world"},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]

        # Mock httpx.AsyncClient.stream to return SSE lines
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = MagicMock(return_value=async_gen_from(sse_lines))
        mock_response.raise_for_status = MagicMock()

        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=None)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_stream_ctx)
        mock_client_ctx = MagicMock()
        mock_client_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_ctx.__aexit__ = AsyncMock(return_value=None)

        httpx_patch = patch("llm_search.tool_loop.httpx.AsyncClient", return_value=mock_client_ctx)
        try:
            httpx_patch.start()
            chunks = []
            async for chunk in call_lm_studio_streaming(
                messages=[{"role": "user", "content": "Hi"}],
                tools=[],
                model="test-model",
                lm_studio_url="http://localhost:1234/v1",
            ):
                chunks.append(chunk)
        finally:
            httpx_patch.stop()

        assert len(chunks) == 3
        assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
        assert chunks[1]["choices"][0]["delta"] == {"content": "Hello"}
        assert chunks[2]["choices"][0]["delta"] == {"content": " world"}

    @pytest.mark.asyncio
    async def test_handles_connect_error(self):
        """Raises LMStudioError when LM Studio is unreachable."""
        mock_client = MagicMock()
        mock_client.stream = MagicMock(side_effect=LMStudioError("not reachable"))
        mock_client_ctx = MagicMock()
        mock_client_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch("llm_search.tool_loop.httpx.AsyncClient", return_value=mock_client_ctx):
            with patch("llm_search.tool_loop.httpx.ConnectError", LMStudioError):
                with pytest.raises(LMStudioError, match="not reachable"):
                    async for _ in call_lm_studio_streaming(
                        messages=[{"role": "user", "content": "Hi"}],
                        tools=[],
                        model="test-model",
                        lm_studio_url="http://localhost:1234/v1",
                    ):
                        pass

    @pytest.mark.asyncio
    async def test_error_chunk_raises_lm_studio_error(self):
        """LM Studio streams a failure as an SSE error chunk with HTTP 200
        (observed for context overflow) → raise LMStudioError, not a
        silently-skipped chunk."""
        sse_lines = [
            'data: {"error":{"message":"request (106724 tokens) exceeds the '
            'available context size (104192 tokens)","code":400}}',
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = MagicMock(return_value=async_gen_from(sse_lines))
        mock_response.raise_for_status = MagicMock()

        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=None)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_stream_ctx)
        mock_client_ctx = MagicMock()
        mock_client_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_ctx.__aexit__ = AsyncMock(return_value=None)

        httpx_patch = patch("llm_search.tool_loop.httpx.AsyncClient", return_value=mock_client_ctx)
        try:
            httpx_patch.start()
            with pytest.raises(LMStudioError) as excinfo:
                async for _ in call_lm_studio_streaming(
                    messages=[{"role": "user", "content": "Hi"}],
                    tools=[],
                    model="test-model",
                    lm_studio_url="http://localhost:1234/v1",
                ):
                    pass
        finally:
            httpx_patch.stop()

        assert "context size" in str(excinfo.value)
        assert excinfo.value.status_code == 400
        assert is_context_overflow(excinfo.value)


def _make_streaming_mock(chunks: list):
    """Return a callable that produces a fresh async generator each call."""
    async def _gen(*args, **kwargs):
        for item in chunks:
            yield item
    return _gen


class TestRunToolLoopStreaming:
    """Tests for the streaming tool-call loop."""

    @pytest.mark.asyncio
    async def test_no_tool_calls_streams_answer(self):
        """Simple Q&A: streams the answer without any searches."""
        provider = FakeSearchProvider()

        # Non-streaming check: no tool calls
        mock_check = make_mock_lm_response(content="Hello!")

        # Streaming chunks
        sse_chunks = [
            make_sse_chunk({"role": "assistant"}),
            make_sse_chunk({"content": "Hello"}, finish_reason=None),
            make_sse_chunk({"content": "!"}, finish_reason="stop"),
        ]

        with patch(
            "llm_search.tool_loop.call_lm_studio",
            new=AsyncMock(return_value=mock_check),
        ), patch(
            "llm_search.tool_loop.call_lm_studio_streaming",
            new=_make_streaming_mock(sse_chunks),
        ):
            events = []
            async for sse_str in run_tool_loop_streaming(
                messages=[{"role": "user", "content": "Hi"}],
                search_provider=provider,
                chatcmpl_id="test-123",
                created=1000,
                model="test-model",
                lm_studio_url="http://localhost:1234/v1",
            ):
                events.append(sse_str)

        # Should have: role chunk, content chunk, content+stop chunk, [DONE]
        assert len(events) >= 3
        assert "data: [DONE]" in events[-1]

        # First chunk should establish role
        first = json.loads(events[0][6:].strip())  # strip "data: "
        assert first["choices"][0]["delta"]["role"] == "assistant"
        assert first["id"] == "test-123"

        # Last content chunk should have finish_reason. The stream ends with
        # an "event: stats" SSE then [DONE], so filter to data events only.
        content_events = [
            json.loads(e[6:].strip())
            for e in events
            if e.startswith("data: ") and e[6:].strip() != "[DONE]"
        ]
        last_content = content_events[-1]
        assert last_content["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_with_tool_call_then_stream(self):
        """Tool call turn → search → streaming final answer."""
        provider = FakeSearchProvider(results=[
            SearchResult("Result 1", "https://ex.com/1", "Snippet 1", 1),
        ])

        # First non-streaming call: tool call
        mock_tool_call = make_mock_lm_response(tool_calls=[{
            "id": "call_abc",
            "type": "function",
            "function": {
                "name": "web_search",
                "arguments": '{"query": "latest news"}',
            },
        }])

        # Second non-streaming call (check after search): no tool calls
        mock_answer = make_mock_lm_response(content="Found it.")

        # Streaming chunks for final answer
        sse_chunks = [
            make_sse_chunk({"role": "assistant"}),
            make_sse_chunk({"content": "Found"}, finish_reason=None),
            make_sse_chunk({"content": " it."}, finish_reason="stop"),
        ]

        with patch(
            "llm_search.tool_loop.call_lm_studio",
            new=AsyncMock(side_effect=[mock_tool_call, mock_answer]),
        ), patch(
            "llm_search.tool_loop.call_lm_studio_streaming",
            new=_make_streaming_mock(sse_chunks),
        ):
            events = []
            async for sse_str in run_tool_loop_streaming(
                messages=[{"role": "user", "content": "News?"}],
                search_provider=provider,
                chatcmpl_id="test-456",
                created=2000,
                model="test-model",
                lm_studio_url="http://localhost:1234/v1",
            ):
                events.append(sse_str)

        # Verify streaming output
        assert any("Found" in e for e in events)
        assert "data: [DONE]" in events[-1]
        # Search was executed (verified via tool call processing in the loop)

    @pytest.mark.asyncio
    async def test_max_iterations_yields_fallback(self):
        """When LLM keeps searching, yields fallback content SSE and [DONE]."""
        provider = FakeSearchProvider(results=[
            SearchResult("R", "https://x.com", "S", 1),
        ])

        # Always stream a web_search tool call — never a plain answer.
        # The streaming loop only uses call_lm_studio_streaming; patching
        # the non-streaming client here would silently hit the real
        # LM Studio (and can hang for the whole context window).
        sse_chunks = [
            make_sse_chunk({"tool_calls": [{
                "index": 0,
                "id": "call_infinite",
                "type": "function",
                "function": {"name": "web_search", "arguments": ""},
            }]}),
            make_sse_chunk({"tool_calls": [{
                "index": 0,
                "function": {"arguments": '{"query": "still searching"}'},
            }]}, finish_reason="tool_calls"),
        ]

        with patch(
            "llm_search.tool_loop.call_lm_studio_streaming",
            new=_make_streaming_mock(sse_chunks),
        ):
            events = []
            async for sse_str in run_tool_loop_streaming(
                messages=[{"role": "user", "content": "Question"}],
                search_provider=provider,
                chatcmpl_id="test-err",
                created=3000,
                model="test-model",
                lm_studio_url="http://localhost:1234/v1",
            ):
                events.append(sse_str)

        # Should have fallback content and [DONE] — not an error
        assert "data: [DONE]" in events[-1]

        # Find the content chunk with finish_reason
        content_events = [
            json.loads(e[6:].strip())
            for e in events
            if e.startswith("data: ") and e[6:].strip() != "[DONE]"
        ]
        # Last content chunk should have finish_reason "tool_loop_max"
        last_chunk = content_events[-1]
        assert last_chunk["choices"][0]["finish_reason"] == "tool_loop_max"
        fallback_text = last_chunk["choices"][0]["delta"].get("content", "")
        assert "unable to synthesize" in fallback_text.lower()

    @pytest.mark.asyncio
    async def test_lm_studio_error_yields_sse_error(self):
        """LM Studio down → yields error SSE and [DONE]."""
        provider = FakeSearchProvider()

        # The streaming loop calls call_lm_studio_streaming (not the
        # non-streaming client), so that is what must be patched — the
        # previous patch target silently hit the real LM Studio.
        async def _error_gen(*args, **kwargs):
            raise LMStudioError("LM Studio not reachable")
            yield  # pragma: no cover — makes this an async generator

        with patch(
            "llm_search.tool_loop.call_lm_studio_streaming",
            new=_error_gen,
        ):
            events = []
            async for sse_str in run_tool_loop_streaming(
                messages=[{"role": "user", "content": "Hi"}],
                search_provider=provider,
                chatcmpl_id="test-err2",
                created=4000,
                model="test-model",
                lm_studio_url="http://localhost:1234/v1",
            ):
                events.append(sse_str)

        error_event = json.loads(events[0][6:].strip())
        assert error_event["error"]["type"] == "lm_studio_error"
        assert "not reachable" in error_event["error"]["message"]
        assert "data: [DONE]" in events[-1]

    @pytest.mark.asyncio
    async def test_length_truncation_with_only_reasoning_yields_context_overflow(self):
        """Model reasons to the context wall (finish_reason=length, zero
        content) → context-overflow error SSE; CoT is NOT relayed as text."""
        provider = FakeSearchProvider()
        sse_chunks = [
            make_sse_chunk({"reasoning_content": "Let me think..."}),
            make_sse_chunk({"reasoning_content": " ...still thinking..."}),
            make_sse_chunk({}, finish_reason="length"),
        ]

        with patch(
            "llm_search.tool_loop.call_lm_studio_streaming",
            new=_make_streaming_mock(sse_chunks),
        ):
            events = []
            async for sse_str in run_tool_loop_streaming(
                messages=[{"role": "user", "content": "Question"}],
                search_provider=provider,
                chatcmpl_id="test-length1",
                created=5000,
                model="test-model",
                lm_studio_url="http://localhost:1234/v1",
                relay_reasoning=False,
            ):
                events.append(sse_str)

        first = json.loads(events[0][6:].strip())
        assert first["error"]["type"] == "context_overflow"
        assert "prompt is too long" in first["error"]["message"]
        # The half-finished chain-of-thought must not leak to the caller.
        assert not any("still thinking" in e for e in events)
        assert "data: [DONE]" in events[-1]

    @pytest.mark.asyncio
    async def test_length_truncation_with_empty_response_yields_context_overflow(self):
        """A single empty chunk with finish_reason=length (no generation
        room left) → context-overflow error, not the fake empty-response
        diagnostic text."""
        provider = FakeSearchProvider()
        sse_chunks = [
            make_sse_chunk({}, finish_reason="length"),
        ]

        with patch(
            "llm_search.tool_loop.call_lm_studio_streaming",
            new=_make_streaming_mock(sse_chunks),
        ):
            events = []
            async for sse_str in run_tool_loop_streaming(
                messages=[{"role": "user", "content": "Question"}],
                search_provider=provider,
                chatcmpl_id="test-length2",
                created=6000,
                model="test-model",
                lm_studio_url="http://localhost:1234/v1",
                relay_reasoning=False,
            ):
                events.append(sse_str)

        first = json.loads(events[0][6:].strip())
        assert first["error"]["type"] == "context_overflow"
        assert "prompt is too long" in first["error"]["message"]
        assert not any("returned an empty response" in e for e in events)
        assert "data: [DONE]" in events[-1]

    @pytest.mark.asyncio
    async def test_hallucinated_tool_blocked_streaming(self):
        """Streaming: hallucinated tool (not in client tools) → error fed to LLM, loop continues."""
        provider = FakeSearchProvider()

        # LLM hallucinates "bash" — not in TOOL_EXECUTORS, no client tools provided
        call1 = make_mock_lm_response(
            content="Let me check something.",
            tool_calls=[{
                "id": "call_bash",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": '{"command": "ls"}',
                },
            }]
        )
        # After error feedback, LLM gives a plain answer
        call2 = make_mock_lm_response(content="I'll use web_search instead.")

        # Streaming chunks for the final answer turn
        sse_chunks = [
            make_sse_chunk({"role": "assistant"}),
            make_sse_chunk({"content": "I'll use web_search instead."}, finish_reason="stop"),
        ]

        with patch(
            "llm_search.tool_loop.call_lm_studio",
            new=AsyncMock(side_effect=[call1, call2]),
        ), patch(
            "llm_search.tool_loop.call_lm_studio_streaming",
            new=_make_streaming_mock(sse_chunks),
        ):
            events = []
            async for sse_str in run_tool_loop_streaming(
                messages=[{"role": "user", "content": "List files"}],
                search_provider=provider,
                chatcmpl_id="test-blocked",
                created=5000,
                model="test-model",
                lm_studio_url="http://localhost:1234/v1",
            ):
                events.append(sse_str)

        # Should end normally with [DONE] — no tool_calls passthrough
        assert "data: [DONE]" in events[-1]

        # Should NOT contain any tool_calls delta chunks
        all_events = [
            json.loads(e[6:].strip())
            for e in events
            if e.startswith("data: ") and e[6:].strip() != "[DONE]"
        ]
        tool_deltas = [
            e for e in all_events
            if e.get("choices", [{}])[0].get("delta", {}).get("tool_calls")
        ]
        assert len(tool_deltas) == 0  # No passthrough — hallucinated tools blocked

        # Should contain the final answer
        content_texts = [
            e["choices"][0]["delta"].get("content", "")
            for e in all_events
        ]
        assert any("web_search instead" in c for c in content_texts)

    # ── Client tool passthrough (streaming) ────────────────────

    @pytest.mark.asyncio
    async def test_client_tool_passthrough_streaming(self):
        """Streaming: LLM calls client-provided tool → emitted as SSE delta, stream ends."""
        provider = FakeSearchProvider()

        # LLM calls "read_file" — it IS in client tools. The streaming loop
        # only uses call_lm_studio_streaming; patching the non-streaming
        # client here would silently hit the real LM Studio.
        sse_chunks = [
            make_sse_chunk({"role": "assistant"}),
            make_sse_chunk({"content": "Let me read that file."}),
            make_sse_chunk({"tool_calls": [{
                "index": 0,
                "id": "call_read",
                "type": "function",
                "function": {"name": "read_file", "arguments": ""},
            }]}),
            make_sse_chunk({"tool_calls": [{
                "index": 0,
                "function": {"arguments": '{"path": "/tmp/test.txt"}'},
            }]}, finish_reason="tool_calls"),
        ]

        with patch(
            "llm_search.tool_loop.call_lm_studio_streaming",
            new=_make_streaming_mock(sse_chunks),
        ):
            events = []
            async for sse_str in run_tool_loop_streaming(
                messages=[{"role": "user", "content": "Read /tmp/test.txt"}],
                search_provider=provider,
                tools=[{"type": "function", "function": {"name": "read_file"}}],
                chatcmpl_id="test-passthrough",
                created=6000,
                model="test-model",
                lm_studio_url="http://localhost:1234/v1",
            ):
                events.append(sse_str)

        # Should contain [DONE]
        assert "data: [DONE]" in events[-1]

        # Should contain tool_calls delta chunks
        all_events = [
            json.loads(e[6:].strip())
            for e in events
            if e.startswith("data: ") and e[6:].strip() != "[DONE]"
        ]
        tool_deltas = [
            e for e in all_events
            if e.get("choices", [{}])[0].get("delta", {}).get("tool_calls")
        ]
        assert len(tool_deltas) == 1  # One passthrough tool call
        tc = tool_deltas[0]["choices"][0]["delta"]["tool_calls"][0]
        assert tc["function"]["name"] == "read_file"
        assert tool_deltas[0]["choices"][0]["finish_reason"] == "tool_use"



class TestValidateURL:
    """Tests for URL validation (security checks)."""

    def test_allows_https_url(self):
        assert validate_url("https://example.com/page") is None

    def test_allows_http_url(self):
        assert validate_url("http://example.com") is None

    def test_rejects_file_protocol(self):
        assert "Unsupported protocol" in validate_url("file:///etc/passwd")

    def test_rejects_localhost(self):
        assert "localhost" in validate_url("http://localhost:8080/admin")

    def test_rejects_loopback_ip(self):
        assert "private/internal" in validate_url("http://127.0.0.1/test")

    def test_rejects_private_ip(self):
        assert "private/internal" in validate_url("http://192.168.1.1/admin")
        assert "private/internal" in validate_url("http://10.0.0.1/")

    def test_rejects_no_hostname(self):
        assert validate_url("not-a-url") is not None


class TestExtractTextFromHTML:
    """Tests for HTML to text extraction."""

    def test_extracts_visible_text(self):
        html = "<html><body><p>Hello world</p></body></html>"
        text = extract_text_from_html(html)
        assert "Hello world" in text

    def test_strips_scripts(self):
        html = "<html><script>alert('xss')</script><p>Safe text</p></html>"
        text = extract_text_from_html(html)
        assert "alert" not in text
        assert "Safe text" in text

    def test_strips_styles(self):
        html = "<html><style>.x{color:red}</style><p>Visible</p></html>"
        text = extract_text_from_html(html)
        assert "color:red" not in text
        assert "Visible" in text

    def test_handles_empty(self):
        assert extract_text_from_html("") == ""

    def test_handles_plain_text(self):
        assert extract_text_from_html("Just plain text") == "Just plain text"


class TestFetchPageInToolLoop:
    """Tests for fetch_page tool in the tool-call loop."""

    @pytest.mark.asyncio
    async def test_fetch_page_is_auto_injected(self):
        """fetch_page tool is injected alongside web_search."""
        provider = FakeSearchProvider(results=[])
        mock_response = make_mock_lm_response(content="Done.")

        with patch(
            "llm_search.tool_loop.call_lm_studio",
            new=AsyncMock(return_value=mock_response),
        ) as mock_call:
            await run_tool_loop(
                messages=[{"role": "user", "content": "Read this page"}],
                search_provider=provider,
                model="test-model",
                lm_studio_url="http://localhost:1234/v1",
            )

            tools_sent = mock_call.call_args.kwargs["tools"]
            tool_names = [t["function"]["name"] for t in tools_sent]
            assert "web_search" in tool_names
            assert "fetch_page" in tool_names

    @pytest.mark.asyncio
    async def test_fetch_page_execution(self):
        """fetch_page tool calls are dispatched correctly."""
        provider = FakeSearchProvider(results=[])

        # LLM calls fetch_page
        call1 = make_mock_lm_response(tool_calls=[{
            "id": "call_fetch",
            "type": "function",
            "function": {
                "name": "fetch_page",
                "arguments": '{"url": "https://example.com"}',
            },
        }])
        call2 = make_mock_lm_response(content="I read the page.")

        with patch(
            "llm_search.tool_loop.call_lm_studio",
            new=AsyncMock(side_effect=[call1, call2]),
        ), patch(
            "llm_search.tool_registry.fetch_page_text",
            new=AsyncMock(return_value="This is the page content."),
        ):
            result = await run_tool_loop(
                messages=[{"role": "user", "content": "Read example.com"}],
                search_provider=provider,
                model="test-model",
                lm_studio_url="http://localhost:1234/v1",
            )

        assert result["content"] == "I read the page."
        assert result["tool_calls_count"] == 1
