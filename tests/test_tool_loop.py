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
    async def test_max_iterations_exceeded(self):
        """When the LLM keeps searching, we eventually stop."""
        provider = FakeSearchProvider(results=[
            SearchResult("R", "https://x.com", "S", 1),
        ])

        # Always return a tool call — never a plain answer
        always_search = make_mock_lm_response(tool_calls=[{
            "id": "call_infinite",
            "type": "function",
            "function": {"name": "web_search", "arguments": '{"query": "still searching"}'},
        }])

        with patch(
            "llm_search.tool_loop.call_lm_studio",
            new=AsyncMock(return_value=always_search),
        ):
            with pytest.raises(ToolLoopExhaustedError) as exc_info:
                await run_tool_loop(
                    messages=[{"role": "user", "content": "Question"}],
                    search_provider=provider,
                    model="test-model",
                    lm_studio_url="http://localhost:1234/v1",
                )

        assert "exceeded maximum iterations" in str(exc_info.value)

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
    async def test_client_provided_tools_are_preserved(self):
        """Client's custom tools are kept alongside injected web_search."""
        provider = FakeSearchProvider(results=[])
        mock_response = make_mock_lm_response(content="Done.")

        with patch("llm_search.tool_loop.call_lm_studio", new=AsyncMock(return_value=mock_response)) as mock_call:
            await run_tool_loop(
                messages=[{"role": "user", "content": "Hi"}],
                search_provider=provider,
                tools=[{"type": "function", "function": {"name": "calculator"}}],
                model="test-model",
            )

            # Verify the call to LM Studio included all tools
            tools_sent = mock_call.call_args.kwargs["tools"]
            tool_names = [t["function"]["name"] for t in tools_sent]
            assert "web_search" in tool_names
            assert "fetch_page" in tool_names
            assert "calculator" in tool_names


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

        # Last content chunk should have finish_reason
        last_content = json.loads(events[-2][6:].strip())
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
    async def test_max_iterations_yields_error(self):
        """When LLM keeps searching, yields error SSE and [DONE]."""
        provider = FakeSearchProvider(results=[
            SearchResult("R", "https://x.com", "S", 1),
        ])

        # Always return a tool call — never a plain answer
        always_search = make_mock_lm_response(tool_calls=[{
            "id": "call_infinite",
            "type": "function",
            "function": {
                "name": "web_search",
                "arguments": '{"query": "still searching"}',
            },
        }])

        with patch(
            "llm_search.tool_loop.call_lm_studio",
            new=AsyncMock(return_value=always_search),
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

        # Should have error and [DONE]
        error_event = json.loads(events[0][6:].strip())
        assert error_event["error"]["type"] == "tool_loop_exhausted"
        assert "data: [DONE]" in events[-1]

    @pytest.mark.asyncio
    async def test_lm_studio_error_yields_sse_error(self):
        """LM Studio down → yields error SSE and [DONE]."""
        provider = FakeSearchProvider()

        with patch(
            "llm_search.tool_loop.call_lm_studio",
            new=AsyncMock(side_effect=LMStudioError("LM Studio not reachable")),
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
