"""Tests for the tool-call intercept loop."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_search.search.base import SearchProvider, SearchResult
from llm_search.tool_loop import (
    LMStudioError,
    ToolLoopExhaustedError,
    extract_assistant_message,
    run_tool_loop,
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

            # Verify the call to LM Studio included both tools
            tools_sent = mock_call.call_args.kwargs["tools"]
            tool_names = [t["function"]["name"] for t in tools_sent]
            assert "web_search" in tool_names
            assert "calculator" in tool_names
