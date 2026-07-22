# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.5] — 2026-07-21

### Added
- **Built-in chat client** — a lightweight web chat UI (`chat-client/`) that launches alongside the middleware. Features streaming chat, image upload (paste or pick, sent as OpenAI vision-format), file upload (text extraction), model selector, dark mode, and markdown rendering. Accessible at `http://localhost:8080`.
- **Chat client proxy server** (`chat-client/server.py`) — FastAPI app that serves the static UI and reverse-proxies `/v1/*`, `/health`, `/stats` to the middleware with SSE streaming passthrough.

### Fixed
- **SSE streaming in chat client proxy** — the `httpx.AsyncClient` was used within `async with`, closing the TCP connection before FastAPI's `StreamingResponse` could read the body. Fixed by keeping the client alive inside the async generator and closing it in `finally` after the stream is exhausted.

### Changed
- **docker-compose.yml** — now starts three services: `searxng`, `llm-search`, and `chat-client` (optional — comment out to disable).

## [0.2.4] — 2026-07-20

### Added
- **OpenAI Responses API endpoint (`POST /v1/responses`)** — the middleware now speaks the Responses API protocol required by Codex Desktop, GPT-5.x, and future OpenAI clients. Includes full streaming SSE support with proper lifecycle events (`response.created` → `output_item.added` → `output_text.delta` → `response.completed`).
- **Codex Desktop app setup docs** — step-by-step guide for Windows and macOS in both English and Chinese READMEs.

### Fixed
- **Tool schema sanitization** — Codex sends tool definitions with missing `type: "object"` in parameters schemas. The Responses adapter now normalizes tool schemas before forwarding to LM Studio, preventing 400 errors.

## [0.2.3] — 2026-07-16

### Fixed
- **"request completed without producing content" error** — the streaming Anthropic adapter could lose text content when the model's streaming deltas contained tool calls alongside text, triggering the empty-response fallback. Refactored `run_tool_loop_streaming` to a single-pass design: one LM Studio streaming call per iteration that simultaneously relays text and accumulates tool-call fragments.
- **Context overflow → useless error message** — when the conversation exceeded the model's context window, the 400 error was embedded inside the SSE stream where Claude Code couldn't see it. The middleware now peeks at the first SSE event before committing to `StreamingResponse`; if it's an error, the error is returned as a proper HTTP 400 (`invalid_request_error` / "prompt is too long") which triggers Claude Code's auto-compaction.
- **Anthropic `message_start` missing fields** — added `stop_reason` and `stop_sequence` fields (both `null`) to all `message_start` SSE events for spec compliance.
- **Empty final response after searches** — if the model executed searches but then produced an empty text response, the caller saw nothing. The middleware now falls back to streaming the raw search results as the response content.
- **Completely empty response (no text, no searches, no tool calls)** — the model sometimes returns a single empty chunk for probing/ping requests, which triggered the confusing "request completed without producing content" fallback. Now yields a diagnostic message explaining possible causes (prompt too long, model not loaded, unsupported task).
- **Anthropic request logging** — the `/v1/messages` handler now logs `model`, `stream`, and `messages` count for every request to aid debugging.

### Changed
- **Streaming refactored to single-pass** — the old "non-streaming check + re-generate streaming" approach (two LM Studio calls per iteration) is replaced with one streaming call per iteration. Tool-call fragments are accumulated across chunks and assembled after the stream finishes. Hallucinated client tools (bash, read, write) are blocked server-side with error feedback to the LLM.
- **`LMStudioError` carries `status_code`** — enables reliable context-overflow detection without fragile string matching.
- **Version tracking** — `server.py` now reads `__version__` from the package instead of hardcoding `v0.1.0`.

## [0.2.2] — 2026-07-14

### Fixed
- **"invalid tool parameters" error** — removed passthrough of unrecognised tool calls. Claude-distilled models hallucinate Bash/Read/Write calls with wrong parameters. The middleware now feeds an error back to the LLM and continues the loop instead of passing garbage tool calls to Claude Code.

## [0.2.1] — 2026-07-14

### Fixed
- **Claude Code "tool call could not be parsed" error** — the streaming Anthropic adapter silently dropped `tool_calls` from delta chunks, causing `stop_reason: "tool_use"` with zero tool_use blocks. Fixed by rewriting `anthropic_stream_from_openai` to capture and emit proper `tool_use` content blocks.
- **Malformed passthrough tool calls** — added `_validate_openai_tool_call()` to filter out hallucinated tool calls with empty names, unparseable JSON, or missing function objects before they reach the client.
- **Empty tool name passthrough** — `run_tool_loop` and `run_tool_loop_streaming` now drop tool calls with empty names instead of passing them through.

## [0.2.0] — 2026-07-09

### Added
- **Tool filtering** — only `web_search` + `fetch_page` reach the LLM. Client tools (Bash, Read, etc.) from Claude Code are stripped to prevent small local models from getting confused by 12+ tool definitions.
- **Tool passthrough** — if the LLM calls an unrecognised tool (e.g. hallucinated name), the loop exits early and returns it to the client instead of wasting iterations.
- **Graceful exhaustion fallback** — when the tool loop reaches max iterations, accumulated search results are returned as content rather than throwing an error. Claude Code can work with the raw search results even if the LLM doesn't synthesise a final answer.
- **Answer nudge** — on later loop iterations, the middleware injects a reminder telling the LLM to synthesise an answer now rather than searching again.

### Changed
- **Max tool loop iterations**: 5 → 10. Gives local models more room to converge before the fallback kicks in.
- Tool loop no longer raises `ToolLoopExhaustedError` — returns a graceful fallback instead.
- Anthropic adapter now converts passthrough tool calls to proper `tool_use` content blocks with `stop_reason: "tool_use"`.

### Fixed
- **Claude Code "tool loop exceeded maximum iterations" bug** — caused by flooding the LLM with 12+ client tools. Fixed via tool filtering + passthrough + fallback.

## [0.1.2] — 2026-07-07

### Added
- `GET /v1/models` endpoint — proxies model list from LLM backend
- Cross-machine support — middleware on one computer, LLM on another
- Cross-machine setup documented in README

## [0.1.1] — 2026-07-07

### Added
- `POST /v1/messages` endpoint — Anthropic Messages API for Claude Code
- Anthropic adapter (`anthropic_adapter.py`) — translates Anthropic ↔ OpenAI formats
- Chinese documentation (`README_zh.md`, `docs/architecture_zh.md`, `docs/requirements_zh.md`)
- Language switcher links in README

## [0.1.0] — 2026-07-07

### Added
- `web_search` tool — internet search via self-hosted SearXNG (80+ engines)
- `fetch_page` tool — fetch and read full web page content from URLs
- Streaming support (`stream: true`) — SSE token-by-token output
- MCP server (`mcp_server.py`) — Model Context Protocol for Claude Desktop
- GitHub Actions publish workflow — builds and pushes to GHCR + Docker Hub on `v*` tags
- pip package with CLI entry point (`llm-search`)
- `--mcp` flag for MCP server mode
- Docker Compose deployment — one command to start SearXNG + middleware
- Configurable timeout (`lm_studio_timeout`)
- Client setup guides: LM Studio, Ollama, Claude Code, Claude Desktop, Cursor, Continue.dev, Open WebUI
- Model compatibility test results (5 working, 3 failing)
- 30 unit tests

### Fixed
- `num_results` default in `execute_web_search()` — optional parameter now has default
- Health check uses SearXNG `/healthz` endpoint (no more search engine rate limits from health checks)

### Changed
- `inject_web_search_tool()` → `inject_tools()` — injects both `web_search` and `fetch_page`

[0.1.2]: https://github.com/haihengh/llm-search/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/haihengh/llm-search/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/haihengh/llm-search/releases/tag/v0.1.0
[0.2.1]: https://github.com/haihengh/llm-search/compare/v0.2.0...v0.2.1
[0.2.3]: https://github.com/haihengh/llm-search/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/haihengh/llm-search/compare/v0.2.1...v0.2.2
[0.2.5]: https://github.com/haihengh/llm-search/compare/v0.2.4...v0.2.5
