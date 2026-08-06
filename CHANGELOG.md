# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.1] — 2026-08-05

### Added
- **Reasoning toggle (🧠)** — header button to enable/disable chain-of-thought. When off, the middleware injects a system prompt telling the model to answer directly and suppresses `reasoning_content` if the model also produces `content`. Pure reasoning models fall back to showing reasoning so the screen is never blank. State persisted to `localStorage`.
- **Prompt token estimation for streaming** — LM Studio doesn't stream `usage`, so prompt tokens are now estimated from conversation length (`chars ÷ 4`) matching the completion-token heuristic. This gives meaningful prompt_tokens_per_second in the stats panel even for streaming-only workloads.

### Fixed
- **Blank screen on reasoning models** — both `tool_loop.py` and `app.js` now capture `delta.reasoning_content` in addition to `delta.content`, so Qwen3.6/DeepSeek-R1 style models render text instead of a blank page.
- **Stats fields stuck at "—"** — `updateLiveStats()` now always resets token speed fields (was only writing on `> 0`, leaving stale values).
- **Mobile input bar bottom padding** — increased from 22px to 44px so the input doesn't sit at the screen edge.

## [0.3.0] — 2026-08-05

### Added
- **Settings modal in chat UI** — click the ⚙️ gear icon to change LLM URL, search provider, timeout, and other settings at runtime without editing files or restarting. Backed by `GET /v1/config` and `PUT /v1/config` API endpoints with a `RuntimeConfig` layer that applies changes immediately.
- **Performance stats sidebar** — click the 📊 chart icon to open a real-time performance panel showing LLM latency, token generation speed (tokens/sec), prompt processing speed, tool-call breakdowns (web searches, page fetches, passthrough tools, blocked hallucinations), cache hit rates, and session uptime. Polls `/stats` every 3 seconds and updates live during streaming via SSE `event: stats` events.
- **Token-level timing** — the middleware now captures token usage from LM Studio responses (`usage.prompt_tokens`, `usage.completion_tokens`) and measures tokens-per-second. Falls back to wall-clock timing and content-length estimation when the backend doesn't provide `timings` data.
- **`max_search_results` runtime config** — the search result count cap is now configurable via the settings modal and enforced by `tool_registry.py`.

### Changed
- **Runtime-configurable settings** — `tool_loop.py` and `server.py` now read from `RuntimeConfig` (mutable) instead of `Settings` (env-var immutable) for `lm_studio_url`, `max_tool_loop_iterations`, `max_client_tools`, `lm_studio_timeout`, and search provider settings. Changes take effect on the next request.
- **Enhanced `/stats` endpoint** — now returns `total_fetch_pages`, `total_hallucinated_tools`, `total_passthrough_tools`, `llm_call_count`, `llm_avg_ms`, `total_prompt_tokens`, `total_completion_tokens`, `tokens_per_second`, `prompt_tokens_per_second`.
- **Search provider hot-swap** — changing `search_provider` or `searxng_url` via the API recreates the search provider singleton without restarting. On failure (e.g. missing API key), the old provider is kept.
- **Chat client layout** — the app now lives in a flex wrapper (`#wrapper`) that houses the main chat app and the collapsible stats sidebar side-by-side. On mobile, the sidebar overlays as a fixed panel.
- **`max_search_results` wired up** — previously defined in config but never enforced; now clamps search result counts in `execute_web_search()`.

### Fixed
- **Streaming stats never aggregated** — `run_tool_loop_streaming()` now accepts a `stats_out` parameter (a mutable list). All three streaming paths (chat completions, Anthropic, Responses) capture per-request stats via `finally` blocks and call `_ingest_request_stats()`, so global counters accumulate from streaming requests.
- **Anthropic + Responses non-streaming stats** — both paths were missing `_ingest_request_stats()` calls; added alongside the existing chat completions path.
- **Stray text artifact in `tool_loop.py`** — fixed a `vnvn` prefix on the `TimeoutException` handler line.
- **Unused `settings` import in `tool_loop.py`** — cleaned up after migrating to `RuntimeConfig`.

## [0.2.8] — 2026-07-28

### Added
- **Default model option in chat UI** — the model dropdown now shows "🔄 Use currently loaded model" as the first and default option, so users don't need to pick a specific model. Whatever model is loaded in LM Studio is used automatically. The model list is still available for explicit selection.

### Fixed
- **Missing `role: "assistant"` in empty-response SSE chunks** — several edge-case paths in the streaming tool loop (empty response after search, completely empty response) were missing the required `role` delta, which could cause some clients to hang or error.
- **`httpx.ReadTimeout` not caught** — streaming timeouts during `aiter_lines()` were not handled by the existing `TimeoutException` catch, causing 500 errors on slow LLM responses. Both `TimeoutException` and `ReadTimeout` are now caught in both streaming and non-streaming paths.
- **Configurable timeout and tool limits in Docker Compose** — `MAX_CLIENT_TOOLS` and `LM_STUDIO_TIMEOUT` are now exposed as environment variables in `docker-compose.yml`, matching the already-documented `.env.example` settings.

## [0.2.7] — 2026-07-25

### Changed
- **Client tool passthrough** — the middleware now passes all client-provided tools (Bash, Read, Write, etc.) to the LLM alongside `web_search` and `fetch_page`. The local LLM sees all available tools, preserving existing capabilities while adding search on top. Previously, client tools were hidden from the LLM to reduce tool-definition load on small models.
- **Three-way tool classification** — tool calls are now classified as search tools (executed server-side), client tools (returned to the caller for execution), or hallucinations (blocked with error feedback). This means Claude Code, Codex, and other clients can now use their native tools through the middleware while still getting search augmentation.
- **Streaming tool passthrough** — client tool calls in streaming mode are emitted as SSE delta chunks with `finish_reason: "tool_use"`, allowing the Anthropic and Responses adapters to capture and relay them to the caller.

### Fixed
- **Local file access blocked** — `fetch_page` rejected `file://` URLs with "only http/https allowed". Now, if a client provides a `read_file` tool, the LLM can call it and the call is passed through to the client for local execution.

## [0.2.6] — 2026-07-23

### Added
- **PWA support for the chat client** — the built-in chat UI is now a Progressive Web App with a custom app icon. Add it to your phone's home screen (iOS or Android) and it launches fullscreen like a native app. Includes a service worker for offline caching and an install banner with platform-specific instructions.
- **Mobile Safari bottom URL bar handling** — the chat input area dynamically adjusts its position using the Visual Viewport API so it's never hidden behind Safari's bottom toolbar.

### Fixed
- **Mobile viewport zoom** — added `maximum-scale=1.0` to the viewport meta tag so iOS doesn't zoom the page on load.
- **Horizontal overflow on mobile** — long code blocks, tables, and unbroken strings no longer push content outside the display frame. Added `overflow-x: hidden` containment at the document and app level, aggressive word-breaking in message bubbles, and scrollable overflow for tables and pre blocks.
- **Header wrapping on narrow screens** — the title, model selector, and clear button now wrap gracefully instead of overflowing on phones.

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
[0.3.1]: https://github.com/haihengh/llm-search/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/haihengh/llm-search/compare/v0.2.8...v0.3.0
[0.2.8]: https://github.com/haihengh/llm-search/compare/v0.2.7...v0.2.8
