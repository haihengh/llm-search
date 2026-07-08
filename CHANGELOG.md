# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
