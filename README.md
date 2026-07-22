# LLM Search

[English](./README.md) | [中文](./README_zh.md)

**Give your local LLM internet search — no API keys, no rate limits, no signups.**

One `docker compose up` bundles a self-hosted search engine (SearXNG) with middleware that wires it into LM Studio's tool-calling. Your LLM stays local, your search stays private, and nothing depends on a third-party service.

[![Docker Image](https://img.shields.io/badge/ghcr.io-haihengh%2Fllm--search-blue)](https://github.com/haihengh/llm-search/pkgs/container/llm-search)

```
┌─ Docker (one command) ───────────────────────────────────────────┐
│                                                                   │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐   │
│  │  Chat Client │─────▶│  Middleware  │─────▶│   SearXNG    │   │
│  │  (Web UI)    │      │  (FastAPI)   │      │  (self-hosted │   │
│  │  :8080       │      │  :8000       │      │   metasearch) │   │
│  └──────────────┘      └──────┬───────┘      └──────┬───────┘   │
│                               │                      │            │
└───────────────────────────────┼──────────────────────┼────────────┘
                                │                      │
                                ▼                      ▼ (anonymized queries)
                         ┌─────────────┐      ┌──────────────────┐
                         │  LM Studio  │      │  Google, Bing,   │
                         │  :1234      │      │  DuckDuckGo ...  │
                         │  (host PC)  │      │  (the internet)  │
                         └─────────────┘      └──────────────────┘
```

## How it works

1. Your chat client sends a request to `localhost:8000` (the middleware)
2. The middleware forwards it to LM Studio with `web_search` and `fetch_page` tools auto-injected
3. When the LLM decides to search, the middleware intercepts the tool call
4. Search goes to **SearXNG** (running in Docker, no API key needed)
5. SearXNG queries Google/Bing/DuckDuckGo anonymously and returns results
6. The LLM can then `fetch_page` on any result URL to read full page content
7. Results go back to the LLM, which crafts the final answer
8. Your client gets the answer — it only made one request

## What you need

| Thing | Why |
|-------|-----|
| [Docker Desktop](https://www.docker.com/products/docker-desktop/) | Runs the search engine + middleware (one-time install) |
| [LM Studio](https://lmstudio.ai/) 0.3+ or [Ollama](https://ollama.com/) | Hosts the LLM on your GPU (see [Client Setup](#client-setup)) |
| A tool-calling model | See [compatibility table](./docs/requirements.md#3-a-compatible-model) — Claude/Opus-distilled or Gemma IT models recommended |
| **That's it.** No API keys. No accounts. |

## Quick Start

### Docker (recommended)

```bash
# 1. Clone
git clone https://github.com/haihengh/llm-search
cd llm-search

# 2. Make sure LM Studio is running with a model loaded on port 1234

# 3. Start everything
docker compose up -d

# 4. Use it
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwythos-9b-claude-mythos-5-1m",
    "messages": [{"role": "user", "content": "What is the latest Linux kernel version?"}]
  }'
```

### Prebuilt image (no build step)

```bash
# Download docker-compose.yml and searxng/ config, then:
# Replace "build: ." with "image: ghcr.io/haihengh/llm-search:latest"
docker compose up -d
```

### pip (native, no Docker for middleware)

```bash
pip install llm-search

# Start SearXNG separately, then:
export SEARXNG_URL=http://localhost:8080
llm-search
```

## Built-in Chat Client

The repo includes a lightweight web chat UI that launches alongside the middleware — no extra setup needed.

```
http://localhost:8080
```

**Features:**
- **Streaming chat** — responses appear token-by-token
- **Image upload** — paste from clipboard (`Ctrl+V`) or pick from file dialog; sent as OpenAI vision-format content
- **File upload** — text files (.txt, .py, .md, .json, etc.) are read and included in the message
- **Model selector** — auto-populated from LM Studio's `/v1/models`
- **Dark mode** — automatic via system preference
- **Markdown rendering** — code blocks, tables, lists with syntax highlighting

The chat client proxies all API calls to the middleware internally, so the browser only talks to one origin. It's a separate Docker service (`chat-client` in `docker-compose.yml`) — disable it by commenting out the service block if you only need the API.

## Streaming

Set `"stream": true` for token-by-token SSE streaming:

```bash
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwythos-9b-claude-mythos-5-1m",
    "messages": [{"role": "user", "content": "What is the latest Go version?"}],
    "stream": true
  }'
```

## Tools

The middleware auto-injects two tools — no client configuration needed:

| Tool | What it does |
|------|-------------|
| `web_search` | Search the internet for current information |
| `fetch_page` | Fetch and read the full text of a web page |

## Client Setup

The middleware sits between your chat client and your LLM backend. You point your **client** at `localhost:8000` and configure which LLM backend to forward to.

---

### LM Studio (recommended)

LM Studio runs models on your GPU and exposes an OpenAI-compatible API on port 1234.

**1. Load a model** — See the [compatibility table](./docs/requirements.md#3-a-compatible-model). Recommended: `qwythos-9b-claude-mythos-5-1m`.

**2. Start the server** — Developer tab → load model → start on port 1234. Verify:
```bash
curl http://localhost:1234/v1/models
```

**3. Start the middleware:**
```bash
docker compose up -d
```

**4. Connect your client** — point at `http://localhost:8000/v1`, model = model ID in LM Studio, API key = anything.

---

### Cross-Machine Setup (LLM on a different computer)

You can run the middleware on one machine and the LLM on another. This is useful when your GPU is in a desktop but you want to access it from a laptop.

**On the LLM machine (with the GPU):**
```bash
# LM Studio: Developer tab → start server on 0.0.0.0:1234
# Or Ollama: set OLLAMA_HOST=0.0.0.0:11434
```

**On the middleware machine:**
```bash
export LM_STUDIO_URL=http://192.168.1.50:1234/v1  # replace with your LLM machine's IP
docker compose up -d
```

**On any client machine:**
```
API Base URL: http://MIDDLEWARE_IP:8000/v1
# Clients can also call GET /v1/models to discover available models
```

> The middleware proxies `/v1/models` to the LLM backend, so chat clients see the full model list no matter where the LLM is running.

### Ollama

[Ollama](https://ollama.com/) is a lightweight alternative to LM Studio. It runs models and exposes an OpenAI-compatible API on port 11434.

**1. Install Ollama and pull a model:**
```bash
# Install from https://ollama.com, then:
ollama pull qwen3.6:27b
```

**2. Start the middleware pointing at Ollama:**
```bash
LM_STUDIO_URL=http://host.docker.internal:11434/v1 docker compose up -d
```

Or if running the middleware via pip:
```bash
export LM_STUDIO_URL=http://localhost:11434/v1
llm-search
```

**3. Connect your client** — same as LM Studio: `http://localhost:8000/v1`, model = `qwen3.6:27b`.

> **Note:** Ollama models may have weaker tool-calling than the Claude-distilled models on LM Studio. Look for models with function-calling support, or use the `qwythos-9b` family if available as GGUF.

---

### Claude Code (CLI)

[Claude Code](https://docs.anthropic.com/en/docs/claude-code) connects natively via the Anthropic Messages API. The middleware exposes `/v1/messages` which accepts Anthropic-format requests directly — no translation layer needed.

**1. Start the middleware** (see LM Studio or Ollama section above).

**2. Set environment variables:**
```bash
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_AUTH_TOKEN=not-needed
export CLAUDE_CODE_ATTRIBUTION_HEADER=0
```

**3. Use Claude Code normally — it now searches the web:**
```bash
claude "What is the latest Go version?"
claude "Search for the current Bitcoin price and tell me the trend"
claude "Read the Python 3.14 release notes and summarize new features"
```

That's it. Claude Code sends Anthropic-format requests → middleware translates internally → runs tool loop → returns Anthropic-format responses with `web_search` and `fetch_page` auto-injected.

> The middleware only sends `web_search` + `fetch_page` to the LLM — Claude Code's own tools (Bash, Read, Write, etc.) are stripped and handled by Claude Code itself. This prevents small local models from getting confused by too many tools. If the LLM can't converge on an answer, the middleware returns accumulated search results as a fallback instead of an error.

---

### Claude Desktop (MCP)

Use the MCP server to give Claude Desktop search capability:

**1. Install with MCP support:**
```bash
pip install llm-search[mcp]
```

**2. Configure Claude Desktop** — add to `claude_desktop_config.json` (open it via Claude Desktop → Settings → Developer → Edit Config, or edit directly):

| OS | File location |
|----|---------------|
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Linux | `~/.config/Claude/claude_desktop_config.json` |

```json
{
  "mcpServers": {
    "llm-search": {
      "command": "python",
      "args": ["-m", "llm_search", "--mcp"],
      "env": {
        "SEARXNG_URL": "http://localhost:8080",
        "LM_STUDIO_URL": "http://localhost:1234/v1"
      }
    }
  }
}
```

**3. Restart Claude Desktop** — `web_search` and `fetch_page` tools will appear in the tool list.

> The MCP server uses stdio transport. It needs SearXNG running (Docker or standalone) and optionally LM Studio for models that use it. If you only need search tools in Claude Desktop, you don't need LM Studio — just SearXNG.

---

### VS Code Copilot Chat (BYOK)

GitHub Copilot Chat can use the middleware as a custom model provider. Requires VS Code 1.122+ (currently Insiders) and any Copilot subscription (the free plan works).

**Option A — UI:** Command Palette → `Chat: Manage Language Models` → **Add Models** → **Custom Endpoint**, then enter `http://localhost:8000/v1/chat/completions`, API type "Chat Completions", any API key.

**Option B — config file:** create `chatLanguageModels.json` in the VS Code user folder:

| OS | File location |
|----|---------------|
| Windows | `%APPDATA%\Code\User\chatLanguageModels.json` |
| macOS | `~/Library/Application Support/Code/User/chatLanguageModels.json` |
| Linux | `~/.config/Code/User/chatLanguageModels.json` |

> Using VS Code Insiders? Replace `Code` with `Code - Insiders` in the path.

```json
[
  {
    "name": "llm-search",
    "vendor": "customendpoint",
    "apiKey": "${input:chat.lm.secret.llmsearch}",
    "apiType": "chat-completions",
    "models": [{
      "id": "local-model",
      "name": "LM Studio + Search",
      "url": "http://localhost:8000/v1/chat/completions",
      "apiType": "chat-completions",
      "toolCalling": true,
      "maxInputTokens": 32768,
      "maxOutputTokens": 8192
    }]
  }
]
```

The file is a top-level **array** — one entry per provider group. After editing it, run `Developer: Reload Window`, then tick the model once in the chat model picker's Manage Models list. If the model still doesn't appear, use Option A once and let VS Code generate the file, then compare.

Set `"id"` to the model ID loaded in LM Studio (or leave `local-model` — LM Studio falls back to the loaded model). "LM Studio + Search" then appears in the Copilot Chat model picker.

**Sizing the token limits** — `maxInputTokens + maxOutputTokens` must fit inside the model's context window, minus headroom for the search results and fetched pages the middleware appends server-side (Copilot doesn't see those, so it can't budget for them). For a 100k-context model, `maxInputTokens: 80000` + `maxOutputTokens: 8192` leaves ~12k of headroom for tool results. Also make sure LM Studio's context length for the loaded model is actually set that high — it defaults to a much smaller value regardless of what the model supports. If you overshoot, the middleware returns a context-overflow error rather than silently truncating.

> BYOK models power **chat only** — inline code completions stay on GitHub's models. Copilot's **agent mode** is not supported: the middleware strips client tools before calling the LLM (see the Claude Code note above), so use ask/chat mode.

---

### OpenAI Codex Desktop App

The [Codex desktop app](https://openai.com/codex) (Windows / macOS) can use the middleware as a custom model provider. Requires a ChatGPT account (free tier works) and Codex version that supports custom providers.

**Step 1 — Set the environment variable** (Codex requires an `env_key` even for local providers):

| OS | How to set |
|----|------------|
| **Windows** | `[System.Environment]::SetEnvironmentVariable('LLM_SEARCH_API_KEY', 'no-key-needed', 'User')` in PowerShell, then log out and back in |
| **macOS** | `launchctl setenv LLM_SEARCH_API_KEY no-key-needed` and restart Codex |

**Step 2 — Edit `~/.codex/config.toml`:**

| OS | File location |
|----|---------------|
| **Windows** | `C:\Users\<username>\.codex\config.toml` |
| **macOS** | `~/.codex/config.toml` |

```toml
model = "qwythos-9b-claude-mythos-5-1m"
model_provider = "llm-search"
model_context_window = 131072
model_auto_compact_token_limit = 110000
model_max_output_tokens = 16384

[model_providers.llm-search]
name = "LM Studio + Search"
base_url = "http://localhost:8000/v1"
env_key = "LLM_SEARCH_API_KEY"
wire_api = "responses"
```

Set `model` to the model ID loaded in LM Studio (must match a model in the `/v1/models` list). **Restart Codex** after editing — the model appears in the model picker (may show as "Custom" due to a known Codex Desktop UI issue).

**Sizing** — `model_context_window` should match the loaded model's context size. `model_auto_compact_token_limit` triggers auto-compaction at ~85% of context. Leave ~12k headroom for search results the middleware appends server-side.

> **Chat/ask mode only** — agent mode is not supported. The middleware strips client tools (bash, read, write, etc.) before calling the LLM; only `web_search` and `fetch_page` are passed through. Use ask/chat mode for search-augmented answers.

---

### Cursor / Continue.dev / Windsurf

These VS Code AI extensions support custom OpenAI-compatible providers:

| Client | Config location | Setting |
|--------|----------------|---------|
| **Cursor** | Settings → Models → Add Model | Base URL: `http://localhost:8000/v1`, API Key: `not-needed` |
| **Continue.dev** | `~/.continue/config.json` | `"apiBase": "http://localhost:8000/v1"` under the model entry |
| **Windsurf** | Settings → AI Provider | Provider: OpenAI, Base URL: `http://localhost:8000/v1` |

For Continue.dev, a full model entry looks like:
```json
{
  "models": [{
    "title": "LLM Search",
    "provider": "openai",
    "model": "qwythos-9b-claude-mythos-5-1m",
    "apiBase": "http://localhost:8000/v1",
    "apiKey": "not-needed"
  }]
}
```

---

### Open WebUI

```bash
docker run -d --network host \
  -e OPENAI_API_BASE_URL=http://localhost:8000/v1 \
  -e OPENAI_API_KEY=not-needed \
  ghcr.io/open-webui/open-webui:main
```

Then add a new OpenAI connection in Open WebUI settings pointing at `http://localhost:8000/v1`.

---

### Direct curl / API

```bash
# Non-streaming
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "local-model", "messages": [{"role": "user", "content": "What is the latest Go version?"}]}'

# Streaming
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "local-model", "messages": [{"role": "user", "content": "Latest Go version?"}], "stream": true}'
```

## What's in the box

| What | Purpose |
|------|---------|
| `docker-compose.yml` | One command to start SearXNG + middleware + chat UI |
| `Dockerfile` | Middleware container build |
| `chat-client/Dockerfile` | Chat UI container build |
| `chat-client/server.py` | FastAPI proxy — serves UI, proxies API calls |
| `chat-client/static/` | Chat UI — HTML, CSS, vanilla JS |
| `.github/workflows/publish.yml` | Push Docker image to GHCR + Docker Hub on `v*` tags |
| `searxng/settings.yml` | SearXNG config — no changes needed |
| `src/llm_search/server.py` | FastAPI server — `/v1/chat/completions`, `/v1/messages`, `/health`, `/stats` |
| `src/llm_search/tool_loop.py` | Tool-call intercept loop (non-streaming + streaming) |
| `src/llm_search/tool_registry.py` | `web_search` + `fetch_page` tool definitions |
| `src/llm_search/mcp_server.py` | MCP server — expose tools over stdio |
| `src/llm_search/fetch_page.py` | URL fetcher with HTML-to-text extraction |
| `.env.example` | Optional overrides

## Configuration (optional)

Everything works out of the box. If you want to tweak:

```bash
# Change LM Studio address (if not on default port)
LM_STUDIO_URL=http://192.168.1.50:1234/v1 docker compose up

# Use Brave Search instead of SearXNG
cp .env.example .env
# Edit .env: SEARCH_PROVIDER=brave, add SEARCH_API_KEY=...
```

All configuration via environment variables — see `.env.example` for the full list.

## Release Process

Releases follow [Semantic Versioning](https://semver.org/) with a [Keep a Changelog](https://keepachangelog.com/) format.

```bash
# 1. Add changes under [Unreleased] in CHANGELOG.md
# 2. Run the release script:
./scripts/release.sh 0.1.3

# This automatically:
#   - Updates CHANGELOG.md with the version header
#   - Commits, tags, and pushes
#   - Triggers .github/workflows/release.yml which:
#     → Creates the GitHub Release with auto-generated notes
#     → Builds + pushes Docker images to GHCR + Docker Hub
```

Manual release (if script unavailable):
```bash
git tag -a v0.1.3 -m "v0.1.3"
git push --tags
# The tag triggers the automated release pipeline
```

## License

MIT
