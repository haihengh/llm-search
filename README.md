# LLM Search

**Give your local LLM internet search — no API keys, no rate limits, no signups.**

One `docker compose up` bundles a self-hosted search engine (SearXNG) with middleware that wires it into LM Studio's tool-calling. Your LLM stays local, your search stays private, and nothing depends on a third-party service.

[![Docker Image](https://img.shields.io/badge/ghcr.io-haihengh%2Fllm--search-blue)](https://github.com/haihengh/llm-search/pkgs/container/llm-search)

```
┌─ Docker (one command) ──────────────────────────────┐
│                                                      │
│  ┌───────────────┐        ┌──────────────────┐      │
│  │  Middleware    │───────▶│    SearXNG       │      │
│  │  (FastAPI)    │        │  (self-hosted    │      │
│  │  :8000        │        │   metasearch)    │      │
│  └──────┬────────┘        └────────┬─────────┘      │
│         │                          │                 │
└─────────┼──────────────────────────┼─────────────────┘
          │                          │
          ▼                          ▼ (anonymized queries)
   ┌─────────────┐          ┌──────────────────┐
   │  LM Studio  │          │  Google, Bing,   │
   │  :1234      │          │  DuckDuckGo ...  │
   │  (host PC)  │          │  (the internet)  │
   └─────────────┘          └──────────────────┘
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

[Claude Code](https://docs.anthropic.com/en/docs/claude-code) can use the middleware as a custom OpenAI provider for its tool-enabled models.

**1. Start the middleware** (see LM Studio or Ollama section above).

**2. Configure Claude Code** — add to `~/.claude/settings.json`:
```json
{
  "customProviders": {
    "llm-search": {
      "baseUrl": "http://localhost:8000/v1",
      "apiKey": "not-needed"
    }
  }
}
```

**3. Use in Claude Code:**
```bash
claude --model llm-search/qwythos-9b-claude-mythos-5-1m "What is the latest Go version?"
```

Or set as default in `.claude/settings.json`:
```json
{
  "model": "llm-search/qwythos-9b-claude-mythos-5-1m"
}
```

> Claude Code sends tool definitions with each request. The middleware preserves them alongside the auto-injected `web_search` and `fetch_page` tools.

---

### Claude Desktop (MCP)

Use the MCP server to give Claude Desktop search capability:

**1. Install with MCP support:**
```bash
pip install llm-search[mcp]
```

**2. Configure Claude Desktop** — add to `claude_desktop_config.json`:
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
| `docker-compose.yml` | One command to start SearXNG + middleware |
| `Dockerfile` | Middleware container build |
| `.github/workflows/publish.yml` | Push Docker image to GHCR + Docker Hub on `v*` tags |
| `searxng/settings.yml` | SearXNG config — no changes needed |
| `src/llm_search/server.py` | FastAPI server — `/v1/chat/completions`, `/health`, `/stats` |
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

## License

MIT
