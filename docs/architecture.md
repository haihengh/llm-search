# Architecture Design Document

## 1. Overview

### 1.1 Design Philosophy

**Zero external dependencies.** No API keys. No signups. No rate limits. One command to start.

The system bundles a self-hosted search engine (SearXNG) with a middleware that proxies between the user's chat client and LM Studio. SearXNG is a metasearch engine — it queries Google, Bing, DuckDuckGo, etc. **anonymously**, aggregates results, and returns them. No single upstream provider knows who you are, and you never hit a "free tier exhausted" wall.

### 1.2 Architecture at a Glance

```
┌─ Docker Compose (docker compose up) ───────────────────────────────┐
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │               chat-client (web UI)                          │   │
│  │               Python/FastAPI :8080                          │   │
│  │                                                             │   │
│  │  Serves static HTML/CSS/JS + proxies API calls ──────┐      │   │
│  └──────────────────────────────────────────────────────┼──────┘   │
│                                                          │          │
│  ┌──────────────────────────────────────────────────────┼──────┐   │
│  │                  llm-search (middleware)              │      │   │
│  │                  Python/FastAPI :8000                 │      │   │
│  │                                                       │      │   │
│  │  POST /v1/chat/completions   (OpenAI-compatible API) ◀┘      │   │
│  │  POST /v1/messages           (Anthropic Messages API)        │   │
│  │  POST /v1/responses          (OpenAI Responses API)          │   │
│  │                                                             │   │
│  │  ┌────────────────────────────────────────────────────┐    │   │
│  │  │              Tool Call Loop                        │    │   │
│  │  │                                                    │    │   │
│  │  │  1. Forward request ─────────────▶ LM Studio :1234 │    │   │
│  │  │  2. LLM responds "tool_calls: [web_search]"        │    │   │
│  │  │  3. Execute search ─────────────▶ SearXNG :8080    │    │   │
│  │  │  4. Feed results back to LLM                       │    │   │
│  │  │  5. LLM writes final answer                        │    │   │
│  │  │  6. Return to client                               │    │   │
│  │  └────────────────────────────────────────────────────┘    │   │
│  │                                                             │   │
│  │  ┌──────────┐  ┌───────────┐  ┌────────────────────┐      │   │
│  │  │  Cache   │  │  Rate     │  │  Search Provider   │      │   │
│  │  │  (TTL)   │  │  Limiter  │  │  Adapter           │      │   │
│  │  └──────────┘  └───────────┘  └─────────┬──────────┘      │   │
│  └──────────────────────────────────────────┼─────────────────┘   │
│                                              │                     │
│  ┌───────────────────────────────────────────┼─────────────────┐   │
│  │               SearXNG (metasearch engine) │                 │   │
│  │               :8080 (internal, no auth)   │                 │   │
│  │                                           │                 │   │
│  │  Queries Google, Bing, DDG, etc. ─────────┘                 │   │
│  │  anonymously — no API keys ever                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
         │                                          │
         ▼                                          ▼
   ┌──────────┐                          ┌─────────────────┐
   │ LM Studio│                          │  The Internet   │
   │ (host)   │                          │  (search engines)│
   │ GPU-     │                          │  queried by     │
   │ backed   │                          │  SearXNG)       │
   └──────────┘                          └─────────────────┘
```

---

## 2. Component Breakdown

### 2.1 SearXNG — The Self-Hosted Search Engine

**Why SearXNG?**

| Property | SearXNG | Brave API | Google CSE |
|----------|---------|-----------|------------|
| Needs API key | ❌ No | ✅ Yes | ✅ Yes |
| Usage limits | ❌ None | 2,000/mo | 100/day |
| Privacy | ✅ Anonymized | ⚠️ Brave sees queries | ❌ Google tracks |
| Setup | `docker compose up` | Sign up → API key | GCP project → CSE → key |
| Search quality | ★★★★ | ★★★★ | ★★★★★ |
| Upstreams | 80+ engines configurable | Brave's own index | Google's index |

SearXNG works by scraping search engine result pages — no official API access needed. It rotates through engines to avoid rate limiting, strips tracking, and returns clean structured results.

**How we configure it:**

A minimal `settings.yml` ships with the repo:

```yaml
use_default_settings: true
general:
  instance_name: "LLM Search"
  debug: false
search:
  safe_search: 0
  autocomplete: ""
  formats:
    - html
    - json
server:
  secret_key: "change-me-in-production"
  bind_address: "0.0.0.0"
  port: 8080
  limiter: false  # No rate limiting (internal use only)
ui:
  enabled: false  # No web UI needed — JSON API only
```

The SearXNG container is **internal-only** — not exposed to the host network. Only the middleware talks to it. This means:
- No external access to SearXNG
- No need to set up authentication
- Rate limiting is pointless (the middleware already rate-limits)

### 2.2 Middleware — Tool Call Interceptor

A FastAPI server that presents an OpenAI-compatible `/v1/chat/completions` endpoint. It is the **only** thing the user connects to.

**Core loop (pseudocode):**

```python
async def chat_completions(request: ChatRequest) -> ChatResponse:
    conversation = request.messages
    # Only search tools are sent to the LLM — client tools (bash, read,
    # etc.) are stripped to avoid confusing small local models.
    tools = [WEB_SEARCH_TOOL, FETCH_PAGE_TOOL]
    iterations = 0

    while iterations < MAX_LOOP_ITERATIONS:
        # On later iterations, nudge the LLM to answer
        if iterations >= MAX_LOOP_ITERATIONS - 2 and searches_done:
            conversation.append({
                "role": "user",
                "content": "Please synthesize a final answer now."
            })

        response = await lm_studio.chat(
            messages=conversation,
            tools=tools,
            stream=False,
        )

        if response.has_tool_calls():
            for tool_call in response.tool_calls:
                if tool_call.name in TOOL_EXECUTORS:
                    results = await search_provider.search(tool_call.args.query)
                    conversation.append({
                        role: "tool",
                        tool_call_id: tool_call.id,
                        content: format_search_results(results),
                    })
                else:
                    # Unrecognised tool — pass through to client
                    return response  # stop_reason: "tool_use"

            iterations += 1
            continue  # loop — feed results back to LLM

        # No tool calls — this is the final answer
        return response

    # Max iterations — return accumulated search results as fallback
    return build_fallback_from_search_results(conversation)
```

**Tool definition injected automatically:**

The middleware always sends `web_search` and `fetch_page` to the LLM. Client-provided tools are **not** forwarded — the local model only sees search tools. This prevents small models from getting confused by tools they shouldn't call (e.g. Claude Code's Bash/Read/Write). The client handles its own tools; the middleware only handles search.

```json
{
  "type": "function",
  "function": {
    "name": "web_search",
    "description": "Search the internet for current, up-to-date information. Use this whenever you need facts, news, or knowledge beyond your training cutoff.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {"type": "string", "description": "The search query"},
        "num_results": {"type": "integer", "description": "Results to return (1-10)", "default": 5}
      },
      "required": ["query"]
    }
  }
}
```

### 2.3 LM Studio (runs on host, not in Docker)

LM Studio is a desktop application, not a service. It runs on the host machine because:
- It needs direct GPU access (problematic in Docker, especially on macOS/Windows)
- It has its own GUI for model management
- It exposes `localhost:1234` which Docker containers can reach via `host.docker.internal`

The middleware connects to LM Studio at `LM_STUDIO_URL` (default: `http://host.docker.internal:1234/v1`).

---

## 3. Deployment Model

### 3.1 Primary: Docker Compose

```
docker compose up -d
```

This starts three containers:

| Container | Image | Port | Notes |
|-----------|-------|------|-------|
| `searxng` | `searxng/searxng:latest` | 8080 (internal) | Production-grade metasearch engine |
| `llm-search` | `ghcr.io/user/llm-search:latest` or local build | 8000 | Middleware |
| `chat-client` | local build (`./chat-client`) | 8080 | Web chat UI (optional — comment out to disable) |

A `docker-compose.yml` ships in the repo. It wires:
- `chat-client` → `llm-search:8000` (internal Docker network)
- `llm-search` → `searxng:8080` (internal Docker network)
- `llm-search` → `host.docker.internal:1234` (LM Studio on host)
- SearXNG settings volume-mounted from `./searxng/settings.yml`

### 3.2 Alternative: Python package (pip)

For users who prefer running natively:

```bash
pip install llm-search
# Start SearXNG separately (or point at an existing one)
export SEARXNG_URL=http://localhost:8080
llm-search
```

### 3.3 Alternative: Single binary (future)

PyInstaller can bundle the middleware into a standalone executable. SearXNG would still need Docker. But for Windows/macOS users, a `.exe`/`.app` could simplify the middleware side.

---

## 4. Search Result Format

SearXNG returns JSON. The middleware normalizes it into a format the LLM can digest efficiently:

```
[1] "Linux kernel 6.10 released with new features"
    https://www.phoronix.com/linux-kernel-6-10
    Linus Torvalds announced the release of Linux 6.10, which includes
    improved hardware support, new filesystem features, and performance...

[2] "Linux Kernel Archives"
    https://www.kernel.org/
    The Linux Kernel Archives. Latest stable version: 6.10. Mainline: 6.11-rc1.
    Longterm: 6.6.41, 6.1.102, 5.15.166...

[3] ...
```

Each result: `[index] "title"\nURL\nsnippet`. Compact, token-efficient, easy for LLMs to read.

---

## 5. Configuration Reference

### 5.1 Environment Variables

```bash
# --- Search Provider ---
SEARCH_PROVIDER=searxng       # searxng (default) | brave | serpapi
SEARXNG_URL=http://searxng:8080   # SearXNG address (Docker service name)
# SEARCH_API_KEY=...          # Only needed if using brave/serpapi

# --- LM Studio ---
LM_STUDIO_URL=http://host.docker.internal:1234/v1

# --- Middleware ---
MIDDLEWARE_HOST=0.0.0.0
MIDDLEWARE_PORT=8000

# --- Limits ---
MAX_TOOL_LOOP_ITERATIONS=10
SEARCH_CACHE_TTL_SECONDS=300
RATE_LIMIT_PER_MINUTE=30         # Higher default since SearXNG has no API cost
MAX_SEARCH_RESULTS=5

# --- Logging ---
LOG_LEVEL=INFO
```

### 5.2 docker-compose.yml structure

```yaml
services:
  searxng:
    image: searxng/searxng:latest
    volumes:
      - ./searxng:/etc/searxng:ro
    environment:
      - SEARXNG_SETTINGS_PATH=/etc/searxng/settings.yml
    restart: unless-stopped
    # No ports exposed — internal only

  llm-search:
    build: .
    # or: image: ghcr.io/user/llm-search:latest
    ports:
      - "8000:8000"
    environment:
      - SEARCH_PROVIDER=searxng
      - SEARXNG_URL=http://searxng:8080
      - LM_STUDIO_URL=http://host.docker.internal:1234/v1
    extra_hosts:
      - "host.docker.internal:host-gateway"
    depends_on:
      - searxng
    restart: unless-stopped

  chat-client:
    build: ./chat-client
    ports:
      - "8080:8080"
    environment:
      - MIDDLEWARE_URL=http://llm-search:8000
    depends_on:
      - llm-search
    restart: unless-stopped
```

---

## 6. API Surface

### `POST /v1/chat/completions`

OpenAI-compatible chat completions. Handles the tool-call loop internally.

**Minimal request (no tools defined — middleware injects web_search):**

```json
{
  "model": "any-model",
  "messages": [
    {"role": "user", "content": "What's new in Python 3.14?"}
  ]
}
```

**With explicit tools (client tools are NOT forwarded to the LLM):**

```json
{
  "model": "any-model",
  "messages": [{"role": "user", "content": "..."}],
  "tools": [
    {"type": "function", "function": {"name": "calculator", "..."}}
  ]
}
```

The middleware sends only `web_search` + `fetch_page` to the LLM. Client tools are stripped — the client (e.g. Claude Code) handles them itself. If the LLM hallucinates a tool name not in the middleware's registry, it's passed through to the client as a `stop_reason: "tool_use"` response.

### `GET /health`

```json
{
  "status": "ok",
  "lm_studio": "connected",
  "searxng": "connected",
  "search_provider": "searxng",
  "cache_hit_rate": 0.34,
  "total_searches": 142
}
```

### `GET /stats`

```json
{
  "total_requests": 89,
  "total_searches": 142,
  "cache_hits": 48,
  "cache_misses": 94,
  "avg_loop_iterations": 1.6,
  "rate_limits_hit": 0
}
```

---

## 7. Error Handling

| Scenario | Behavior |
|----------|----------|
| LM Studio unreachable | 502 — `{"error": "LM Studio not reachable at http://..."}`  |
| SearXNG unreachable | 502 — `{"error": "Search engine not available"}` |
| SearXNG returns no results | Empty results passed to LLM — it informs the user naturally |
| Tool loop exceeds max iterations | 200 — returns accumulated search results as fallback with `finish_reason: "tool_loop_max"` |
| Client sends malformed tools | 400 — validation error |
| Rate limit hit | 429 — `{"error": "Too many requests", "retry_after": 5}` |

---

## 8. Project Structure

```
llm-search/
├── README.md
├── docs/
│   ├── architecture.md          # This document
│   └── requirements.md          # Prerequisites & dependencies
├── docker-compose.yml           # One-command startup (3 services)
├── Dockerfile                   # Middleware container
├── chat-client/                 # Built-in web chat UI
│   ├── Dockerfile               # Chat UI container
│   ├── server.py                # FastAPI proxy (serves UI + proxies API)
│   └── static/
│       ├── index.html           # Chat UI shell
│       ├── style.css            # Light/dark theme, responsive layout
│       └── app.js               # Chat logic, SSE streaming, image/file upload
├── .dockerignore
├── .env.example                 # Optional overrides
├── searxng/
│   ├── settings.yml             # SearXNG configuration
│   └── limiter.toml             # SearXNG rate limiter (disabled)
├── pyproject.toml
├── requirements.txt
├── src/
│   └── llm_search/
│       ├── __init__.py
│       ├── __main__.py          # Entry point
│       ├── config.py            # Settings from env vars
│       ├── server.py            # FastAPI app + routes
│       ├── tool_loop.py         # The tool-call intercept loop
│       ├── tool_registry.py     # Available tools & their executors
│       ├── search/
│       │   ├── __init__.py
│       │   ├── base.py          # Abstract SearchProvider
│       │   ├── searxng.py       # SearXNG adapter (DEFAULT)
│       │   ├── brave.py         # Brave Search adapter (optional)
│       │   └── serpapi.py       # SerpAPI adapter (optional)
│       ├── cache.py             # TTL cache
│       └── rate_limiter.py      # Token bucket rate limiter
├── tests/
│   ├── test_tool_loop.py
│   ├── test_search_providers.py
│   └── test_cache.py
└── tools.yaml                   # Optional custom tool definitions
```

---

## 9. Implementation Phases

| Phase | What | Effort |
|-------|------|--------|
| **Phase 1: Core** | Docker Compose, SearXNG config, FastAPI server, tool loop, SearXNG adapter, cache | Main effort |
| **Phase 2: Polish** | Rate limiting, health/stats endpoints, error handling, logging, Dockerfile | Smaller |
| **Phase 3: Optional providers** | Brave + SerpAPI adapters (for users who want them) | Small |
| **Phase 4: Streaming** | `stream: true` support — buffer tool calls, stream final response | Medium |
| **Phase 5: Extra tools** | `fetch_page` tool (read full page content behind a URL) | Small |
| **Phase 6: Distribution** | Publish Docker image to GHCR, pip package, binary builds | Small |

---

## 10. Design Decisions Log

| Decision | Chosen | Rejected | Why |
|----------|--------|----------|-----|
| Search engine | **SearXNG** (self-hosted) | Brave API, Google CSE | Zero API keys, no rate limits, one-command setup |
| Deployment | **Docker Compose** | Bare Python, Kubernetes | Single command, works everywhere, isolates SearXNG |
| LM Studio location | **Host machine** | Docker container | GPU passthrough is painful in Docker; LM Studio is a desktop app |
| Tool injection | **Auto-inject web_search** | Require client to define it | Simpler for users — no tool definition needed in basic requests |
| SearXNG exposure | **Internal only** (no host port) | Expose on host port | Security — only the middleware needs to talk to SearXNG |
| Streaming in v1 | **Deferred** | Blocking support | Tool-call loops are inherently non-streaming; stream the final response only |

---

## 11. Open Questions

1. **SearXNG engine selection**: Which search engines should be enabled by default? Google + Bing + DuckDuckGo covers 95% of use cases. Should we enable more niche engines?

2. **SearXNG instance sharing**: If multiple users on the same machine each run `docker compose up`, they'll each get their own SearXNG. Should we document how to share a single SearXNG across multiple middleware instances?

3. **Mobile/remote access**: Should we document how to expose the middleware on the LAN so mobile chat clients can use it? (Simple — just note that `0.0.0.0:8000` is already configured.)

4. **Offline mode**: What happens when SearXNG can't reach upstream search engines? Should we detect this and tell the LLM to work with what it knows?

5. **Result deduplication**: SearXNG sometimes returns similar results from different engines. Should the middleware deduplicate?
