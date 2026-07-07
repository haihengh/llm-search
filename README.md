# LLM Search

**Give your local LLM internet search — no API keys, no rate limits, no signups.**

One `docker compose up` bundles a self-hosted search engine (SearXNG) with middleware that wires it into LM Studio's tool-calling. Your LLM stays local, your search stays private, and nothing depends on a third-party service.

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
2. The middleware forwards it to LM Studio with a `web_search` tool defined
3. When the LLM decides to search, the middleware intercepts the tool call
4. Search goes to **SearXNG** (running in Docker, no API key needed)
5. SearXNG queries Google/Bing/DuckDuckGo anonymously and returns results
6. Results go back to the LLM, which crafts the final answer
7. Your client gets the answer — it only made one request

## What you need

| Thing | Why |
|-------|-----|
| [Docker Desktop](https://www.docker.com/products/docker-desktop/) | Runs the search engine + middleware (one-time install) |
| [LM Studio](https://lmstudio.ai/) 0.3+ | Hosts the LLM on your GPU |
| A tool-calling model | Llama 3.1 8B, Qwen 2.5 7B, Mistral 7B — loaded in LM Studio |
| **That's it.** No API keys. No accounts. |

## Quick Start

```bash
# 1. Clone
git clone https://github.com/user/llm-search
cd llm-search

# 2. Make sure LM Studio is running with a model loaded on port 1234

# 3. Start everything
docker compose up -d

# 4. Use it — point any OpenAI-compatible client at http://localhost:8000/v1
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-model",
    "messages": [{"role": "user", "content": "What is the latest Linux kernel version?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "web_search",
        "description": "Search the internet for current information",
        "parameters": {
          "type": "object",
          "properties": {"query": {"type": "string"}},
          "required": ["query"]
        }
      }
    }]
  }'
```

## With a chat UI

The middleware speaks the OpenAI API, so any compatible frontend works:

```bash
# Open WebUI (popular choice)
docker run -d --network host \
  -e OPENAI_API_BASE_URL=http://localhost:8000/v1 \
  -e OPENAI_API_KEY=not-needed \
  ghcr.io/open-webui/open-webui:main
```

Or connect anything that supports custom OpenAI endpoints: **Chatbox**, **Continue.dev**, **Cursor**, etc.

## What's in the box

| File | Purpose |
|------|---------|
| `docker-compose.yml` | One command to start SearXNG + middleware |
| `searxng/settings.yml` | SearXNG config — no changes needed |
| `src/llm_search/` | Middleware (FastAPI) — tool-call intercept loop |
| `.env.example` | Optional overrides (SearXNG is the default) |

## Configuration (optional)

Everything works out of the box. If you want to tweak:

```bash
# Change LM Studio address (if not on default port)
LM_STUDIO_URL=http://192.168.1.50:1234/v1 docker compose up

# Use Brave Search instead of SearXNG
cp .env.example .env
# Edit .env: SEARCH_PROVIDER=brave, add SEARCH_API_KEY=...
```

## License

MIT
