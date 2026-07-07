# Requirements & Dependencies

## Overview

The system has two parts:
1. **Docker side** (SearXNG + middleware) — runs via `docker compose up`
2. **Host side** (LM Studio) — runs natively for GPU access

---

## Prerequisites

### 1. Docker Desktop (or Docker Engine + Compose)

- **macOS**: [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- **Windows**: [Docker Desktop](https://www.docker.com/products/docker-desktop/) (WSL2 backend recommended)
- **Linux**: Docker Engine + Compose plugin (`apt install docker.io docker-compose-v2`)

Verify:
```bash
docker --version         # 24.0+
docker compose version   # 2.0+
```

### 2. LM Studio (on host, not in Docker)

- **Version**: 0.3.0 or later (needed for reliable tool/function calling)
- **Download**: https://lmstudio.ai/
- **Setup**: Download and load a **tool-calling-capable model** (see below)
- **Verify**: LM Studio should be running on `http://localhost:1234`

### 3. A Compatible Model

The model MUST support function/tool calling in OpenAI format. Not all models do — here are our actual test results:

**✅ Works — end-to-end tested with this middleware:**

| Model | Size | Notes |
|-------|------|-------|
| `qwythos-9b-claude-mythos-5-1m` | 9B | Claude-distilled, fast & reliable |
| `qwen3.6-27b-claude-mythos-distilled-mtp` | 27B | Claude-distilled, detailed answers |
| `qwopus3.6-27b-v2-mtp` | 27B | Opus-distilled, thorough answers |
| `qwen3.6-27b-uncensored-heretic-v2-native-mtp-preserved` | 27B | "Native" variant preserves tool-calling |
| `gemma-4-31b-it-qat` | 31B | Google Gemma IT, solid tool use |

**❌ Does NOT work — loops without answering:**

| Model | Size | Reason |
|-------|------|--------|
| `qwen3.6-27b` | 27B | No OpenAI function-calling support |
| `qwen3.6-27b@q4_k_m` | 27B Q4 | No tool-calling (quantization may strip it) |
| `qwen3.6-35b-a3b` | 35B | No tool-calling |

**⏱️ Untested (timed out — likely works but model was too slow):**

| Model | Size |
|-------|------|
| `qwen3.6-27b-mtp` | 31B |

**Pattern**: Claude/Opus-distilled Qwen models and Google Gemma IT models handle tool-calling reliably. Base Qwen models do not — they lack the OpenAI function-calling training. Look for "mythos", "opus", "claude-distilled", or "native" in the model name for Qwen variants. For Gemma, use the "it" (instruction-tuned) variants.

**Recommendation**: `qwythos-9b-claude-mythos-5-1m` is the best tradeoff — small, fast, and reliable tool-calling.

---

## What You DON'T Need

| Thing | Why you don't need it |
|-------|----------------------|
| ❌ **OpenAI API key** | LLM runs locally in LM Studio |
| ❌ **Brave/Google/SerpAPI key** | SearXNG searches anonymously — no API key needed |
| ❌ **GPU** | LM Studio can CPU-run (slow) but GPU is highly recommended |
| ❌ **Cloud account** | Everything runs on your machine |
| ❌ **Python installed** | Middleware runs in Docker |
| ❌ **Node.js / npm** | No JavaScript in this stack |
| ❌ **Vector database** | Not doing RAG — this is live search |

---

## Hardware Recommendations

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| RAM | 8 GB | 16 GB+ | 8 GB model + 1-2 GB for Docker |
| GPU VRAM | 4 GB (3B model) | 8 GB+ (7-8B model) | Not needed if CPU-only (but slow) |
| Disk | 5 GB free | 15 GB free | Models are 2-7 GB each |
| CPU | x86_64 or Apple Silicon | Apple M1+ or modern x86_64 | |

---

## Network Architecture

```
┌─────────────────────────────────────────────────┐
│                   YOUR MACHINE                   │
│                                                  │
│  Docker (Linux VM or native)                     │
│  ┌───────────────────────────────────────────┐  │
│  │  searxng        llm-search                │  │
│  │  :8080 ───────▶ :8000                     │  │
│  │  (internal)     (host port 8000)          │  │
│  └────────────────┬──────────────────────────┘  │
│                   │                              │
│          host.docker.internal:1234               │
│                   │                              │
│  ┌────────────────▼──────────────────────────┐  │
│  │           LM Studio (native app)           │  │
│  │           :1234 (localhost only)           │  │
│  └───────────────────────────────────────────┘  │
│                                                  │
│  Chat clients connect to localhost:8000          │
└─────────────────────────────────────────────────┘
```

**Key points:**
- `llm-search` reaches LM Studio via `host.docker.internal:1234` (Docker's magic hostname for the host machine)
- SearXNG is **not** exposed to the host — only `llm-search` can talk to it
- Clients (chat UIs, curl, apps) talk to `localhost:8000`
- LM Studio only listens on localhost (no security risk)

---

## Docker Compose Quick Reference

```bash
# Start
docker compose up -d

# View logs
docker compose logs -f

# Check status
docker compose ps

# Restart after config changes
docker compose restart

# Stop
docker compose down

# Completely reset (clear cache, start fresh)
docker compose down -v
```

---

## Troubleshooting Common Issues

### "LM Studio not reachable"

LM Studio isn't running, or isn't on port 1234.
```bash
# Check if LM Studio is listening
curl http://localhost:1234/v1/models
```
If it fails, start LM Studio and load a model.

### "Search engine not available"

SearXNG container isn't healthy.
```bash
# Check SearXNG directly
docker compose exec searxng curl http://localhost:8080/search?q=test&format=json
```

### "Tool call loop exceeded"

The LLM keeps searching without producing an answer. This can happen with weaker models. Try:
- Increase `MAX_TOOL_LOOP_ITERATIONS=8` in `.env`
- Use a stronger model (Llama 3.1 8B instead of 3B)
- Simplify your prompt

### "Docker can't reach LM Studio on host.docker.internal"

This hostname works on Docker Desktop (macOS/Windows). On Linux, you may need:
```bash
# In docker-compose.yml, change:
extra_hosts:
  - "host.docker.internal:host-gateway"
# Then use: LM_STUDIO_URL=http://172.17.0.1:1234/v1
```
Or set `LM_STUDIO_URL=http://172.17.0.1:1234/v1` (Docker's default gateway to host on Linux).

### Rate limited by upstream search engines (rare)

SearXNG queries Google, Bing, DuckDuckGo, etc. Each engine has its own rate limits. Under rapid-fire use (e.g., testing many searches in quick succession), some engines may get temporarily suspended:

- **DuckDuckGo**: common CAPTCHA during testing — resolves on its own within minutes
- **Google**: may suspend for 180s under heavy load — self-recovers
- **Brave**: same 180s suspension pattern

SearXNG handles this automatically: suspended engines are skipped and other engines fill in. Results may be fewer during a suspension window, but search still works. Normal single-user usage rarely triggers this — we only hit it during batch testing.

If it happens persistently:
- Wait 3-5 minutes for suspensions to expire
- The middleware health check (every 30s) uses a lightweight `/healthz` endpoint that doesn't count toward search engine quotas
- Consider adding a search API key as backup: `SEARCH_PROVIDER=brave` with `SEARCH_API_KEY=...`
- To disable problematic engines, override in `searxng/settings.yml`:
  ```yaml
  engines:
    - name: duckduckgo
      disabled: true
  ```
