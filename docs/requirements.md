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

The model MUST support function/tool calling in OpenAI format. Not all models do.

| Model | Size | Tool Calling | RAM | Recommended? |
|-------|------|-------------|-----|--------------|
| **Llama 3.1 8B Instruct** | ~5 GB | ✅ Reliable | 8 GB | ✅ Best all-around |
| **Qwen 2.5 7B Instruct** | ~4.5 GB | ✅ Reliable | 8 GB | ✅ Excellent tool use |
| **Mistral Nemo 12B** | ~7 GB | ✅ Reliable | 12 GB | ✅ Strong, needs more RAM |
| **Llama 3.2 3B Instruct** | ~2 GB | ⚠️ Weaker | 4 GB | ⚠️ For low-resource systems |
| **Phi-3 Mini 4K** | ~2.5 GB | ⚠️ Weaker | 4 GB | ⚠️ Hit or miss on tool calling |

**Recommendation**: Start with **Llama 3.1 8B Instruct** or **Qwen 2.5 7B Instruct**. They're the sweet spot of quality, tool-calling reliability, and hardware requirements.

**How to check if your model supports tool calling**: In LM Studio, load the model and look at the chat tab — if you can enable "Tool Use" mode, it supports function calling.

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

SearXNG queries Google/Bing/etc. Heavy usage might trigger temporary blocks from individual engines. SearXNG automatically rotates to other engines. If it happens persistently:
- Enable more engines in `searxng/settings.yml`
- Consider using Brave API as a fallback (`SEARCH_PROVIDER=brave`)
