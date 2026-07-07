# 架构设计文档

## 1. 概览

### 1.1 设计理念

**零外部依赖。** 无需 API Key。无需注册。无频率限制。一条命令启动。

系统将自托管搜索引擎（SearXNG）与中间件打包在一起，中间件代理用户的聊天客户端与 LM Studio 之间的通信。SearXNG 是一个元搜索引擎 — 它**匿名**查询 Google、Bing、DuckDuckGo 等，聚合结果并返回。没有任何上游服务商知道你是谁，你永远不会遇到"免费额度耗尽"的情况。

### 1.2 架构一览

```
┌─ Docker Compose（docker compose up）──────────────────────────────┐
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                  llm-search（中间件）                     │ │
│  │                  Python/FastAPI :8000                     │ │
│  │                                                          │ │
│  │  POST /v1/chat/completions   (OpenAI 兼容 API)           │ │
│  │  POST /v1/messages           (Anthropic 兼容 API)        │ │
│  │                                                          │ │
│  │  ┌────────────────────────────────────────────────────┐  │ │
│  │  │              工具调用循环                          │  │ │
│  │  │                                                    │  │ │
│  │  │  1. 转发请求 ─────────────────▶ LM Studio :1234    │  │ │
│  │  │  2. LLM 响应 "tool_calls: [web_search]"           │  │ │
│  │  │  3. 执行搜索 ─────────────────▶ SearXNG :8080     │  │ │
│  │  │  4. 将结果反馈给 LLM                               │  │ │
│  │  │  5. LLM 生成最终答案                               │  │ │
│  │  │  6. 返回给客户端                                   │  │ │
│  │  └────────────────────────────────────────────────────┘  │ │
│  │                                                          │ │
│  │  ┌──────────┐  ┌───────────┐  ┌────────────────────┐    │ │
│  │  │  缓存    │  │  速率     │  │  搜索提供商        │    │ │
│  │  │  (TTL)   │  │  限制器   │  │  适配器            │    │ │
│  │  └──────────┘  └───────────┘  └─────────┬──────────┘    │ │
│  └──────────────────────────────────────────┼───────────────┘ │
│                                              │                  │
│  ┌───────────────────────────────────────────┼──────────────┐ │
│  │               SearXNG（元搜索引擎）       │              │ │
│  │               :8080（内部，无需认证）     │              │ │
│  │                                           │              │ │
│  │  匿名查询 Google、Bing、DDG 等 ───────────┘              │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
└────────────────────────────────────────────────────────────────┘
         │                                          │
         ▼                                          ▼
   ┌──────────┐                          ┌─────────────────┐
   │ LM Studio│                          │  互联网          │
   │ (主机)   │                          │  (搜索引擎)      │
   │ GPU-     │                          │  由 SearXNG    │
   │ 加速     │                          │  查询           │
   └──────────┘                          └─────────────────┘
```

---

## 2. 组件详解

### 2.1 SearXNG — 自托管搜索引擎

**为什么选择 SearXNG？**

| 特性 | SearXNG | Brave API | Google CSE |
|----------|---------|-----------|------------|
| 需要 API Key | ❌ 否 | ✅ 是 | ✅ 是 |
| 使用限制 | ❌ 无 | 2,000/月 | 100/天 |
| 隐私 | ✅ 匿名 | ⚠️ Brave 可见查询 | ❌ Google 追踪 |
| 部署 | `docker compose up` | 注册 → API Key | GCP 项目 → CSE → Key |
| 搜索质量 | ★★★★ | ★★★★ | ★★★★★ |
| 上游引擎 | 80+ 引擎可配置 | Brave 自有索引 | Google 索引 |

SearXNG 通过爬取搜索引擎结果页面工作 — 无需官方 API 访问。它在引擎之间轮换以避免频率限制，去除追踪，返回干净的结构化结果。

**配置方式：**

仓库中包含最简 `settings.yml`：

```yaml
use_default_settings: true
search:
  safe_search: 0
  formats:
    - json
server:
  bind_address: "0.0.0.0"
  port: 8080
  limiter: false
ui:
  enabled: false
```

SearXNG 容器是**内部专用的** — 不暴露到主机网络。只有中间件可以访问它。这意味着：
- 无外部访问 SearXNG
- 无需设置认证
- 速率限制无意义（中间件已有速率限制）

### 2.2 中间件 — 工具调用拦截器

一个 FastAPI 服务器，提供 OpenAI 兼容的 `/v1/chat/completions` 和 Anthropic 兼容的 `/v1/messages` 端点。这是用户**唯一**需要连接的服务。

**核心循环（伪代码）：**

```python
async def chat_completions(request):
    conversation = request.messages
    tools = inject_tools(request.tools)  # 自动注入 web_search + fetch_page
    iterations = 0

    while iterations < MAX_LOOP_ITERATIONS:
        response = await lm_studio.chat(
            messages=conversation,
            tools=tools,
            stream=False,
        )

        if response.has_tool_calls():
            for tool_call in response.tool_calls:
                if tool_call.name == "web_search":
                    results = await search_provider.search(tool_call.args.query)
                    conversation.append({
                        role: "tool",
                        content: format_search_results(results),
                    })
                elif tool_call.name == "fetch_page":
                    text = await fetch_page_text(tool_call.args.url)
                    conversation.append({
                        role: "tool",
                        content: text,
                    })
            iterations += 1
            continue  # 循环 — 将结果反馈给 LLM

        # 无工具调用 — 这是最终答案
        return response

    raise MaxIterationsExceeded()
```

**自动注入的工具定义：**

中间件始终向工具列表添加 `web_search` 和 `fetch_page`。客户端无需定义它们（但可以覆盖定义）。这意味着不包含 `tools` 字段的裸请求也能获得搜索能力。

**`web_search` 工具：**
```json
{
  "type": "function",
  "function": {
    "name": "web_search",
    "description": "搜索互联网获取最新信息。当你需要超出训练截止日期的知识时使用。",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {"type": "string", "description": "搜索查询"},
        "num_results": {"type": "integer", "description": "返回结果数 (1-10)", "default": 5}
      },
      "required": ["query"]
    }
  }
}
```

**`fetch_page` 工具：**
```json
{
  "type": "function",
  "function": {
    "name": "fetch_page",
    "description": "通过 URL 获取网页的全文内容。在 web_search 后使用以详细阅读特定页面。",
    "parameters": {
      "type": "object",
      "properties": {
        "url": {"type": "string", "description": "要获取的页面完整 URL"}
      },
      "required": ["url"]
    }
  }
}
```

### 2.3 Anthropic 适配器

`/v1/messages` 端点接受 Anthropic Messages API 格式的请求，内部转换为 OpenAI 格式，处理完毕后将响应转回 Anthropic 格式。这使得 Claude Code 和 Anthropic SDK 客户端无需更改即可使用中间件。

**请求转换：** Anthropic `system` → OpenAI system 消息，`tool_use`/`tool_result` 内容块 → OpenAI `tool_calls`/`tool` 消息
**响应转换：** OpenAI `choices[0].message.content` → Anthropic `content: [{type: "text", text: "..."}]`

### 2.4 LM Studio / Ollama（在主机上运行，不在 Docker 中）

LM Studio 是桌面应用，不是服务。它在主机上运行因为：
- 需要直接 GPU 访问（在 Docker 中尤其是在 macOS/Windows 上很困难）
- 有自己的 GUI 用于模型管理
- 暴露 `localhost:1234`，Docker 容器可通过 `host.docker.internal` 访问

中间件在 `LM_STUDIO_URL`（默认：`http://host.docker.internal:1234/v1`）连接 LM Studio。

Ollama 是替代方案，暴露 `localhost:11434` 上的 OpenAI 兼容 API。

---

## 3. 部署模式

### 3.1 主要方式：Docker Compose

```
docker compose up -d
```

启动两个容器：

| 容器 | 镜像 | 端口 | 备注 |
|-----------|-------|------|-------|
| `searxng` | `searxng/searxng:latest` | 8080 (内部) | 生产级元搜索引擎 |
| `llm-search` | `ghcr.io/haihengh/llm-search:latest` 或本地构建 | 8000 | 中间件 |

### 3.2 备选方式：pip 包

对于希望原生运行的用户：

```bash
pip install llm-search
# 单独启动 SearXNG（或指向已有实例）
export SEARXNG_URL=http://localhost:8080
llm-search
```

### 3.3 备选方式：MCP 服务器

```bash
pip install llm-search[mcp]
llm-search --mcp
```

---

## 4. 搜索结果格式

SearXNG 返回 JSON。中间件将其规范化为 LLM 可以高效处理的格式：

```
[1] "Linux kernel 6.10 发布，带来新特性"
    https://www.phoronix.com/linux-kernel-6-10
    Linus Torvalds 宣布发布 Linux 6.10，包括改进的硬件支持、
    新的文件系统特性以及性能优化...

[2] "Linux Kernel Archives"
    https://www.kernel.org/
    最新稳定版本：6.10。主线：6.11-rc1。
    长期支持：6.6.41, 6.1.102, 5.15.166...

[3] ...
```

每条结果：`[序号] "标题"\nURL\n摘要`。紧凑、节省 token、易于 LLM 阅读。

---

## 5. 配置参考

### 5.1 环境变量

```bash
# --- 搜索提供商 ---
SEARCH_PROVIDER=searxng       # searxng（默认）| brave | serpapi
SEARXNG_URL=http://searxng:8080   # SearXNG 地址（Docker 服务名）
# SEARCH_API_KEY=...          # 仅在使用 brave/serpapi 时需要

# --- LLM 后端 ---
LM_STUDIO_URL=http://host.docker.internal:1234/v1   # LM Studio
# LM_STUDIO_URL=http://host.docker.internal:11434/v1 # Ollama
LM_STUDIO_TIMEOUT=120.0       # LLM 请求超时时间

# --- 中间件服务器 ---
MIDDLEWARE_HOST=0.0.0.0
MIDDLEWARE_PORT=8000

# --- 限制 ---
MAX_TOOL_LOOP_ITERATIONS=5
SEARCH_CACHE_TTL_SECONDS=300
RATE_LIMIT_PER_MINUTE=30
MAX_SEARCH_RESULTS=5

# --- 日志 ---
LOG_LEVEL=INFO
```

---

## 6. API 接口

### `POST /v1/chat/completions` — OpenAI 兼容

```json
{
  "model": "any-model",
  "messages": [
    {"role": "user", "content": "Python 3.14 有什么新特性？"}
  ],
  "stream": false
}
```

### `POST /v1/messages` — Anthropic 兼容

```json
{
  "model": "claude-sonnet-4-20250514",
  "max_tokens": 1024,
  "messages": [
    {"role": "user", "content": "Python 3.14 有什么新特性？"}
  ]
}
```

### `GET /health`

```json
{
  "status": "ok",
  "lm_studio_url": "http://host.docker.internal:1234/v1",
  "search_provider": "searxng (http://searxng:8080)",
  "searxng_ok": true,
  "lm_studio_ok": true,
  "cache_hit_rate": 0.0,
  "total_searches": 42,
  "uptime_seconds": 3600.0
}
```

### `GET /stats`

```json
{
  "total_requests": 89,
  "total_searches": 142,
  "cache_hits": 48,
  "cache_misses": 94,
  "cache_hit_rate": 0.338,
  "rate_limits_hit": 0
}
```

---

## 7. 错误处理

| 场景 | 行为 |
|----------|----------|
| LM Studio 不可访问 | 502 — `{"error": "LM Studio not reachable at http://..."}` |
| SearXNG 不可访问 | 502 — `{"error": "Search engine not available"}` |
| SearXNG 无结果 | 空结果传递给 LLM — 自然地告知用户 |
| 工具循环超限 | 200 — 返回部分响应，`finish_reason: "tool_loop_max"` |
| 客户端发送格式错误的工具定义 | 400 — 验证错误 |
| 触发频率限制 | 429 — `{"error": "Too many requests"}` |

---

## 8. 设计决策记录

| 决策 | 选择 | 舍弃 | 原因 |
|----------|--------|----------|-----|
| 搜索引擎 | **SearXNG**（自托管） | Brave API, Google CSE | 零 API Key、无频率限制、一键部署 |
| 部署方式 | **Docker Compose** | 裸 Python、Kubernetes | 一条命令、跨平台、隔离 SearXNG |
| LLM 位置 | **主机** | Docker 容器 | GPU 直通在 Docker 中很困难；LM Studio 是桌面应用 |
| 工具注入 | **自动注入 web_search + fetch_page** | 要求客户端定义 | 用户更简单 — 基本请求无需工具定义 |
| SearXNG 暴露 | **仅内部**（无主机端口） | 暴露到主机端口 | 安全 — 只有中间件需要访问 SearXNG |
| 流式输出 | **最终回答流式传输** | 全流程流式 | 工具调用循环本身非流式；仅流式输出最终回答 |
| 双 API 支持 | **OpenAI + Anthropic** | 仅 OpenAI | Claude Code 和 Anthropic SDK 原生使用 Anthropic 格式 |

---

## 9. 项目结构

```
llm-search/
├── README.md
├── README_zh.md
├── docs/
│   ├── architecture.md          # 架构设计文档（英文）
│   ├── architecture_zh.md       # 架构设计文档（中文）
│   ├── requirements.md          # 环境要求（英文）
│   └── requirements_zh.md       # 环境要求（中文）
├── docker-compose.yml           # 一键启动
├── Dockerfile                   # 中间件容器
├── .dockerignore
├── .github/workflows/
│   └── publish.yml              # 发布 Docker 镜像到 GHCR + Docker Hub
├── .env.example                 # 可选配置
├── searxng/
│   ├── settings.yml             # SearXNG 配置
│   └── limiter.toml             # SearXNG 速率限制（已禁用）
├── pyproject.toml
├── requirements.txt
├── src/
│   └── llm_search/
│       ├── __init__.py
│       ├── __main__.py          # 入口点
│       ├── config.py            # 环境变量配置
│       ├── server.py            # FastAPI 应用 + 路由
│       ├── tool_loop.py         # 工具调用拦截循环
│       ├── tool_registry.py     # 可用工具定义
│       ├── anthropic_adapter.py # Anthropic ↔ OpenAI 适配
│       ├── mcp_server.py        # MCP 服务器
│       ├── fetch_page.py        # 网页抓取 + 文本提取
│       ├── cache.py             # TTL 缓存
│       ├── rate_limiter.py      # 令牌桶速率限制
│       └── search/
│           ├── __init__.py
│           ├── base.py          # 抽象 SearchProvider
│           ├── searxng.py       # SearXNG 适配器（默认）
│           ├── brave.py         # Brave Search 适配器（可选）
│           └── serpapi.py       # SerpAPI 适配器（可选）
└── tests/
    ├── test_tool_loop.py
    ├── test_search_providers.py
    └── test_cache.py
```
