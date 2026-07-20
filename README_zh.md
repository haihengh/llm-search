# LLM Search

**让你的本地大模型联网搜索 — 无需 API Key、无频率限制、无需注册。**

一条 `docker compose up` 命令即可部署自托管搜索引擎（SearXNG）+ 中间件，连接到 LM Studio 的工具调用功能。你的 LLM 完全本地运行，搜索完全私密，不依赖任何第三方服务。

[![Docker Image](https://img.shields.io/badge/ghcr.io-haihengh%2Fllm--search-blue)](https://github.com/haihengh/llm-search/pkgs/container/llm-search)

```
┌─ Docker（一条命令）────────────────────────────────┐
│                                                      │
│  ┌───────────────┐        ┌──────────────────┐      │
│  │  中间件        │───────▶│    SearXNG       │      │
│  │  (FastAPI)    │        │  (自托管          │      │
│  │  :8000        │        │   元搜索引擎)     │      │
│  └──────┬────────┘        └────────┬─────────┘      │
│         │                          │                 │
└─────────┼──────────────────────────┼─────────────────┘
          │                          │
          ▼                          ▼ (匿名查询)
   ┌─────────────┐          ┌──────────────────┐
   │  LM Studio  │          │  Google, Bing,   │
   │  :1234      │          │  DuckDuckGo ...  │
   │  (主机)     │          │  (互联网)        │
   └─────────────┘          └──────────────────┘
```

## 工作原理

1. 你的聊天客户端发送请求到 `localhost:8000`（中间件）
2. 中间件将请求转发到 LM Studio，并自动注入 `web_search` 和 `fetch_page` 工具
3. 当 LLM 决定搜索时，中间件拦截工具调用
4. 搜索请求发送到 **SearXNG**（Docker 中运行，无需 API Key）
5. SearXNG 匿名查询 Google/Bing/DuckDuckGo 并返回结果
6. LLM 可以进一步使用 `fetch_page` 读取任意结果 URL 的完整内容
7. 结果返回给 LLM，生成最终答案
8. 你的客户端获得答案 — 只发了一次请求

## 你需要什么

| 组件 | 用途 |
|-------|-----|
| [Docker Desktop](https://www.docker.com/products/docker-desktop/) | 运行搜索引擎 + 中间件（一次性安装） |
| [LM Studio](https://lmstudio.ai/) 0.3+ 或 [Ollama](https://ollama.com/) | 在 GPU 上运行 LLM（见[客户端配置](#客户端配置)） |
| 支持工具调用的模型 | 参考[兼容性列表](./docs/requirements_zh.md#3-兼容模型) — 推荐 Claude/Opus 蒸馏版或 Gemma IT 模型 |
| **就这些。** 无需 API Key、无需账号。 |

## 快速开始

### Docker（推荐）

```bash
# 1. 克隆仓库
git clone https://github.com/haihengh/llm-search
cd llm-search

# 2. 确保 LM Studio 已启动，模型已加载，运行在 1234 端口

# 3. 启动所有服务
docker compose up -d

# 4. 开始使用
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwythos-9b-claude-mythos-5-1m",
    "messages": [{"role": "user", "content": "Linux 最新内核版本是什么？"}]
  }'
```

### 预构建镜像（无需编译）

```bash
# 下载 docker-compose.yml 和 searxng/ 配置，然后：
# 将 "build: ." 替换为 "image: ghcr.io/haihengh/llm-search:latest"
docker compose up -d
```

### pip 安装（原生运行，无需 Docker 运行中间件）

```bash
pip install llm-search

# 单独启动 SearXNG，然后：
export SEARXNG_URL=http://localhost:8080
llm-search
```

## 流式输出

设置 `"stream": true` 启用逐 token 的 SSE 流式输出：

```bash
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwythos-9b-claude-mythos-5-1m",
    "messages": [{"role": "user", "content": "Go 最新版本是什么？"}],
    "stream": true
  }'
```

## 内置工具

中间件自动注入两个工具 — 客户端无需配置：

| 工具 | 功能 |
|------|-------------|
| `web_search` | 搜索互联网获取最新信息 |
| `fetch_page` | 获取并阅读网页的完整文本内容 |

## 客户端配置

中间件位于你的聊天客户端和 LLM 后端之间。将**客户端**指向 `localhost:8000`，并配置要转发的 LLM 后端。

---

### LM Studio（推荐）

LM Studio 在你的 GPU 上运行模型，在 1234 端口暴露 OpenAI 兼容 API。

**1. 加载模型** — 参考[兼容性列表](./docs/requirements_zh.md#3-兼容模型)。推荐：`qwythos-9b-claude-mythos-5-1m`。

**2. 启动服务** — 开发者标签页 → 加载模型 → 在 1234 端口启动。验证：
```bash
curl http://localhost:1234/v1/models
```

**3. 启动中间件：**
```bash
docker compose up -d
```

**4. 连接客户端** — 指向 `http://localhost:8000/v1`，模型 = LM Studio 中的模型 ID，API Key = 任意值。

---

### Ollama

[Ollama](https://ollama.com/) 是 LM Studio 的轻量替代方案，在 11434 端口暴露 OpenAI 兼容 API。

**1. 安装 Ollama 并拉取模型：**
```bash
# 从 https://ollama.com 安装，然后：
ollama pull qwen3.6:27b
```

**2. 配置中间件指向 Ollama：**
```bash
LM_STUDIO_URL=http://host.docker.internal:11434/v1 docker compose up -d
```

或使用 pip 运行中间件：
```bash
export LM_STUDIO_URL=http://localhost:11434/v1
llm-search
```

**3. 连接客户端** — 与 LM Studio 相同：`http://localhost:8000/v1`，模型 = `qwen3.6:27b`。

> **注意：** Ollama 模型的工具调用能力可能弱于 LM Studio 上的 Claude 蒸馏模型。请选择支持函数调用的模型。

---

### Claude Code（命令行）

[Claude Code](https://docs.anthropic.com/en/docs/claude-code) 通过 Anthropic Messages API 原生连接。中间件暴露 `/v1/messages` 端点，直接接受 Anthropic 格式的请求 — 无需转换层。

**1. 启动中间件**（参考上方的 LM Studio 或 Ollama 配置）。

**2. 设置环境变量：**
```bash
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_AUTH_TOKEN=not-needed
export CLAUDE_CODE_ATTRIBUTION_HEADER=0
```

**3. 像平常一样使用 Claude Code — 它现在可以搜索网页：**
```bash
claude "Go 最新版本是什么？"
claude "搜索当前的比特币价格并分析趋势"
claude "阅读 Python 3.14 发布说明并总结新特性"
```

搞定。Claude Code 发送 Anthropic 格式请求 → 中间件内部转换 → 运行工具循环 → 返回 Anthropic 格式响应，自动注入 `web_search` 和 `fetch_page`。

> 中间件会保留 Claude Code 发送的任何工具，与自动注入的搜索工具并存。

---

### Claude Desktop（MCP）

使用 MCP 服务器为 Claude Desktop 添加搜索能力：

**1. 安装 MCP 支持：**
```bash
pip install llm-search[mcp]
```

**2. 配置 Claude Desktop** — 添加到 `claude_desktop_config.json`（可通过 Claude Desktop → 设置 → 开发者 → 编辑配置打开，或直接编辑）：

| 操作系统 | 文件位置 |
|----------|----------|
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

**3. 重启 Claude Desktop** — `web_search` 和 `fetch_page` 工具将出现在工具列表中。

> MCP 服务器使用 stdio 传输。需要 SearXNG 运行（Docker 或独立）以及可选的 LM Studio。如果只需要 Claude Desktop 中的搜索工具，不需要 LM Studio — 只需要 SearXNG。

---

### VS Code Copilot Chat（BYOK）

GitHub Copilot Chat 可以将中间件作为自定义模型服务商使用。需要 VS Code 1.122+（目前为 Insiders 版本）以及任意 Copilot 订阅（免费版即可）。

**方式 A — 界面操作：** 命令面板 → `Chat: Manage Language Models` → **Add Models** → **Custom Endpoint**，然后输入 `http://localhost:8000/v1/chat/completions`，API 类型选 "Chat Completions"，API key 随便填。

**方式 B — 配置文件：** 在 VS Code 用户目录下创建 `chatLanguageModels.json`：

| 操作系统 | 文件位置 |
|----------|----------|
| Windows | `%APPDATA%\Code\User\chatLanguageModels.json` |
| macOS | `~/Library/Application Support/Code/User/chatLanguageModels.json` |
| Linux | `~/.config/Code/User/chatLanguageModels.json` |

> 使用 VS Code Insiders？将路径中的 `Code` 替换为 `Code - Insiders`。

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

该文件顶层是一个**数组** — 每个服务商组一个条目。编辑后运行 `Developer: Reload Window`，然后在聊天模型选择器的 Manage Models 列表中勾选一次该模型。如果模型仍未出现，可先用方式 A 让 VS Code 自动生成该文件，再对照检查。

将 `"id"` 设置为 LM Studio 中加载的模型 ID（或保留 `local-model` — LM Studio 会回退到已加载的模型）。之后 "LM Studio + Search" 就会出现在 Copilot Chat 的模型选择器中。

**如何设置 token 上限** — `maxInputTokens + maxOutputTokens` 必须小于模型的上下文窗口，还要为中间件在服务端追加的搜索结果和网页内容预留空间（Copilot 看不到这些内容，无法为其预算）。以 100k 上下文的模型为例：`maxInputTokens: 80000` + `maxOutputTokens: 8192`，为工具结果留出约 12k 的余量。另外请确认 LM Studio 中加载模型的上下文长度确实设置到了相应大小 — 无论模型支持多少，LM Studio 默认值都小得多。如果超出上限，中间件会返回上下文溢出错误，而不是静默截断。

> BYOK 模型仅用于**聊天** — 内联代码补全仍使用 GitHub 的模型。不支持 Copilot 的 **agent 模式**：中间件在调用 LLM 前会剥离客户端工具（见上文 Claude Code 说明），请使用 ask/chat 模式。

---

### OpenAI Codex 桌面应用

[Codex 桌面应用](https://openai.com/codex)（Windows / macOS）可以使用中间件作为自定义模型服务商。需要 ChatGPT 账号（免费版即可）和支持自定义 provider 的 Codex 版本。

**步骤 1 — 设置环境变量**（Codex 要求 `env_key`，即使是本地服务商也需要）：

| 操作系统 | 设置方法 |
|----------|----------|
| **Windows** | PowerShell 执行 `[System.Environment]::SetEnvironmentVariable('LLM_SEARCH_API_KEY', 'no-key-needed', 'User')`，然后注销重新登录 |
| **macOS** | `launchctl setenv LLM_SEARCH_API_KEY no-key-needed` 并重启 Codex |

**步骤 2 — 编辑 `~/.codex/config.toml`：**

| 操作系统 | 文件位置 |
|----------|----------|
| **Windows** | `C:\Users\<用户名>\.codex\config.toml` |
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

将 `model` 设置为 LM Studio 中加载的模型 ID（必须匹配 `/v1/models` 列表中的模型）。编辑后**重启 Codex** — 模型将出现在模型选择器中（由于 Codex Desktop 已知 UI 问题，可能显示为 "Custom"）。

**容量设置** — `model_context_window` 应与加载模型的上下文大小匹配。`model_auto_compact_token_limit` 在约 85% 上下文时触发自动压缩。为中间件服务端追加的搜索结果预留约 12k 的余量。

> **仅支持 chat/ask 模式** — 不支持 agent 模式。中间件在调用 LLM 前会剥离客户端工具（bash、read、write 等），仅保留 `web_search` 和 `fetch_page`。请使用 ask/chat 模式获取搜索增强的回答。

---

### Cursor / Continue.dev / Windsurf

这些 VS Code AI 扩展支持自定义 OpenAI 兼容服务商：

| 客户端 | 配置位置 | 设置 |
|--------|----------------|---------|
| **Cursor** | 设置 → 模型 → 添加模型 | Base URL: `http://localhost:8000/v1`, API Key: `not-needed` |
| **Continue.dev** | `~/.continue/config.json` | 模型条目下的 `"apiBase": "http://localhost:8000/v1"` |
| **Windsurf** | 设置 → AI 服务商 | 服务商: OpenAI, Base URL: `http://localhost:8000/v1` |

Continue.dev 完整模型配置示例：
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

然后在 Open WebUI 设置中添加新的 OpenAI 连接，指向 `http://localhost:8000/v1`。

---

### 直接 curl / API

```bash
# 非流式
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "local-model", "messages": [{"role": "user", "content": "Go 最新版本是什么？"}]}'

# 流式
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "local-model", "messages": [{"role": "user", "content": "Go 最新版本？"}], "stream": true}'
```

## 项目结构

| 文件 | 用途 |
|------|---------|
| `docker-compose.yml` | 一条命令启动 SearXNG + 中间件 |
| `Dockerfile` | 中间件容器构建 |
| `.github/workflows/publish.yml` | 推送 `v*` 标签时发布 Docker 镜像到 GHCR + Docker Hub |
| `searxng/settings.yml` | SearXNG 配置 — 无需修改 |
| `src/llm_search/server.py` | FastAPI 服务器 — `/v1/chat/completions`、`/v1/messages`、`/health`、`/stats` |
| `src/llm_search/tool_loop.py` | 工具调用拦截循环（非流式 + 流式） |
| `src/llm_search/tool_registry.py` | `web_search` + `fetch_page` 工具定义 |
| `src/llm_search/anthropic_adapter.py` | Anthropic Messages API ↔ OpenAI 格式适配 |
| `src/llm_search/mcp_server.py` | MCP 服务器 — 通过 stdio 暴露工具 |
| `src/llm_search/fetch_page.py` | URL 抓取器，HTML 转文本提取 |
| `.env.example` | 可选的环境变量配置 |

## 可选配置

一切开箱即用。如需自定义：

```bash
# 更改 LM Studio 地址（如果不是默认端口）
LM_STUDIO_URL=http://192.168.1.50:1234/v1 docker compose up

# 使用 Brave Search 替代 SearXNG
cp .env.example .env
# 编辑 .env: SEARCH_PROVIDER=brave, 添加 SEARCH_API_KEY=...
```

所有配置均通过环境变量设置 — 详见 `.env.example`。

## 协议

MIT
