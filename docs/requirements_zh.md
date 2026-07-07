# 环境要求与依赖

## 概览

系统由两部分组成：
1. **Docker 端**（SearXNG + 中间件）— 通过 `docker compose up` 运行
2. **主机端**（LM Studio / Ollama）— 原生运行以访问 GPU

---

## 前置条件

### 1. Docker Desktop（或 Docker Engine + Compose）

- **macOS**: [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- **Windows**: [Docker Desktop](https://www.docker.com/products/docker-desktop/)（推荐 WSL2 后端）
- **Linux**: Docker Engine + Compose 插件（`apt install docker.io docker-compose-v2`）

验证：
```bash
docker --version         # 24.0+
docker compose version   # 2.0+
```

### 2. LM Studio 或 Ollama（在主机上，不在 Docker 中）

- **LM Studio**: 0.3.0 或更高版本（需要可靠的工具/函数调用支持）
  - 下载：https://lmstudio.ai/
  - 配置：下载并加载**支持工具调用的模型**（见下方）
  - 验证：LM Studio 应运行在 `http://localhost:1234`

- **Ollama**: 轻量替代方案
  - 下载：https://ollama.com/
  - 暴露 OpenAI 兼容 API 在 11434 端口

### 3. 兼容模型

模型必须支持 OpenAI 格式的函数/工具调用。并非所有模型都支持 — 以下是我们的实际测试结果：

**✅ 可用 — 已与该中间件端到端测试：**

| 模型 | 大小 | 备注 |
|-------|------|-------|
| `qwythos-9b-claude-mythos-5-1m` | 9B | Claude 蒸馏，快速可靠 |
| `qwen3.6-27b-claude-mythos-distilled-mtp` | 27B | Claude 蒸馏，回答详细 |
| `qwopus3.6-27b-v2-mtp` | 27B | Opus 蒸馏，回答全面 |
| `qwen3.6-27b-uncensored-heretic-v2-native-mtp-preserved` | 27B | "Native" 变体保留了工具调用 |
| `gemma-4-31b-it-qat` | 31B | Google Gemma IT，工具使用稳定 |

**❌ 不可用 — 循环搜索不回答：**

| 模型 | 大小 | 原因 |
|-------|------|--------|
| `qwen3.6-27b` | 27B | 无 OpenAI 函数调用支持 |
| `qwen3.6-27b@q4_k_m` | 27B Q4 | 无工具调用（量化可能移除了该能力） |
| `qwen3.6-35b-a3b` | 35B | 无工具调用 |

**⏱️ 未测试（超时 — 可能可用但模型太慢）：**

| 模型 | 大小 |
|-------|------|
| `qwen3.6-27b-mtp` | 31B |

**规律**：Claude/Opus 蒸馏版 Qwen 模型和 Google Gemma IT 模型能可靠处理工具调用。Qwen 基础模型不支持 — 它们缺少 OpenAI 函数调用训练。Qwen 变体中注意选择带有 "mythos"、"opus"、"claude-distilled" 或 "native" 标识的。Gemma 使用 "it"（指令微调）变体。

**推荐**：`qwythos-9b-claude-mythos-5-1m` 是最佳权衡 — 小巧、快速、工具调用可靠。

---

## 你不需要的东西

| 项目 | 为什么不需要 |
|-------|----------------------|
| ❌ **OpenAI API Key** | LLM 在 LM Studio 中本地运行 |
| ❌ **Brave/Google/SerpAPI Key** | SearXNG 匿名搜索 — 无需 API Key |
| ❌ **GPU** | LM Studio 可用 CPU 运行（较慢），但强烈推荐 GPU |
| ❌ **云账号** | 一切在你的机器上运行 |
| ❌ **Python 环境** | 中间件在 Docker 中运行 |
| ❌ **Node.js / npm** | 该技术栈无 JavaScript |
| ❌ **向量数据库** | 非 RAG 方案 — 这是实时搜索 |

---

## 硬件建议

| 组件 | 最低配置 | 推荐配置 | 备注 |
|-----------|---------|-------------|-------|
| 内存 | 8 GB | 16 GB+ | 8 GB 模型 + 1-2 GB Docker |
| 显存 | 4 GB（3B 模型） | 8 GB+（7-8B 模型） | 仅 CPU 可无需（但很慢） |
| 磁盘 | 5 GB 空闲 | 15 GB 空闲 | 模型每个 2-7 GB |
| CPU | x86_64 或 Apple Silicon | Apple M1+ 或现代 x86_64 | |

---

## 网络架构

```
┌─────────────────────────────────────────────────┐
│                    你的机器                      │
│                                                  │
│  Docker（Linux VM 或原生）                        │
│  ┌───────────────────────────────────────────┐  │
│  │  searxng        llm-search                │  │
│  │  :8080 ───────▶ :8000                     │  │
│  │  (内部)         (主机端口 8000)            │  │
│  └────────────────┬──────────────────────────┘  │
│                   │                              │
│          host.docker.internal:1234               │
│                   │                              │
│  ┌────────────────▼──────────────────────────┐  │
│  │           LM Studio（原生应用）             │  │
│  │           :1234 (仅 localhost)             │  │
│  └───────────────────────────────────────────┘  │
│                                                  │
│  聊天客户端连接到 localhost:8000                  │
└─────────────────────────────────────────────────┘
```

**关键点：**
- `llm-search` 通过 `host.docker.internal:1234`（Docker 访问主机的特殊主机名）连接 LM Studio
- SearXNG **不**暴露到主机 — 只有 `llm-search` 能访问它
- 客户端（聊天 UI、curl、应用）访问 `localhost:8000`
- LM Studio 仅监听 localhost（无安全风险）

---

## Docker Compose 快速参考

```bash
# 启动
docker compose up -d

# 查看日志
docker compose logs -f

# 检查状态
docker compose ps

# 配置变更后重启
docker compose restart

# 停止
docker compose down

# 完全重置（清除缓存，重新开始）
docker compose down -v
```

---

## 常见问题排查

### "LM Studio 不可访问"

LM Studio 未运行或不在 1234 端口。
```bash
# 检查 LM Studio 是否在监听
curl http://localhost:1234/v1/models
```
如果失败，启动 LM Studio 并加载模型。

### "搜索引擎不可用"

SearXNG 容器不健康。
```bash
# 直接检查 SearXNG
docker compose exec searxng curl http://localhost:8080/search?q=test&format=json
```

### "工具调用循环超限"

LLM 持续搜索但不生成答案。这通常发生在较弱的模型上。尝试：
- 增加 `.env` 中的 `MAX_TOOL_LOOP_ITERATIONS=8`
- 使用更强的模型
- 简化你的提示词

### "Docker 无法通过 host.docker.internal 访问 LM Studio"

该主机名在 Docker Desktop（macOS/Windows）上可用。在 Linux 上可能需要：
```bash
# 在 docker-compose.yml 中已包含：
extra_hosts:
  - "host.docker.internal:host-gateway"
# 或使用：LM_STUDIO_URL=http://172.17.0.1:1234/v1
```
或设置 `LM_STUDIO_URL=http://172.17.0.1:1234/v1`（Linux 上 Docker 访问主机的默认网关）。

### "Ollama 连接失败"

确保在启动中间件时设置了正确的 `LM_STUDIO_URL`：
```bash
LM_STUDIO_URL=http://host.docker.internal:11434/v1 docker compose up -d
```

### 上游搜索引擎频率限制（罕见）

SearXNG 查询 Google、Bing、DuckDuckGo 等。每个引擎有其自身的频率限制。在快速连续使用的情况下（如在短时间内测试大量搜索），部分引擎可能会被暂时挂起：

- **DuckDuckGo**：测试时常遇到 CAPTCHA — 几分钟内自动恢复
- **Google**：高负载下可能挂起 180 秒 — 自动恢复
- **Brave**：同样 180 秒挂起

SearXNG 会自动处理：挂起的引擎被跳过，其他引擎填补空缺。挂起期间结果可能减少，但搜索仍然可用。正常单用户使用很少触发此问题 — 我们仅在批量测试时遇到。

如果持续发生：
- 等待 3-5 分钟让挂起过期
- 中间件健康检查（每 30 秒）使用轻量 `/healthz` 端点，不计入搜索引擎配额
- 考虑添加搜索 API Key 作为备份：`SEARCH_PROVIDER=brave` 配合 `SEARCH_API_KEY=...`
- 在 `searxng/settings.yml` 中禁用有问题的引擎：
  ```yaml
  engines:
    - name: duckduckgo
      disabled: true
  ```
