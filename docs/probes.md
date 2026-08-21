# Tool-Calling Diagnostic Probes

`scripts/probes/` contains standalone scripts for manually verifying that
tool calling and web search work correctly against a running middleware +
LM Studio setup. They were built while investigating reports of Claude Code
(and other Anthropic-API clients) failing or hallucinating during tool use,
and are meant to be reused whenever:

- you load a new/different model in LM Studio and want to confirm it
  handles tool calls correctly before relying on it,
- you change anything in `anthropic_adapter.py`, `tool_loop.py`, or
  `tool_registry.py`,
- a user reports "issues with tool calls" and you need to reproduce it.

They are plain scripts, not part of the `pytest` suite — run them directly
with `python`. All of them talk to the middleware over HTTP, so the stack
must be running first:

```bash
docker compose up -d
```

## Prerequisites

- The middleware container reachable at `http://localhost:8001` (adjust the
  `URL` constant at the top of each script if your setup differs).
- LM Studio running with a model loaded that has tool-use capability.

## Probes

### `tool_use_stream.py` — single-tool streaming sanity check

Sends one Claude Code-style `/v1/messages` request with a `Read` tool and
inspects the raw SSE stream. Confirms:

- a `tool_use` content block is emitted,
- its arguments arrive via `input_json_delta` chunks (the Anthropic spec —
  SDK clients ignore anything only present in `content_block_start`),
- `stop_reason` is `tool_use`.

```bash
python scripts/probes/tool_use_stream.py [model-id]
```

Fastest check to run first against a newly loaded model. Prints a `VERDICT`
section and flags the specific bug it was built to catch (empty `input={}`
with zero `input_json_delta` events).

### `tool_selection.py` — tool selection under a realistic tool set

Runs the same request under four scenarios: `Read` only, `Read` + `Bash`,
the full Claude Code + middleware set (`Read`, `Bash`, `web_search`,
`fetch_page`), and the same set with the search tools listed first. Confirms
the model still picks the correct tool (`Read`) with valid arguments
regardless of how many other tools are offered or their order — this is
where models can get confused and call the wrong tool or hallucinate one.

```bash
python scripts/probes/tool_selection.py [model-id]
```

### `multiturn_roundtrip.py` — full tool_use → tool_result → answer cycle

Sends a request, parses the resulting `tool_use` block, then manually
constructs the follow-up turn (assistant `tool_use` + user `tool_result`)
and sends it back, checking that the model produces a correct final text
answer. This is the scenario most likely to expose reasoning-leak bugs
(private chain-of-thought bleeding into the visible answer) or broken
conversation-state handling, since those only surface after a tool result
is fed back in.

```bash
python scripts/probes/multiturn_roundtrip.py [model-id]
```

### `web_search.py` — end-to-end internet search verification

Asks a question that requires current information and lets the middleware's
built-in `web_search` / `fetch_page` tools run server-side (these never
surface to the Anthropic client as visible `tool_use` blocks — see note
below). Do not trust the model's answer text alone; cross-check the
middleware's container logs to confirm a real search actually happened:

```bash
python scripts/probes/web_search.py [model-id]
docker logs llm-search-middleware --since 2m | Select-String "Tool call:|searxng"
```

You want to see something like:

```
Tool call: web_search({'query': '...'})
GET http://searxng:8080/search?q=... "HTTP/1.1 200 OK"
Tool call: fetch_page({'url': '...'})
```

If those log lines are missing, the model answered from its own training
data instead of actually searching, even if the final text looks plausible.

### `probe_context_boundary.py` — raw LM Studio behavior at the context wall

Sends three non-streaming requests with growing filler prompts (~60k,
~103k, ~120k tokens) straight to LM Studio (bypassing the middleware) and
reports status code + `finish_reason` + content length for each. Shows
exactly what the middleware receives when a long Claude Code session fills
the loaded model's context window — including the streaming quirk where the
failure arrives as an `event: error` SSE chunk with HTTP 200.

```bash
python scripts/probes/probe_context_boundary.py [model-id]
```

### `probe_middleware_overflow.py` — middleware behavior at the context wall

Three cases through the middleware (port 8001) `/v1/messages`, stream=True:

- **A. Over limit** — prompt exceeds LM Studio's loaded context. LM Studio
  400s with "exceeds the available context size"; the middleware must
  classify it as context overflow and return HTTP 400 `prompt is too long`.
- **B. Near limit + tiny max_tokens** — the model burns its whole
  generation budget on reasoning and finishes with `finish_reason=length`
  and zero content. The middleware must return HTTP 400 `prompt is too
  long` (first SSE event is a `context_overflow` error) instead of
  streaming "The model returned an empty response...".
- **C. Sanity** — a short request still streams a normal 200 with content.

```bash
python scripts/probes/probe_middleware_overflow.py [model-id]
```

Rerun after changing context-overflow handling in `tool_loop.py`,
`anthropic_adapter.py`, or `server.py`.

## Notes

- All scripts default to a hardcoded fallback model id if you don't
  pass one — always pass the model id you actually have loaded in LM
  Studio, e.g. `python scripts/probes/tool_use_stream.py qwen/qwen3.8-27b`.
- `web_search` / `fetch_page` are executed inside the middleware and are
  invisible to the Anthropic-adapter client stream by design — only
  client-supplied tools (`Read`, `Bash`, etc.) pass through as `tool_use`
  blocks. A probe reporting no client-visible `tool_use` during a search
  question is expected, not a bug; check the container logs instead.
- See `CHANGELOG.md` (`[0.3.2]`, `[0.3.3]`) for the investigations that
  produced these probes and the bugs they helped find and fix.
