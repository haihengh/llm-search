"""Diagnostic: does a client tool_use survive the Anthropic adapter intact?

Sends a Claude Code-shaped /v1/messages request with a `Read` tool and
reports exactly what the middleware streams back:
  - which tool names the model emitted
  - whether tool_use input arrived via input_json_delta (Anthropic spec)
    or only inside content_block_start (which SDKs ignore)
"""
import json
import sys

import httpx

URL = "http://localhost:8001/v1/messages"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3.8-27b-claude-opus-reasoning-distilled"

TOOLS = [
    {
        "name": "Read",
        "description": "Reads a file from the local filesystem.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute path"},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "Bash",
        "description": "Executes a bash command.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Command to run"},
                "description": {"type": "string"},
            },
            "required": ["command"],
        },
    },
]

body = {
    "model": MODEL,
    "max_tokens": 1000,
    "stream": True,
    "system": "You are a coding assistant. Use the Read tool to look at files.",
    "messages": [
        {
            "role": "user",
            "content": "Read the file C:\\Users\\hai\\Desktop\\code\\llm-search\\README.md "
            "and tell me what it says. Call the Read tool now.",
        }
    ],
    "tools": TOOLS,
}

events: list[tuple[str, dict]] = []
cur_event = None

with httpx.Client(timeout=300) as client:
    with client.stream("POST", URL, json=body) as r:
        print("HTTP", r.status_code)
        for line in r.iter_lines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("event: "):
                cur_event = line[7:]
            elif line.startswith("data: "):
                raw = line[6:]
                try:
                    events.append((cur_event or "?", json.loads(raw)))
                except json.JSONDecodeError:
                    events.append((cur_event or "?", {"_raw": raw}))

print(f"\n=== {len(events)} SSE events ===")
for name, data in events:
    print(f"[{name}] {json.dumps(data)[:300]}")

# ── Verdict ────────────────────────────────────────────────────
tool_use_blocks = [
    d["content_block"]
    for n, d in events
    if n == "content_block_start" and d.get("content_block", {}).get("type") == "tool_use"
]
input_json_deltas = [
    d for n, d in events
    if n == "content_block_delta" and d.get("delta", {}).get("type") == "input_json_delta"
]
stop_reasons = [d.get("delta", {}).get("stop_reason") for n, d in events if n == "message_delta"]

print("\n=== VERDICT ===")
print(f"tool_use blocks emitted : {len(tool_use_blocks)}")
for b in tool_use_blocks:
    print(f"   name={b.get('name')!r} input={json.dumps(b.get('input'))}")
print(f"input_json_delta events : {len(input_json_deltas)}  <-- MUST be > 0 for SDK clients")
print(f"stop_reason             : {stop_reasons}")
if tool_use_blocks and not input_json_deltas:
    print("\n>>> BUG CONFIRMED: tool_use input is only in content_block_start.")
    print(">>> Anthropic SDK clients (Claude Code) accumulate input from")
    print(">>> input_json_delta and will see input={} — the tool call fails.")
