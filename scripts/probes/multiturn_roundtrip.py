"""Multi-turn probe: tool_use -> tool_result -> follow-up answer, through the middleware."""
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
            },
            "required": ["file_path"],
        },
    },
]

messages = [
    {"role": "user", "content": "Read the file C:\\Users\\hai\\Desktop\\code\\llm-search\\README.md and summarize it in one sentence. Call the Read tool now."},
]

body = {
    "model": MODEL,
    "max_tokens": 1000,
    "stream": True,
    "system": "You are a coding assistant.",
    "messages": messages,
    "tools": TOOLS,
}

def run(body):
    events = []
    cur = None
    with httpx.Client(timeout=180) as client:
        with client.stream("POST", URL, json=body) as r:
            print("HTTP", r.status_code)
            for line in r.iter_lines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("event: "):
                    cur = line[7:]
                elif line.startswith("data: "):
                    raw = line[6:]
                    try:
                        events.append((cur, json.loads(raw)))
                    except json.JSONDecodeError:
                        events.append((cur, {"_raw": raw}))
    return events

print("=== Turn 1 ===")
events = run(body)
tool_use = None
for name, data in events:
    print(name, json.dumps(data)[:200])
    if name == "content_block_start" and data.get("content_block", {}).get("type") == "tool_use":
        tool_use = data["content_block"]
    if name == "content_block_delta" and data.get("delta", {}).get("type") == "input_json_delta":
        if tool_use:
            tool_use.setdefault("_args", "")
            tool_use["_args"] += data["delta"]["partial_json"]

if not tool_use:
    print("NO TOOL USE EMITTED - stopping")
    raise SystemExit(1)

print("\nParsed tool_use:", tool_use["name"], tool_use.get("_args"))

messages.append({
    "role": "assistant",
    "content": [{"type": "tool_use", "id": tool_use["id"], "name": tool_use["name"], "input": json.loads(tool_use.get("_args", "{}"))}],
})
messages.append({
    "role": "user",
    "content": [{"type": "tool_result", "tool_use_id": tool_use["id"], "content": "# llm-search\nA middleware that adds web search tool-calling to local LLMs served via LM Studio."}],
})

body["messages"] = messages

print("\n=== Turn 2 (after tool_result) ===")
events2 = run(body)
final_text = ""
for name, data in events2:
    print(name, json.dumps(data)[:300])
    if name == "content_block_delta" and data.get("delta", {}).get("type") == "text_delta":
        final_text += data["delta"]["text"]

print("\n=== Final answer text ===")
print(final_text)
