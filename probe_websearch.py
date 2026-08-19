"""Probe: does web_search actually work end-to-end through the middleware
with the currently loaded LM Studio model?"""
import json
import httpx

URL = "http://localhost:8001/v1/messages"
MODEL = "qwen/qwen3.8-27b"

body = {
    "model": MODEL,
    "max_tokens": 2000,
    "stream": True,
    "system": "You are a helpful assistant with access to web search.",
    "messages": [
        {
            "role": "user",
            "content": "What is the current version number of LM Studio? Use web search to find out.",
        }
    ],
}

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

print(f"\n=== {len(events)} SSE events ===")
final_text = ""
saw_tool_use = False
for name, data in events:
    if name == "content_block_start" and data.get("content_block", {}).get("type") == "tool_use":
        saw_tool_use = True
        print("TOOL_USE:", data["content_block"].get("name"))
    if name == "content_block_delta" and data.get("delta", {}).get("type") == "text_delta":
        final_text += data["delta"]["text"]
    if name == "error":
        print("ERROR EVENT:", data)

print("\nsaw_tool_use (client-visible) :", saw_tool_use)
print("\n=== Final answer text ===")
print(final_text)
