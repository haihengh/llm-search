"""Isolate WHY the model stops calling client tools through the middleware.

Sends the same prompt directly to LM Studio with different tool sets:
  A: Read only                      (control)
  B: Read + Bash                    (client tools only)
  C: Read + Bash + web_search + fetch_page   (what the middleware sends)

Prints which tool the model chose in each case.
"""
import json
import sys

import httpx

URL = "http://localhost:1234/v1/chat/completions"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3.8-27b-claude-opus-reasoning-distilled"

READ = {
    "type": "function",
    "function": {
        "name": "Read",
        "description": "Reads a file from the local filesystem.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute path"},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
            },
            "required": ["file_path"],
        },
    },
}

BASH = {
    "type": "function",
    "function": {
        "name": "Bash",
        "description": "Executes a bash command.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Command to run"},
                "description": {"type": "string"},
            },
            "required": ["command"],
        },
    },
}

# Copied verbatim from src/llm_search/tool_registry.py
WEB_SEARCH = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the internet for current, up-to-date information. "
            "Use this whenever you need facts, news, or knowledge beyond "
            "your training cutoff date. "
            "IMPORTANT: After receiving search results, if you have enough "
            "information to answer the user's question, respond directly — "
            "do NOT call web_search again with the same or a similar query. "
            "Only search again if you need genuinely DIFFERENT information "
            "than what the results already provide."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (1-10)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}

FETCH_PAGE = {
    "type": "function",
    "function": {
        "name": "fetch_page",
        "description": (
            "Fetch the full text content of a web page by its URL. "
            "Use this after web_search to read a specific page in detail — "
            "for example, to get full release notes, documentation, or article text. "
            "Returns clean readable text (max ~8000 chars)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL of the page to fetch (must start with http:// or https://)",
                },
            },
            "required": ["url"],
        },
    },
}

MESSAGES = [
    {
        "role": "system",
        "content": "You are a coding assistant. Use the Read tool to look at files.",
    },
    {
        "role": "user",
        "content": (
            "Read the file C:\\Users\\hai\\Desktop\\code\\llm-search\\README.md "
            "and tell me what it says. Call the Read tool now."
        ),
    },
]

CASES = [
    ("A: Read only", [READ]),
    ("B: Read + Bash", [READ, BASH]),
    ("C: Read + Bash + web_search + fetch_page (middleware)", [READ, BASH, WEB_SEARCH, FETCH_PAGE]),
    ("D: Read + Bash + web_search + fetch_page, search tools FIRST", [WEB_SEARCH, FETCH_PAGE, READ, BASH]),
]

for label, tools in CASES:
    payload = {
        "model": MODEL,
        "messages": MESSAGES,
        "tools": tools,
        "stream": False,
        "max_tokens": 400,
    }
    with httpx.Client(timeout=300) as client:
        r = client.post(URL, json=payload)
        data = r.json()
    msg = data.get("choices", [{}])[0].get("message", {})
    tcs = msg.get("tool_calls") or []
    names = [t.get("function", {}).get("name") for t in tcs]
    text = (msg.get("content") or "").strip().replace("\n", " ")[:110]
    print(f"\n{label}")
    print(f"   tool_calls : {names if names else 'NONE'}")
    if tcs:
        print(f"   args       : {tcs[0].get('function', {}).get('arguments', '')[:120]}")
    if not tcs:
        print(f"   text       : {text!r}")
