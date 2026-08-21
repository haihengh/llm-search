"""Verify the middleware surfaces context exhaustion as HTTP 400, not fake text.

Three cases through the middleware (port 8001) /v1/messages, stream=True:

  A. Over limit — prompt exceeds LM Studio's loaded context. LM Studio
     400s with "exceeds the available context size"; the middleware must
     classify it as context overflow and return HTTP 400 "prompt is too long".

  B. Near limit + tiny max_tokens — the model burns its whole generation
     budget on reasoning and finishes with finish_reason=length and zero
     content. The middleware must turn this into HTTP 400 "prompt is too
     long" (first SSE event is a context_overflow error) instead of
     streaming "The model returned an empty response...".

  C. Sanity — a short request still streams a normal 200 with content.

Usage:  python scripts/probes/probe_middleware_overflow.py
"""
import sys

import httpx

URL = "http://localhost:8001/v1/messages"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "deepseek-v4-pro"

# Roughly 4 chars per token for English filler text.
FILLER = "The quick brown fox jumps over the lazy dog. "  # 46 chars ≈ 11 tokens


def filler(chars: int) -> str:
    return (FILLER * (chars // len(FILLER) + 1))[:chars]


def probe(label: str, system_chars: int, max_tokens: int) -> None:
    body = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "stream": True,
        "system": filler(system_chars),
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "tools": [{
            "name": "web_search",
            "description": "Search the web.",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }],
    }
    print(f"\n=== {label} ===")
    with httpx.Client(timeout=600.0) as client:
        with client.stream("POST", URL, json=body) as resp:
            status = resp.status_code
            first_lines = []
            for i, line in enumerate(resp.iter_lines()):
                if line:
                    first_lines.append(line[:200])
                if i >= 40:
                    break
            print(f"HTTP {status}")
            for line in first_lines[:8]:
                print(f"  {line}")


if __name__ == "__main__":
    probe("A. over limit (~120k tokens)", 480_000, 1024)
    probe("B. near limit, tiny max_tokens (~90k tokens)", 360_000, 16)
    probe("C. sanity (short prompt)", 0, 256)
