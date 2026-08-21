"""Probe LM Studio's behavior when the prompt approaches / exceeds the loaded context window.

Sends three requests with growing filler prompts (~60k, ~103k, ~120k tokens)
and reports status code + finish_reason + content length for each, so we can
see exactly what the middleware receives when a long Claude Code session
fills the 104k-token context window.

Usage:  python scripts/probes/probe_context_boundary.py
"""
import sys

import httpx

URL = "http://localhost:1234/v1/chat/completions"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen/qwen3.8-27b"

# Roughly 4 chars per token for English filler text.
FILLER = "The quick brown fox jumps over the lazy dog. "  # 46 chars ≈ 11 tokens


def probe(label: str, filler_chars: int) -> None:
    filler = (FILLER * (filler_chars // len(FILLER) + 1))[:filler_chars]
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": filler},
            {"role": "user", "content": "Reply with exactly: OK"},
        ],
        "max_tokens": 16,
        "stream": False,
        "temperature": 0,
    }
    print(f"\n=== {label}: ~{filler_chars // 4:,} filler tokens + short question ===")
    try:
        with httpx.Client(timeout=600.0) as client:
            resp = client.post(URL, json=body)
        print(f"HTTP {resp.status_code}")
        data = resp.json()
        if resp.status_code == 200:
            choice = data["choices"][0]
            content = choice["message"].get("content") or ""
            print(
                f"finish_reason={choice.get('finish_reason')!r} "
                f"content_len={len(content)} content={content[:80]!r}"
            )
            usage = data.get("usage") or {}
            print(f"usage={usage}")
        else:
            print(f"body: {data}")
    except Exception as exc:  # noqa: BLE001
        print(f"EXCEPTION: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    probe("fits comfortably", 240_000)   # ~60k tokens
    probe("near limit", 410_000)         # ~103k tokens
    probe("over limit", 480_000)         # ~120k tokens
